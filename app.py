import streamlit as st
import pandas as pd
import numpy as np
import base64
import google.generativeai as genai
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
from PIL import Image
from streamlit_mic_recorder import mic_recorder

# =========================
# 1. 초기 설정 및 세션 관리
# =========================
st.set_page_config(page_title="Med-Study AI v1.0 (PRO)", layout="wide", page_icon="🩺")

# 세션 상태 초기화 (NameError 방지)
keys = ['jokbo_done', 'lecture_done', 'exam_db', 'exam_embeddings', 'pre_analysis', 'pdf_bytes', 'total_pages', 'notebook', 'ai_cache']
for key in keys:
    if key not in st.session_state:
        if key in ['exam_db', 'pre_analysis', 'notebook']: st.session_state[key] = []
        elif key == 'ai_cache': st.session_state[key] = {}
        elif key in ['jokbo_done', 'lecture_done']: st.session_state[key] = False
        else: st.session_state[key] = None

with st.sidebar:
    st.title("⚙️ 시스템 설정")
    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        st.success("✅ 유료 엔진 연결됨")
    
    if st.button("🔄 전체 초기화"):
        for key in keys: st.session_state[key] = {} if key == 'ai_cache' else ([] if key in ['exam_db', 'pre_analysis', 'notebook'] else None)
        st.rerun()

# --- 핵심 엔진 함수 ---
def get_embedding(text):
    if not api_key: return None
    try:
        # Embedding API는 할당량이 매우 커서 에러 없이 대량 처리가 가능함
        result = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")
        return result['embedding']
    except: return None

def display_pdf_page(file_bytes, page_num):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    st.image(img, use_container_width=True)

def analyze_on_demand(lec_text, jokbo_text, page_key):
    """유료 버전의 속도를 활용, 사용자가 클릭할 때만 상세 분석 수행"""
    if page_key in st.session_state.ai_cache:
        return st.session_state.ai_cache[page_key]
    
    model = genai.GenerativeModel("gemini-1.5-flash") # 속도와 가성비가 가장 뛰어난 모델
    prompt = f"강의록: {lec_text[:800]}\n족보: {jokbo_text[:800]}\n이 족보가 이 페이지와 어떻게 연결되는지 의대생 조교로서 핵심만 한 줄 요약해줘."
    
    try:
        response = model.generate_content(prompt)
        st.session_state.ai_cache[page_key] = response.text
        return response.text
    except Exception as e:
        return f"분석 중 오류 발생 (결제 수단 및 한도 확인 필요): {e}"

# =========================
# 2. 메인 UI
# =========================
tab1, tab2, tab3 = st.tabs(["📂 1. 데이터 준비", "🎙️ 2. 수업 중 (Live)", "🎯 3. 복습 리스트"])

# --- [Tab 1: 데이터 학습] ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. 족보 아카이브 학습")
        exam_files = st.file_uploader("족보 PDF 업로드", type="pdf", accept_multiple_files=True)
        if not st.session_state.jokbo_done and st.button("🚀 딥러닝 인덱싱 시작"):
            all_exams, embeddings = [], []
            bar = st.progress(0)
            for idx, f in enumerate(exam_files):
                pages = [p.extract_text() for p in PdfReader(f).pages]
                for i, text in enumerate(pages):
                    if len(text) > 30:
                        emb = get_embedding(text)
                        if emb:
                            all_exams.append({"info": f"{f.name} (p.{i+1})", "text": text})
                            embeddings.append(emb)
                bar.progress((idx + 1) / len(exam_files))
            st.session_state.exam_db, st.session_state.exam_embeddings = all_exams, np.array(embeddings)
            st.session_state.jokbo_done = True
            st.rerun()

    with col2:
        st.subheader("2. 강의록 사전 분석")
        lec_file = st.file_uploader("오늘 강의 PDF", type="pdf")
        if lec_file and not st.session_state.lecture_done:
            st.session_state.pdf_bytes = lec_file.getvalue()
            st.session_state.total_pages = len(PdfReader(lec_file).pages)
            if st.button("🔍 강의-족보 정밀 대조"):
                results = []
                lec_pages = [p.extract_text() for p in PdfReader(lec_file).pages]
                bar2 = st.progress(0)
                for i, p_text in enumerate(lec_pages):
                    if len(p_text) < 30: continue
                    q_emb = get_embedding(p_text)
                    if q_emb is not None:
                        sims = cosine_similarity([q_emb], st.session_state.exam_embeddings).flatten()
                        if sims.max() > 0.45:
                            best_idx = sims.argmax()
                            results.append({
                                "page": i+1, "score": sims.max(),
                                "exam_info": st.session_state.exam_db[best_idx]['info'],
                                "exam_text": st.session_state.exam_db[best_idx]['text'],
                                "lec_text": p_text
                            })
                    bar2.progress((i+1)/len(lec_pages))
                st.session_state.pre_analysis = results
                st.session_state.lecture_done = True
                st.rerun()

# --- [Tab 2: 수업 중 뷰어 & 실시간 저장] ---
with tab2:
    if st.session_state.pdf_bytes:
        page_num = st.slider("페이지 이동", 1, st.session_state.total_pages, 1)
        c_pdf, c_tool = st.columns([1.2, 0.8])
        
        with c_pdf:
            display_pdf_page(st.session_state.pdf_bytes, page_num)
            
        with c_tool:
            st.subheader("🎙️ 실시간 보이스 트래킹")
            mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="⏹️ 분석", key='mic')
            st.divider()

            st.subheader(f"📍 {page_num}p 기출 분석")
            matches = [r for r in st.session_state.pre_analysis if r['page'] == page_num]
            if matches:
                for idx, m in enumerate(matches):
                    with st.expander(f"🔥 매칭률 {int(m['score']*100)}% - {m['exam_info']}", expanded=True):
                        # 유료 버전의 속도를 체감할 수 있는 On-Demand 분석 버튼
                        if st.button(f"🤖 AI 상세 분석 요청", key=f"ai_{page_num}_{idx}"):
                            with st.spinner("프로 버전 분석 중..."):
                                st.write(analyze_on_demand(m['lec_text'], m['exam_text'], f"{page_num}_{idx}"))
                        
                        st.markdown(f"> **족보 원문:** {m['exam_text'][:200]}...")
                        user_memo = st.text_input("메모", key=f"memo_{page_num}_{idx}")
                        if st.button("📌 정리본 저장", key=f"btn_{page_num}_{idx}"):
                            st.session_state.notebook.append({"page": page_num, "exam": m['exam_text'], "note": user_memo})
                            st.toast("저장 완료!")
            else: st.info("이 페이지는 족보 연관성이 낮습니다.")
    else: st.warning("Tab 1에서 분석을 완료해주세요.")
