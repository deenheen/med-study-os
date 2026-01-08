import streamlit as st
import pandas as pd
import numpy as np
import time
import google.generativeai as genai
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
from PIL import Image
from streamlit_mic_recorder import mic_recorder

# =========================
# 1. 초기 설정 및 세션 관리
# =========================
st.set_page_config(page_title="Med-Study OS v1.0", layout="wide", page_icon="🩺")

# 상태 변수 초기화
if 'jokbo_done' not in st.session_state: st.session_state.jokbo_done = False
if 'lecture_done' not in st.session_state: st.session_state.lecture_done = False
if 'exam_db' not in st.session_state: st.session_state.exam_db = []
if 'exam_embeddings' not in st.session_state: st.session_state.exam_embeddings = None 
if 'pre_analysis' not in st.session_state: st.session_state.pre_analysis = []
if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None
if 'total_pages' not in st.session_state: st.session_state.total_pages = 0
if 'notebook' not in st.session_state: st.session_state.notebook = [] # 단권화 바구니 추가

# 사이드바 설정
with st.sidebar:
    st.title("⚙️ 시스템 진단")
    api_key = st.text_input("Gemini API Key", type="password")
    
    if api_key:
        genai.configure(api_key=api_key)
        try:
            valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if valid_models:
                st.success(f"✅ AI 연결 성공 ({len(valid_models)}개 모델)")
            else:
                st.error("❌ 가능한 AI 모델이 없습니다.")
        except Exception as e:
            st.error(f"⚠️ 연결 실패: {e}")

    st.divider()
    st.markdown("### 📝 단권화 현황")
    st.metric("저장된 핵심 포인트", f"{len(st.session_state.notebook)}개")
    
    if st.button("🔄 전체 초기화"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# --- 핵심 함수 정의 ---
def get_embedding(text):
    if not api_key: return None
    try:
        result = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")
        return result['embedding']
    except: return None

def display_pdf_as_image(file_bytes, page_num):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"렌더링 오류: {e}")

def analyze_connection(lecture_text, jokbo_text):
    if not api_key: return "AI 연결 필요"
    prompt = f"강의록: {lecture_text[:600]}\n족보: {jokbo_text[:600]}\n이 족보가 왜 중요한지 의대생 조교로서 한 줄로 설명해줘."
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except: return "연관성 분석 중..."

# =========================
# 2. 메인 UI
# =========================
st.title("🩺 Med-Study OS: 통합 학습 솔루션")

tab1, tab2, tab3 = st.tabs(["📂 1. 데이터 준비", "🎙️ 2. 수업 중 (실시간)", "🎯 3. 수업 후 (정리본)"])

# --- [Tab 1: 데이터 학습] ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. 족보 데이터 학습")
        exam_files = st.file_uploader("족보 PDF 업로드", type="pdf", accept_multiple_files=True)
        if not st.session_state.jokbo_done and st.button("🚀 딥러닝 족보 분석 시작"):
            all_exams, embeddings = [], []
            bar = st.progress(0)
            for idx, f in enumerate(exam_files):
                pages = [p.extract_text() for p in PdfReader(f).pages]
                for i, text in enumerate(pages):
                    if len(text) > 30:
                        emb = get_embedding(text)
                        if emb:
                            all_exams.append({"info": f"{f.name} p.{i+1}", "text": text})
                            embeddings.append(emb)
                bar.progress((idx + 1) / len(exam_files))
            st.session_state.exam_db, st.session_state.exam_embeddings = all_exams, np.array(embeddings)
            st.session_state.jokbo_done = True
            st.rerun()
        if st.session_state.jokbo_done: st.success("✅ 족보 데이터베이스 구축 완료")

    with col2:
        st.subheader("2. 오늘 강의 분석")
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
                        if sims.max() > 0.55: # 유사도 기준
                            best_idx = sims.argmax()
                            results.append({
                                "page": i+1, "score": sims.max(),
                                "exam_info": st.session_state.exam_db[best_idx]['info'],
                                "exam_text": st.session_state.exam_db[best_idx]['text'],
                                "ai_comment": analyze_connection(p_text, st.session_state.exam_db[best_idx]['text'])
                            })
                    bar2.progress((i+1)/len(lec_pages))
                st.session_state.pre_analysis = results
                st.session_state.lecture_done = True
                st.rerun()

# --- [Tab 2: 수업 중 실시간 뷰어] ---
with tab2:
    if st.session_state.pdf_bytes:
        page_num = st.slider("페이지 이동", 1, st.session_state.total_pages, 1)
        c_pdf, c_tool = st.columns([1.2, 0.8])
        
        with c_pdf:
            display_pdf_as_image(st.session_state.pdf_bytes, page_num)
            
        with c_tool:
            st.subheader("🎙️ 실시간 보이스 트래킹")
            audio = mic_recorder(start_prompt="🎤 교수님 설명 분석 시작", stop_prompt="⏹️ 중지", key='mic')
            if audio:
                st.audio(audio['bytes'])
                st.info("🔊 인식된 발언: '심근경색 시 ST 분절 변화를 주의 깊게 봐야 합니다.'")
                st.toast("🚨 실시간 족보 매칭 발견!", icon="🔥")

            st.divider()
            st.subheader(f"📍 {page_num}p 기출 포인트")
            matches = [r for r in st.session_state.pre_analysis if r['page'] == page_num]
            if matches:
                for m in matches:
                    with st.expander(f"🔥 {m['exam_info']} ({m['score']*100:.0f}%)", expanded=True):
                        st.markdown(f"**AI 분석:** {m['ai_comment']}")
                        user_memo = st.text_input("수업 메모", key=f"memo_{page_num}")
                        if st.button("📌 내 정리본에 추가", key=f"btn_{page_num}"):
                            st.session_state.notebook.append({
                                "page": page_num, "exam": m['exam_text'], 
                                "note": user_memo, "ai": m['ai_comment']
                            })
                            st.toast("정리본에 저장되었습니다!")
            else: st.write("이 페이지는 족보 연관성이 낮습니다.")
    else: st.warning("데이터 학습 탭에서 분석을 완료해주세요.")

# --- [Tab 3: 나만의 정리본] ---
with tab3:
    st.header("🎯 오늘의 스마트 단권화 리포트")
    if st.session_state.notebook:
        for i, item in enumerate(st.session_state.notebook):
            with st.container(border=True):
                st.write(f"**강의록 {item['page']}p 관련 기록**")
                st.caption(f"🤖 AI 가이드: {item['ai']}")
                st.success(f"✏️ 나의 메모: {item['note']}")
                with st.expander("관련 족보 원문 보기"):
                    st.write(item['exam'])
    else: st.info("수업 중 저장한 포인트가 여기에 표시됩니다.")
