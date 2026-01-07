import streamlit as st
import pandas as pd
import numpy as np
import time
import google.generativeai as genai
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
# from streamlit_mic_recorder import mic_recorder # 마이크 기능은 일단 주석 처리 (필요시 해제)
import fitz  # PyMuPDF (PDF 렌더링용 필수)
from PIL import Image # 이미지 처리용

# =========================
# 1. 초기 설정 및 세션 관리
# =========================
st.set_page_config(page_title="Med-Study OS v0.5", layout="wide", page_icon="🩺")

# 상태 변수 초기화
if 'jokbo_done' not in st.session_state: st.session_state.jokbo_done = False
if 'lecture_done' not in st.session_state: st.session_state.lecture_done = False
if 'exam_db' not in st.session_state: st.session_state.exam_db = []
if 'exam_embeddings' not in st.session_state: st.session_state.exam_embeddings = None 
if 'pre_analysis' not in st.session_state: st.session_state.pre_analysis = []
if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None
if 'total_pages' not in st.session_state: st.session_state.total_pages = 0

# 사이드바 설정
with st.sidebar:
    st.title("⚙️ 설정")
    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        st.success("✅ AI 연결됨")
    
    st.divider()
    st.markdown("### 상태 모니터")
    if st.session_state.jokbo_done:
        st.success("족보 학습 완료")
    else:
        st.warning("족보 학습 대기 중")
        
    if st.session_state.lecture_done:
        st.success("강의 분석 완료")
    else:
        st.warning("강의 분석 대기 중")
    
    if st.button("🔄 전체 초기화"):
        st.session_state.jokbo_done = False
        st.session_state.lecture_done = False
        st.session_state.exam_embeddings = None
        st.session_state.pdf_bytes = None
        st.session_state.pre_analysis = []
        st.rerun()

# --- 함수 정의 ---
def get_embedding(text):
    if not api_key: return None
    try:
        # 모델명은 최신 버전에 맞게 수정될 수 있음
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"임베딩 오류: {e}")
        return None

def get_pdf_text(file):
    reader = PdfReader(file)
    return [page.extract_text() or "" for page in reader.pages]

# [수정됨] PyMuPDF를 사용하여 PDF 페이지를 이미지로 변환해 보여주는 함수
def display_pdf_as_image(file_bytes, page_num):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        # page_num은 1부터 시작하므로 인덱스는 -1 해줘야 함
        page_idx = page_num - 1
        
        if 0 <= page_idx < len(doc):
            page = doc.load_page(page_idx)
            
            # 해상도 높이기 (zoom=2) -> 글씨가 선명해짐
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            
            # PIL 이미지로 변환
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Streamlit 이미지로 출력
            st.image(img, use_container_width=True)
        else:
            st.error("페이지 범위를 벗어났습니다.")
    except Exception as e:
        st.error(f"PDF 렌더링 오류: {e}")

# =========================
# 2. 메인 UI
# =========================
st.title("🩺 Med-Study OS: 스마트 뷰어")

tab1, tab2 = st.tabs(["📂 데이터 학습 (준비)", "📖 강의 뷰어 (공부)"])

# --- [Tab 1: 데이터 학습] ---
with tab1:
    col1, col2 = st.columns(2)
    
    # 1. 족보 학습 섹션
    with col1:
        st.subheader("1. 족보 데이터베이스 구축")
        exam_files = st.file_uploader("족보 PDF 업로드", type="pdf", accept_multiple_files=True)
        
        if not st.session_state.jokbo_done:
            if st.button("족보 학습 시작 🚀"):
                if not api_key:
                    st.error("API 키를 입력하세요.")
                elif not exam_files:
                    st.error("파일을 업로드하세요.")
                else:
                    all_exams = []
                    embeddings = []
                    progress_text = st.empty()
                    bar = st.progress(0)
                    
                    total_files = len(exam_files)
                    
                    for idx, f in enumerate(exam_files):
                        texts = get_pdf_text(f)
                        for i, text in enumerate(texts):
                            if len(text) > 30:
                                progress_text.text(f"학습 중: {f.name} p.{i+1}")
                                emb = get_embedding(text)
                                if emb:
                                    all_exams.append({"info": f"{f.name} p.{i+1}", "text": text})
                                    embeddings.append(emb)
                                time.sleep(0.5) # API 제한 고려
                        bar.progress((idx + 1) / total_files)
                    
                    if embeddings:
                        st.session_state.exam_db = all_exams
                        st.session_state.exam_embeddings = np.array(embeddings)
                        st.session_state.jokbo_done = True
                        st.rerun()
        else:
            st.success(f"✅ 족보 학습 완료! (총 {len(st.session_state.exam_db)} 페이지 저장됨)")

    # 2. 강의 분석 섹션
    with col2:
        st.subheader("2. 강의록 연결")
        lec_file = st.file_uploader("오늘 강의 PDF", type="pdf")
        
        if lec_file:
            # 파일 바이트 저장 (뷰어용)
            if st.session_state.pdf_bytes is None:
                st.session_state.pdf_bytes = lec_file.getvalue()
                
            reader = PdfReader(lec_file)
            st.session_state.total_pages = len(reader.pages)
            
            if not st.session_state.lecture_done:
                if st.button("강의록 분석 시작 🔍"):
                    if not st.session_state.jokbo_done:
                        st.error("족보 학습을 먼저 완료해주세요!")
                    else:
                        lec_pages = [page.extract_text() for page in reader.pages]
                        results = []
                        bar2 = st.progress(0)
                        
                        for i, p_text in enumerate(lec_pages):
                            if len(p_text) < 30: continue
                            
                            try:
                                q_emb = genai.embed_content(
                                    model="models/text-embedding-004",
                                    content=p_text,
                                    task_type="retrieval_query"
                                )['embedding']
                                
                                sims = cosine_similarity([q_emb], st.session_state.exam_embeddings).flatten()
                                
                                if sims.max() > 0.55: # 유사도 기준 살짝 상향
                                    best_idx = sims.argmax()
                                    results.append({
                                        "page": i+1,
                                        "score": sims.max(),
                                        "exam_info": st.session_state.exam_db[best_idx]['info'],
                                        "exam_text": st.session_state.exam_db[best_idx]['text']
                                    })
                            except Exception as e:
                                print(f"Error on page {i}: {e}")
                            
                            time.sleep(0.5)
                            bar2.progress((i+1)/len(lec_pages))
                        
                        st.session_state.pre_analysis = results
                        st.session_state.lecture_done = True
                        st.rerun()
            else:
                st.success(f"✅ 강의 분석 완료! ({len(st.session_state.pre_analysis)}개 중요 포인트 발견)")
                st.markdown("👉 **'강의 뷰어' 탭으로 이동하세요.**")

# --- [Tab 2: 강의 뷰어 (핵심 기능)] ---
with tab2:
    if st.session_state.pdf_bytes and st.session_state.total_pages > 0:
        
        # 1. 페이지 슬라이더
        page_num = st.slider("페이지 이동", 1, st.session_state.total_pages, 1)
        st.caption(f"총 {st.session_state.total_pages}페이지 중 {page_num}페이지")
        
        # 2. 화면 분할 (왼쪽: PDF 이미지 / 오른쪽: 분석 결과)
        c_pdf, c_info = st.columns([1.2, 1]) # PDF를 조금 더 넓게
        
        with c_pdf:
            st.markdown("### 📄 강의록")
            # [수정됨] 여기에 수정된 이미지 뷰어 함수 적용
            display_pdf_as_image(st.session_state.pdf_bytes, page_num)
            
        with c_info:
            st.markdown(f"### 📊 분석 리포트")
            
            matches = [r for r in st.session_state.pre_analysis if r['page'] == page_num]
            
            if matches:
                st.info(f"💡 이 페이지에서 **{len(matches)}개**의 족보 연관 내용을 찾았습니다!")
                
                for match in matches:
                    with st.expander(f"🔥 기출 적중 ({match['score']*100:.0f}%) - {match['exam_info']}", expanded=True):
                        st.markdown(
                            f"""
                            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; color: #856404;">
                                <b>📌 관련 족보 내용:</b><br>
                                {match['exam_text'][:300]}...
                            </div>
                            """, 
                            unsafe_allow
