import streamlit as st
import pandas as pd
import base64
import numpy as np
import time
import google.generativeai as genai
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# =========================
# 1. 초기 설정 및 세션 관리
# =========================
st.set_page_config(page_title="Med-Study OS v0.5", layout="wide", page_icon="🩺")

# 상태 변수 초기화 (버튼 상태 기억용)
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
        st.rerun()

# --- 함수 정의 ---
def get_embedding(text):
    if not api_key: return None
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except:
        return None

def get_pdf_text(file):
    reader = PdfReader(file)
    return [page.extract_text() or "" for page in reader.pages]

def display_pdf(file_bytes, page_num):
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    # #page= 숫자 옵션을 사용하여 해당 페이지를 엽니다.
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page_num}" width="100%" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

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
        
        # 버튼 상태 로직: 학습이 안 끝났을 때만 버튼 보임
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
                    
                    for f in exam_files:
                        texts = get_pdf_text(f)
                        for i, text in enumerate(texts):
                            if len(text) > 30:
                                progress_text.text(f"학습 중: {f.name} p.{i+1}")
                                emb = get_embedding(text)
                                if emb:
                                    all_exams.append({"info": f"{f.name} p.{i+1}", "text": text})
                                    embeddings.append(emb)
                                time.sleep(1.0) # 속도 제한
                    
                    if embeddings:
                        st.session_state.exam_db = all_exams
                        st.session_state.exam_embeddings = np.array(embeddings)
                        st.session_state.jokbo_done = True # 상태 변경!
                        st.rerun() # 화면 새로고침 (버튼 바꾸기 위해)
        else:
            # 학습이 끝난 경우
            st.success(f"✅ 족보 학습 완료! (총 {len(st.session_state.exam_db)} 페이지 저장됨)")
            st.info("새로운 족보를 넣으려면 사이드바의 '전체 초기화'를 누르세요.")

    # 2. 강의 분석 섹션
    with col2:
        st.subheader("2. 강의록 연결")
        lec_file = st.file_uploader("오늘 강의 PDF", type="pdf")
        
        if lec_file:
            st.session_state.pdf_bytes = lec_file.getvalue()
            # 전체 페이지 수 계산
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
                            
                            q_emb = genai.embed_content(
                                model="models/text-embedding-004",
                                content=p_text,
                                task_type="retrieval_query"
                            )['embedding']
                            
                            sims = cosine_similarity([q_emb], st.session_state.exam_embeddings).flatten()
                            
                            if sims.max() > 0.5: # 유사도 기준
                                best_idx = sims.argmax()
                                results.append({
                                    "page": i+1,
                                    "score": sims.max(),
                                    "exam_info": st.session_state.exam_db[best_idx]['info'],
                                    "exam_text": st.session_state.exam_db[best_idx]['text']
                                })
                            
                            time.sleep(1.0)
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
        
        # 1. 페이지 슬라이더 (여기서 페이지를 조작)
        page_num = st.slider("페이지 이동", 1, st.session_state.total_pages, 1)
        st.caption(f"총 {st.session_state.total_pages}페이지 중 {page_num}페이지")
        
        # 화면 분할 (왼쪽: PDF / 오른쪽: 분석 결과)
        c_pdf, c_info = st.columns([1.5, 1])
        
        with c_pdf:
            display_pdf(st.session_state.pdf_bytes, page_num)
            
        with c_info:
            st.subheader(f"📄 {page_num}p 분석 리포트")
            
            # 현재 페이지에 해당하는 분석 결과 찾기
            matches = [r for r in st.session_state.pre_analysis if r['page'] == page_num]
            
            if matches:
                st.toast(f"{page_num}페이지에서 족보 내용을 발견했습니다!", icon="🔥")
                
                for match in matches:
                    # 카드 형태로 보여주기
                    with st.container(border=True):
                        st.markdown(f"### 🔥 기출 적중 ({match['score']*100:.0f}%)")
                        st.markdown(f"**출처:** `{match['exam_info']}`")
                        
                        # 형광펜 효과처럼 배경색 입히기
                        st.markdown(
                            f"""
                            <div style="background-color: #fff9c4; padding: 10px; border-radius: 5px;">
                                <b>관련 족보 내용:</b><br>
                                {match['exam_text'][:200]}...
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
            else:
                st.info("이 페이지와 직접적으로 관련된 족보 내용은 발견되지 않았습니다.")
                st.markdown("Try: 다음 페이지로 넘겨보세요!")
                
    else:
        st.warning("데이터 학습 탭에서 강의록을 먼저 업로드하고 분석해주세요.")
