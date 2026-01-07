import streamlit as st
import pandas as pd
import numpy as np
import time
import google.generativeai as genai
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
from PIL import Image

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
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        # 에러 발생 시 로그만 찍고 넘어감
        print(f"임베딩 에러: {e}")
        return None

def get_pdf_text(file):
    reader = PdfReader(file)
    return [page.extract_text() or "" for page in reader.pages]

def display_pdf_as_image(file_bytes, page_num):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_idx = page_num - 1
        
        if 0 <= page_idx < len(doc):
            page = doc.load_page(page_idx)
            mat = fitz.Matrix(2, 2) # 해상도 2배
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            st.image(img, use_container_width=True)
        else:
            st.error("페이지 범위를 벗어났습니다.")
    except Exception as e:
        st.error(f"PDF 렌더링 오류: {e}")

# [수정됨] 모델 변경(gemini-1.5-flash) 및 상세 에러 출력 추가
def analyze_connection(lecture_text, jokbo_text):
    if not api_key: return "AI 연결 필요"
    
    prompt = f"""
    당신은 의대생의 공부를 돕는 조교입니다.
    [강의록 내용]과 [족보(기출) 내용]을 비교하여, 왜 이 족보가 강의록의 이 부분과 관련이 있는지 설명해주세요.
    
    [강의록]
    {lecture_text[:800]} 
    
    [족보]
    {jokbo_text[:800]}
    
    요청사항:
    1. 두 내용의 공통된 의학적/생물학적 주제가 무엇인지 한 단어로 정의하세요.
    2. 족보 내용이 강의록 공부에 어떻게 도움이 되는지 한 문장으로 요약하세요.
    
    출력 형식:
    **핵심 주제:** (주제)
    **분석:** (설명)
    """
    try:
        # 모델명을 최신/경량 모델인 'gemini-1.5-flash'로 변경 (속도 빠름, 에러 적음)
        model = genai.GenerativeModel("gemini-1.5-flash") 
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # [중요] 어떤 에러인지 화면에 보이게 수정함
        return f"오류 발생: {str(e)}"

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
                                time.sleep(0.5)
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
                                # 1. 임베딩 검색
                                q_emb = genai.embed_content(
                                    model="models/text-embedding-004",
                                    content=p_text,
                                    task_type="retrieval_query"
                                )['embedding']
                                sims = cosine_similarity([q_emb], st.session_state.exam_embeddings).flatten()
                                
                                # 2. 매칭 및 AI 분석
                                if sims.max() > 0.55:
                                    best_idx = sims.argmax()
                                    matched_text = st.session_state.exam_db[best_idx]['text']
                                    matched_info = st.session_state.exam_db[best_idx]['info']
                                    
                                    # 여기서 AI에게 이유를 물어봅니다
                                    ai_reason = analyze_connection(p_text, matched_text)
                                    
                                    results.append({
                                        "page": i+1,
                                        "score": sims.max(),
                                        "exam_info": matched_info,
                                        "exam_text": matched_text,
                                        "ai_comment": ai_reason
                                    })
                            except Exception as e:
                                print(f"Error page {i}: {e}")
                            
                            time.sleep(1.0) # AI 분석하느라 시간이 좀 걸리므로 딜레이
                            bar2.progress((i+1)/len(lec_pages))
                        
                        st.session_state.pre_analysis = results
                        st.session_state.lecture_done = True
                        st.rerun()
            else:
                st.success(f"✅ 강의 분석 완료! ({len(st.session_state.pre_analysis)}개 중요 포인트 발견)")
                st.markdown("👉 **'강의 뷰어' 탭으로 이동하세요.**")

# --- [Tab 2: 강의 뷰어] ---
with tab2:
    if st.session_state.pdf_bytes and st.session_state.total_pages > 0:
        page_num = st.slider("페이지 이동", 1, st.session_state.total_pages, 1)
        st.caption(f"총 {st.session_state.total_pages}페이지 중 {page_num}페이지")
        
        c_pdf, c_info = st.columns([1.2, 1])
        
        with c_pdf:
            st.markdown("### 📄 강의록")
            display_pdf_as_image(st.session_state.pdf_bytes, page_num)
            
        with c_info:
            st.markdown(f"### 📊 분석 리포트")
            matches = [r for r in st.session_state.pre_analysis if r['page'] == page_num]
            
            if matches:
                st.info(f"💡 이 페이지에서 **{len(matches)}개**의 족보 연관 내용을 찾았습니다!")
                for match in matches:
                    with st.expander(f"🔥 기출 적중 ({match['score']*100:.0f}%) - {match['exam_info']}", expanded=True):
                        
                        # AI 분석 결과 출력
                        if 'ai_comment' in match:
                            st.markdown(f"""
                            <div style="background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 5px solid #2196f3; color: #0d47a1;">
                                {match['ai_comment'].replace(chr(10), '<br>')}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 족보 원문 출력
                        st.markdown(
                            f"""
                            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; color: #856404; font-size: 0.9em;">
                                <b>📌 관련 족보 원문:</b><br>
                                {match['exam_text'][:300]}...
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
            else:
                st.write("---")
                st.write("✅ 이 페이지는 특별히 감지된 족보 내용이 없습니다.")
                st.write("가볍게 읽고 넘어가셔도 좋습니다.")
    else:
        st.warning("데이터 학습 탭에서 강의록을 먼저 업로드하고 분석해주세요.")

