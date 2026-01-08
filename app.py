import streamlit as st
import pandas as pd
import numpy as np
import time
import google.generativeai as genai
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai_ver # 라이브러리 버전 확인용

# =========================
# 1. 초기 설정 및 세션 관리
# =========================
st.set_page_config(page_title="Med-Study OS v0.5 (Final)", layout="wide", page_icon="🩺")

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
    st.title("🔧 시스템 진단")
    api_key = st.text_input("Gemini API Key", type="password")
    
    # [진단] 현재 설치된 라이브러리 버전 표시 (0.8.3 이상인지 확인용)
    st.caption(f"📦 라이브러리 버전: {genai_ver.__version__}")

    if api_key:
        genai.configure(api_key=api_key)
        
        # [핵심] 연결 가능한 모델을 실시간으로 조회
        try:
            my_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    my_models.append(m.name)
            
            if my_models:
                st.success(f"✅ 연결 성공! ({len(my_models)}개 모델 감지)")
                with st.expander("사용 가능한 모델 목록"):
                    st.write(my_models)
            else:
                st.error("❌ 사용 가능한 모델이 없습니다.")
        except Exception as e:
            st.error(f"⚠️ API 연결 실패: {e}")

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
        # 임베딩 에러는 보통 조용히 넘어가는게 낫습니다.
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
            mat = fitz.Matrix(2, 2) 
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            st.image(img, use_container_width=True)
        else:
            st.error("페이지 범위를 벗어났습니다.")
    except Exception as e:
        st.error(f"PDF 렌더링 오류: {e}")

# [최종 해결] 이름을 추측하지 않고, 조회된 모델 중 하나를 골라 쓰는 함수
def analyze_connection(lecture_text, jokbo_text):
    if not api_key: return "AI 연결 필요"
    
    prompt = f"""
    당신은 의대생 조교입니다.
    [강의록]과 [족보]의 연결고리를 아주 짧게 설명하세요.
    
    [강의록]
    {lecture_text[:800]} 
    
    [족보]
    {jokbo_text[:800]}
    
    형식:
    **핵심:** (단어)
    **분석:** (한 줄 요약)
    """
    
    try:
        # 1. 사용 가능한 모델 목록 다시 조회
        valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        if not valid_models:
            return "분석 실패: 사용 가능한 모델이 없습니다."

        # 2. 가장 좋은 모델 자동 선택 (Flash > Pro > 아무거나)
        best_model = valid_models[0] # 기본값: 목록의 첫 번째
        
        for m in valid_models:
            if 'flash' in m.lower(): # Flash가 있으면 1순위
                best_model = m
                break
            if 'pro' in m.lower() and 'flash' not in best_model.lower(): # Pro는 2순위
                best_model = m
        
        # 3. 선택된 모델로 실행 (이제 이름 틀릴 일이 없음)
        model = genai.GenerativeModel(best_model)
        response = model.generate_content(prompt)
        return response.text 

    except Exception as e:
        return f"분석 에러 ({best_model} 사용 시도): {e}"

# =========================
# 2. 메인 UI
# =========================
st.title("🩺 Med-Study OS: Final Ver.")

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
                                time.sleep(0.3)
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
                                q_emb = genai.embed_content(
                                    model="models/text-embedding-004",
                                    content=p_text,
                                    task_type="retrieval_query"
                                )['embedding']
                                sims = cosine_similarity([q_emb], st.session_state.exam_embeddings).flatten()
                                
                                if sims.max() > 0.55:
                                    best_idx = sims.argmax()
                                    matched_text = st.session_state.exam_db[best_idx]['text']
                                    matched_info = st.session_state.exam_db[best_idx]['info']
                                    
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
                            
                            time.sleep(0.3)
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
                st.info(f"⚡ {len(matches)}개의 족보 내용을 빠르게 찾았습니다!")
                for match in matches:
                    with st.expander(f"🔥 기출 적중 ({match['score']*100:.0f}%) - {match['exam_info']}", expanded=True):
                        
                        if 'ai_comment' in match:
                            st.markdown(f"""
                            <div style="background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 5px solid #2196f3; color: #0d47a1;">
                                {match['ai_comment'].replace(chr(10), '<br>')}
                            </div>
                            """, unsafe_allow_html=True)
                        
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
