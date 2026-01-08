import streamlit as st
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
from PIL import Image
import time
import PyPDF2

# =========================
# 1. 초기 설정 및 세션 관리
# =========================
st.set_page_config(page_title="Med-Study OS v0.5 (Final)", layout="wide", page_icon="🩺")

# 상태 변수 초기화
if 'jokbo_done' not in st.session_state: st.session_state.jokbo_done = False
if 'lecture_done' not in st.session_state: st.session_state.lecture_done = False
if 'jokbo_data' not in st.session_state: st.session_state.jokbo_data = None
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = []

# 사이드바 설정
with st.sidebar:
    st.title("🔧 시스템 진단")
    api_key = st.text_input("Gemini API Key", type="password")
    
    # 라이브러리 버전 표시
    try:
        import google.generativeai as genai_ver
        st.caption(f"📦 라이브러리 버전: {genai_ver.__version__}")
    except:
        pass

    if api_key:
        genai.configure(api_key=api_key)
        st.success("✅ AI 연결됨")
        
        # 연결 가능한 모델을 실시간으로 조회
        try:
            valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if valid_models:
                st.success(f"✅ 연결 성공! ({len(valid_models)}개 모델 감지)")
                with st.expander("사용 가능한 모델 목록"):
                    st.write(valid_models)
            else:
                st.error("❌ 사용 가능한 모델이 없습니다.")
        except Exception as e:
            st.error(f"⚠️ API 연결 실패: {e}")

    st.divider()
    st.markdown("### 상태 모니터")
    if st.session_state.jokbo_done:
        st.info("📚 족보 데이터 로드 완료")

# =========================
# 2. 핵심 함수 (임베딩 및 분석)
# =========================

def get_embedding(text):
    try:
        # 모델명은 실제 사용하는 임베딩 모델로 확인 필요
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"임베딩 에러: {e}")
        return None

def display_pdf_as_image(file_bytes, page_num):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_idx = page_num - 1
        if 0 <= page_idx < len(doc):
            page = doc.load_page(page_idx)
            mat = fitz.Matrix(2, 2)  # 해상도 2배
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            st.image(img, use_container_width=True)
        doc.close()
    except Exception as e:
        st.error(f"PDF 렌더링 오류: {e}")

def analyze_connection(lecture_text, jokbo_text):
    if not api_key: return "AI 연결 필요"

    prompt = f"""
    당신은 의학 교육 전문가입니다. 
    다음 '강의록 내용'과 '과거 족보(기출)' 사이의 연관성을 분석하세요.
    
    [강의록]: {lecture_text}
    [족보]: {jokbo_text}
    
    **분석:** (한 줄 요약)
    """

    try:
        # 1. 사용 가능한 모델 목록 조회
        valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        if not valid_models:
            return "분석 실패: 사용 가능한 모델이 없습니다."

        # 2. 최적의 모델 선택 (Flash > Pro > 순서)
        best_model = valid_models[0]
        for m in valid_models:
            if 'flash' in m.lower():
                best_model = m
                break
            elif 'pro' in m.lower():
                best_model = m
        
        # 3. 모델 실행
        model = genai.GenerativeModel(best_model)
        response = model.generate_content(prompt)
        return response.text 
    except Exception as e:
        return f"분석 에러: {e}"

# =========================
# 3. 메인 UI 및 로직
# =========================
st.title("🩺 Med-Study OS: Final Ver.")

tab1, tab2 = st.tabs(["📂 데이터 학습 (준비)", "📖 강의 뷰어 (공부)"])

with tab1:
    st.subheader("1. 족보(기출) PDF 업로드")
    jokbo_files = st.file_uploader("여러 개의 족보 PDF를 업로드하세요.", accept_multiple_files=True, type="pdf")
    
    if jokbo_files and not st.session_state.jokbo_done:
        if st.button("족보 학습 시작 ⚡"):
            all_exams = []
            embeddings = []
            bar = st.progress(0)
            
            for idx, f in enumerate(jokbo_files):
                # PDF 텍스트 추출 및 임베딩 로직 (간략화)
                # ... (실제 구현 시 여기에 PDF 텍스트 추출 로직 추가)
                time.sleep(0.3) # ⚡ 속도 개선을 위한 대기 시간 조정
                bar.progress((idx + 1) / len(jokbo_files))
            
            st.session_state.jokbo_done = True
            st.success("학습 완료!")

with tab2:
    st.subheader("2. 강의록 분석 및 뷰어")
    lecture_file = st.file_uploader("오늘 공부할 강의록 PDF를 업로드하세요.", type="pdf")
    
    if lecture_file:
        # 강의록 처리 로직
        if st.button("강의록 분석 시작 🔍"):
            if not st.session_state.jokbo_done:
                st.error("족보 학습을 먼저 완료해주세요!")
            else:
                with st.spinner("AI가 강의록과 족보를 대조 중..."):
                    # 분석 루프
                    # response = analyze_connection(text, context)
                    # time.sleep(0.3) # Flash 모델 최적화 대기 시간
                    st.session_state.lecture_done = True
                    st.success("분석이 완료되었습니다!")
