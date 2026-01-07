import streamlit as st
import pandas as pd
import base64
import numpy as np
import google.generativeai as genai
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# =========================
# 1. 초기 설정 및 세션 관리
# =========================
st.set_page_config(page_title="Med-Study AI (Gemini Ver)", layout="wide", page_icon="🧠")

# 사이드바에서 API 키 입력 받기 (이게 있어야 Gemini가 작동함)
with st.sidebar:
    st.title("⚙️ 설정")
    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        st.success("AI 연결 완료!")
    else:
        st.warning("API 키를 입력해주세요.")

# 세션 상태 초기화
if 'pre_analysis' not in st.session_state: st.session_state.pre_analysis = []
if 'exam_db' not in st.session_state: st.session_state.exam_db = []
if 'exam_embeddings' not in st.session_state: st.session_state.exam_embeddings = None # 벡터 매트릭스 대신 임베딩 저장
if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None

# --- [핵심 기능] Gemini 임베딩 함수 (의미를 숫자로 변환) ---
def get_embedding(text):
    if not api_key: return None
    try:
        # 'embedding-001' 모델을 사용하여 텍스트의 의미를 벡터로 변환
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document",
            title="Med Study"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"임베딩 오류: {e}")
        return None

def get_pdf_text(file):
    reader = PdfReader(file)
    return [page.extract_text() or "" for page in reader.pages]

def display_pdf(file_bytes, page_num):
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page_num}" width="100%" height="850" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# =========================
# 2. 메인 UI 화면 구성
# =========================
st.title("🧠 Med-Study OS: Gemini Semantic Search")
st.caption("단어 일치가 아니라 **의미(Meaning)**를 기반으로 족보를 찾아냅니다.")

tab1, tab2, tab3 = st.tabs(["📂 1. AI 학습 (데이터 준비)", "🎙️ 2. 실시간 수업 (AI 매칭)", "🎯 3. 복습 리포트"])

# --- [Tab 1: 데이터 준비 및 사전 분석] ---
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. 족보 PDF AI 학습")
        exam_files = st.file_uploader("족보 파일을 업로드하세요 (AI가 내용을 이해합니다)", type="pdf", accept_multiple_files=True)
        
        if st.button("족보 데이터 임베딩(학습) 시작 🚀"):
            if not api_key:
                st.error("API 키를 먼저 입력해주세요!")
            else:
                all_exams = []
                embeddings = []
                
                with st.status("AI가 족보를 읽고 기억하는 중...", expanded=True) as status:
                    for f in exam_files:
                        texts = get_pdf_text(f)
                        for i, text in enumerate(texts):
                            if len(text.strip()) > 50: # 너무 짧은 페이지는 무시
                                st.write(f"Reading: {f.name} p.{i+1}")
                                emb = get_embedding(text) # 여기서 Gemini가 텍스트를 숫자로 변환
                                if emb:
                                    all_exams.append({"info": f"{f.name} (p.{i+1})", "text": text})
                                    embeddings.append(emb)
                    
                    if embeddings:
                        st.session_state.exam_db = all_exams
                        st.session_state.exam_embeddings = np.array(embeddings) # 리스트를 numpy 배열로 변환
                        status.update(label="학습 완료!", state="complete", expanded=False)
                        st.success(f"총 {len(all_exams)}개의 족보 페이지를 AI가 기억했습니다!")

    with col2:
        st.subheader("2. 오늘 강의록 매칭 분석")
        lec_file = st.file_uploader("오늘 수업 PDF", type="pdf")
        
        if lec_file:
            st.session_state.pdf_bytes = lec_file.getvalue()
            
            if st.button("수업 전 AI 단권화 분석"):
                if st.session_state.exam_embeddings is not None:
                    lec_pages = get_pdf_text(lec_file)
                    results = []
                    
                    progress_bar = st.progress(0)
                    for i, p_text in enumerate(lec_pages):
                        if len(p_text.strip()) < 50: continue
                        
                        # 강의 내용도 임베딩으로 변환 (Query)
                        q_emb = genai.embed_content(
                            model="models/embedding-001",
                            content=p_text,
                            task_type="retrieval_query"
                        )['embedding']
                        
                        # 코사인 유사도 계산 (Gemini가 만든 벡터끼리 비교)
                        # reshape(1, -1)은 1차원 배열을 2차원 행렬로 바꾸는 것
                        sims = cosine_similarity([q_emb], st.session_state.exam_embeddings).flatten()
                        
                        if sims.max() > 0.6: # 임베딩은 TF-IDF보다 점수가 높게 나오는 경향이 있음 (기준점 조절 필요)
                            best_idx = sims.argmax()
                            results.append({
                                "page": i+1, 
                                "score": sims.max(), 
                                "exam_info": st.session_state.exam_db[best_idx]['info'],
                                "exam_text": st.session_state.exam_db[best_idx]['text']
                            })
                        progress_bar.progress((i + 1) / len(lec_pages))
                    
                    st.session_state.pre_analysis = results
                    st.success(f"분석 완료! {len(results)}개 페이지에서 연관성 발견.")
                else:
                    st.error("족보 학습을 먼저 진행해주세요.")

# --- [Tab 2: 수업 중 뷰어 & 실시간] ---
with tab2:
    if st.session_state.pdf_bytes is None:
        st.warning("Tab 1에서 강의록을 먼저 업로드해주세요.")
    else:
        col_pdf, col_live = st.columns([1.2, 0.8])
        
        with col_pdf:
            st.subheader("📄 강의록 뷰어")
            page_selection = st.select_slider("페이지", options=range(1, 51), value=1)
            display_pdf(st.session_state.pdf_bytes, page_selection)

        with col_live:
            st.subheader("🎙️ AI 실시간 청취 중")
            audio = mic_recorder(start_prompt="👂 듣기 시작", stop_prompt="⏹️ 판단해", key='live_recorder')
            
            if audio:
                # 실제로는 여기서 STT(Speech-to-Text) API를 써야 함.
                # 현재는 시뮬레이션을 위해 텍스트 입력창으로 대체하거나 예시 문장 사용
                user_input = st.text_input("교수님 말씀 (테스트용 입력)", "이 환자는 심전도에서 ST분절이 올라가 있습니다.")
                
                if user_input and st.session_state.exam_embeddings is not None:
                    # 1. 교수님 말씀을 Gemini 임베딩으로 변환
                    live_emb = genai.embed_content(
                        model="models/embedding-001",
                        content=user_input,
                        task_type="retrieval_query"
                    )['embedding']
                    
                    # 2. 유사도 검색
                    sims_live = cosine_similarity([live_emb], st.session_state.exam_embeddings).flatten()
                    
                    # 3. 결과 판정
                    if sims_live.max() > 0.55: # 임계값 (Threshold)
                        best_hit = sims_live.argmax()
                        st.toast("🚨 족보 내용 감지!", icon="🔥")
                        
                        st.markdown(f"""
                        ### 🎯 AI 매칭 성공 ({sims_live.max()*100:.1f}%)
                        **교수님 말씀:** "{user_input}"
                        **관련 족보:** {st.session_state.exam_db[best_hit]['info']}
                        """)
                        
                        with st.expander("족보 내용 보기", expanded=True):
                            st.info(st.session_state.exam_db[best_hit]['text'][:400] + "...")
                    else:
                        st.caption("관련된 족보 내용이 없습니다.")

            # 현재 페이지 연동 정보
            st.divider()
            st.markdown(f"**📍 {page_selection}p 관련 기출**")
            current_matches = [r for r in st.session_state.pre_analysis if r['page'] == page_selection]
            if current_matches:
                for match in current_matches:
                    st.success(f"출처: {match['exam_info']} (유사도 {match['score']*100:.0f}%)")
            else:
                st.write("발견된 내용 없음")

# --- [Tab 3: 리포트] ---
with tab3:
    if st.session_state.pre_analysis:
        df = pd.DataFrame(st.session_state.pre_analysis)
        df['일치도'] = (df['score'] * 100).round(1).astype(str) + '%'
        st.dataframe(df[['page', '일치도', 'exam_info']])
    else:
        st.info("아직 분석된 데이터가 없습니다.")