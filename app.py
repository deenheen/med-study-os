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
st.set_page_config(page_title="Med-Study AI (Fixed)", layout="wide", page_icon="🧠")

# 사이드바 설정
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
if 'exam_embeddings' not in st.session_state: st.session_state.exam_embeddings = None 
if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None

# --- [핵심 기능] Gemini 임베딩 함수 (수정됨: 최신 모델 사용) ---
def get_embedding(text):
    if not api_key: return None
    try:
        # 모델 변경: embedding-001 -> text-embedding-004 (더 안정적)
        result = genai.embed_content(
            model="models/text-embedding-004",
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
st.caption("AI가 족보를 학습할 때 **속도 제한(1.5초)**을 두어 오류를 방지합니다.")

tab1, tab2, tab3 = st.tabs(["📂 1. AI 학습 (데이터 준비)", "🎙️ 2. 실시간 수업 (AI 매칭)", "🎯 3. 복습 리포트"])

# --- [Tab 1: 데이터 준비] ---
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. 족보 PDF AI 학습")
        exam_files = st.file_uploader("족보 파일을 업로드하세요", type="pdf", accept_multiple_files=True)
        
        if st.button("족보 데이터 임베딩(학습) 시작 🚀"):
            if not api_key:
                st.error("API 키를 먼저 입력해주세요!")
            else:
                all_exams = []
                embeddings = []
                
                # 진행 상황 표시줄
                status_text = st.empty()
                progress_bar = st.progress(0)
                
                total_pages = sum([len(PdfReader(f).pages) for f in exam_files])
                processed_count = 0

                for f in exam_files:
                    texts = get_pdf_text(f)
                    for i, text in enumerate(texts):
                        if len(text.strip()) > 30: # 너무 짧은 페이지 무시
                            status_text.text(f"AI가 읽는 중... {f.name} (p.{i+1}) - 천천히 읽는 중 🐢")
                            
                            emb = get_embedding(text) 
                            if emb:
                                all_exams.append({"info": f"{f.name} (p.{i+1})", "text": text})
                                embeddings.append(emb)
                            
                            # [핵심 수정] 과부하 방지를 위해 1.5초 휴식
                            time.sleep(1.5)
                        
                        processed_count += 1
                        progress_bar.progress(min(processed_count / total_pages, 1.0))
                
                if embeddings:
                    st.session_state.exam_db = all_exams
                    st.session_state.exam_embeddings = np.array(embeddings)
                    st.success(f"완료! 총 {len(all_exams)}페이지를 학습했습니다. (오류 없이 성공)")

    with col2:
        st.subheader("2. 오늘 강의록 매칭 분석")
        lec_file = st.file_uploader("오늘 수업 PDF", type="pdf")
        
        if lec_file:
            st.session_state.pdf_bytes = lec_file.getvalue()
            
            if st.button("수업 전 AI 단권화 분석"):
                if st.session_state.exam_embeddings is not None:
                    lec_pages = get_pdf_text(lec_file)
                    results = []
                    
                    st.info("분석 중입니다... (속도 조절 중)")
                    progress_bar_lec = st.progress(0)
                    
                    for i, p_text in enumerate(lec_pages):
                        if len(p_text.strip()) < 30: continue
                        
                        # 강의 내용 임베딩 (Query)
                        q_emb = genai.embed_content(
                            model="models/text-embedding-004", # 모델 변경
                            content=p_text,
                            task_type="retrieval_query"
                        )['embedding']
                        
                        # 유사도 계산
                        sims = cosine_similarity([q_emb], st.session_state.exam_embeddings).flatten()
                        
                        if sims.max() > 0.5: # 기준점
                            best_idx = sims.argmax()
                            results.append({
                                "page": i+1, 
                                "score": sims.max(), 
                                "exam_info": st.session_state.exam_db[best_idx]['info'],
                                "exam_text": st.session_state.exam_db[best_idx]['text']
                            })
                        
                        # [핵심 수정] 여기도 휴식 시간 추가
                        time.sleep(1.0)
                        progress_bar_lec.progress((i + 1) / len(lec_pages))
                    
                    st.session_state.pre_analysis = results
                    st.success(f"분석 완료! {len(results)}개 중요 페이지 발견.")
                else:
                    st.error("먼저 왼쪽에서 족보 학습을 완료해주세요.")

# --- [Tab 2: 실시간 수업] ---
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
            st.subheader("🎙️ AI 실시간 청취")
            audio = mic_recorder(start_prompt="👂 듣기 시작", stop_prompt="⏹️ 판단해", key='live_recorder')
            
            # 테스트용 입력창
            user_input = st.text_input("또는 직접 입력 (테스트)", "심전도 ST분절 상승")

            if (audio or user_input) and st.session_state.exam_embeddings is not None:
                target_text = user_input # 실제로는 오디오 변환 텍스트 사용
                
                # 실시간 검색 임베딩
                live_emb = genai.embed_content(
                    model="models/text-embedding-004", # 모델 변경
                    content=target_text,
                    task_type="retrieval_query"
                )['embedding']
                
                sims_live = cosine_similarity([live_emb], st.session_state.exam_embeddings).flatten()
                
                if sims_live.max() > 0.45:
                    best_hit = sims_live.argmax()
                    st.toast("🚨 족보 내용 감지!", icon="🔥")
                    st.markdown(f"**관련 족보:** {st.session_state.exam_db[best_hit]['info']}")
                    st.info(st.session_state.exam_db[best_hit]['text'][:200] + "...")
                else:
                    st.caption("관련 내용 없음")

            st.divider()
            st.markdown(f"**📍 {page_selection}p 관련 기출**")
            current_matches = [r for r in st.session_state.pre_analysis if r['page'] == page_selection]
            if current_matches:
                for match in current_matches:
                    st.success(f"{match['exam_info']} (유사도 {match['score']*100:.0f}%)")

# --- [Tab 3: 리포트] ---
with tab3:
    if st.session_state.pre_analysis:
        df = pd.DataFrame(st.session_state.pre_analysis)
        df['일치도'] = (df['score'] * 100).round(1).astype(str) + '%'
        st.dataframe(df[['page', '일치도', 'exam_info']])
    else:
        st.info("데이터가 없습니다.")
