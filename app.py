import streamlit as st
import pandas as pd
import base64
import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# =========================
# 1. 초기 설정 및 세션 관리
# =========================
st.set_page_config(page_title="Med-Study AI Visualizer", layout="wide")

# 세션 상태 초기화
if 'pre_analysis' not in st.session_state: st.session_state.pre_analysis = []
if 'exam_db' not in st.session_state: st.session_state.exam_db = []
if 'vectorizer' not in st.session_state: st.session_state.vectorizer = None
if 'matrix' not in st.session_state: st.session_state.matrix = None
if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None

def get_pdf_text(file):
    reader = PdfReader(file)
    return [page.extract_text() or "" for page in reader.pages]

def display_pdf(file_bytes, page_num):
    """PDF를 베이스64로 인코딩하여 브라우저에 표시 (페이지 연동 포함)"""
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page_num}" width="100%" height="850" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# =========================
# 2. 메인 UI 화면 구성
# =========================
st.title("🩺 Med-Study OS: 시각적 뷰어 & 실시간 족보")

tab1, tab2, tab3 = st.tabs(["📂 1. 데이터 준비", "🎙️ 2. 수업 중: 뷰어 & 실시간 매칭", "🎯 3. 수업 후: 복습 리포트"])

# --- [Tab 1: 데이터 준비 및 사전 분석] ---
with tab1:
    st.header("강의실 가기 전: 족보 데이터와 강의록 연동")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. 족보 PDF 등록")
        exam_files = st.file_uploader("과거 족보 파일들을 업로드하세요", type="pdf", accept_multiple_files=True)
        if st.button("족보 데이터 인덱싱 시작"):
            all_exams = []
            for f in exam_files:
                texts = get_pdf_text(f)
                for i, text in enumerate(texts):
                    if text.strip():
                        all_exams.append({"info": f"{f.name} (p.{i+1})", "text": text})
            
            if all_exams:
                st.session_state.exam_db = all_exams
                vec = TfidfVectorizer(ngram_range=(1, 2))
                st.session_state.matrix = vec.fit_transform([e['text'] for e in all_exams])
                st.session_state.vectorizer = vec
                st.success(f"{len(all_exams)}개의 족보 페이지 인덱싱 완료!")

    with col2:
        st.subheader("2. 오늘 강의록 매칭")
        lec_file = st.file_uploader("오늘 수업용 강의록 PDF", type="pdf")
        if lec_file:
            # 뷰어용 바이너리 저장
            st.session_state.pdf_bytes = lec_file.getvalue()
            if st.button("수업 전 자동 단권화 분석"):
                if st.session_state.vectorizer:
                    lec_pages = get_pdf_text(lec_file)
                    results = []
                    for i, p_text in enumerate(lec_pages):
                        if not p_text.strip(): continue
                        qv = st.session_state.vectorizer.transform([p_text])
                        sims = cosine_similarity(qv, st.session_state.matrix).flatten()
                        if sims.max() > 0.2:
                            best_idx = sims.argmax()
                            results.append({
                                "page": i+1, 
                                "score": sims.max(), 
                                "exam_info": st.session_state.exam_db[best_idx]['info'],
                                "exam_text": st.session_state.exam_db[best_idx]['text']
                            })
                    st.session_state.pre_analysis = results
                    st.success(f"분석 완료! {len(results)}개 페이지에서 족보 적중이 예상됩니다.")
                else:
                    st.error("먼저 족보 데이터를 등록해주세요.")

# --- [Tab 2: 수업 중 시각적 뷰어 & 실시간 녹음] ---
with tab2:
    if st.session_state.pdf_bytes is None:
        st.warning("Tab 1에서 강의록 PDF를 먼저 업로드해주세요.")
    else:
        # 화면 레이아웃 분할
        col_pdf, col_live = st.columns([1.2, 0.8])
        
        with col_pdf:
            st.subheader("📄 강의록 실시간 뷰어")
            # PDF 페이지 조절 슬라이더
            page_selection = st.select_slider("페이지 이동", options=range(1, 51), value=1)
            display_pdf(st.session_state.pdf_bytes, page_selection)

        with col_live:
            st.subheader("🎙️ 실시간 족보 매칭 알림")
            
            # 1. 실시간 녹음 컨트롤러
            st.write("교수님 설명을 인식하여 족보와 대조합니다.")
            audio = mic_recorder(start_prompt="🔴 실시간 분석 시작", stop_prompt="⏹️ 중지 및 매칭", key='live_recorder')
            
            if audio:
                st.audio(audio['bytes'])
                # 시뮬레이션: 실제로는 STT API 연동 구간
                simulated_speech = "심근경색 환자가 응급실에 오면 가장 먼저 ST분절 상승 여부를 확인해야 합니다."
                st.info(f"🗣️ 교수님 발언 인식: \"{simulated_speech}\"")
                
                # 실시간 매칭 로직 (전체 족보 DB 대상 검색)
                if st.session_state.vectorizer is not None:
                    qv_live = st.session_state.vectorizer.transform([simulated_speech])
                    sims_live = cosine_similarity(qv_live, st.session_state.matrix).flatten()
                    if sims_live.max() > 0.15:
                        best_hit = sims_live.argmax()
                        st.toast("🔥 족보 적중!", icon="🚨")
                        with st.warning():
                            st.markdown(f"### 🚨 실시간 기출 매칭 완료")
                            st.write(f"**출처:** {st.session_state.exam_db[best_hit]['info']}")
                            st.write(f"**기출 지문:** {st.session_state.exam_db[best_hit]['text'][:300]}...")
                
            st.divider()
            
            # 2. 현재 PDF 페이지 기준 사전 분석 정보 표시
            st.subheader(f"📍 현재 {page_selection}p 기출 정보")
            page_matches = [res for res in st.session_state.pre_analysis if res['page'] == page_selection]
            
            if page_matches:
                for match in page_matches:
                    with st.expander("✅ 이 페이지와 연관된 족보 확인", expanded=True):
                        st.error(f"기출 출처: {match['exam_info']}")
                        st.write(f"지문 내용: {match['exam_text'][:300]}...")
                        if st.button("📌 오늘 단권화 노트에 마킹"):
                            st.toast("노트에 저장되었습니다!")
            else:
                st.info("이 페이지와 관련된 기출 내역이 없습니다.")

# --- [Tab 3: 복습 리포트 개선 코드] ---
with tab3:
    st.header("🎯 오늘의 스마트 단권화 리포트")
    
    if st.session_state.pre_analysis:
        # 데이터프레임 가공
        df = pd.DataFrame(st.session_state.pre_analysis)
        
        # 1. 소수점 점수를 백분율로 변환
        df['일치도'] = (df['score'] * 100).round(1).astype(str) + '%'
        
        # 2. 점수에 따른 중요도 등급 부여 함수
        def get_importance(score):
            if score >= 0.35: return "🔥 매우 높음 (필암기)"
            elif score >= 0.25: return "✅ 보통 (빈출)"
            else: return "⚠️ 참고 (유사성 낮음)"
            
        df['중요도'] = df['score'].apply(get_importance)
        
        # 3. 사용자에게 보여줄 열만 선택 및 이름 변경
        display_df = df[['page', '중요도', '일치도', 'exam_info']].rename(columns={
            'page': '강의록 페이지',
            'exam_info': '관련 족보 출처'
        })
        
        st.subheader("📋 기출 적중 분석 요약")
        
        # 4. 보기 좋게 스타일링된 표 출력
        st.table(display_df) 
        
        # Anki 카드 생성 기능 유지
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 오늘 기출 기반 Anki 카드 다운로드", csv, "anki_cards.csv", "text/csv")
    else:
        st.write("표시할 분석 리포트가 없습니다.")
