from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai_ver # 라이브러리 버전 확인용

# =========================
# 1. 초기 설정 및 세션 관리
# =========================
st.set_page_config(page_title="Med-Study OS v0.5 (Light)", layout="wide", page_icon="⚡")
st.set_page_config(page_title="Med-Study OS v0.5 (Final)", layout="wide", page_icon="🩺")

# 상태 변수 초기화
if 'jokbo_done' not in st.session_state: st.session_state.jokbo_done = False
@@ -24,12 +25,31 @@

# 사이드바 설정
with st.sidebar:
    st.title("⚡ 설정 (Light Ver.)")
    st.title("🔧 시스템 진단")
    api_key = st.text_input("Gemini API Key", type="password")
    
    # [진단] 현재 설치된 라이브러리 버전 표시 (0.8.3 이상인지 확인용)
    st.caption(f"📦 라이브러리 버전: {genai_ver.__version__}")

    if api_key:
        genai.configure(api_key=api_key)
        st.success("✅ AI 연결됨")
    
        
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
@@ -61,7 +81,7 @@ def get_embedding(text):
        )
        return result['embedding']
    except Exception as e:
        print(f"임베딩 에러: {e}")
        # 임베딩 에러는 보통 조용히 넘어가는게 낫습니다.
        return None

def get_pdf_text(file):
@@ -75,7 +95,7 @@ def display_pdf_as_image(file_bytes, page_num):

        if 0 <= page_idx < len(doc):
            page = doc.load_page(page_idx)
            mat = fitz.Matrix(2, 2) # 해상도 2배
            mat = fitz.Matrix(2, 2) 
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            st.image(img, use_container_width=True)
@@ -84,7 +104,7 @@ def display_pdf_as_image(file_bytes, page_num):
    except Exception as e:
        st.error(f"PDF 렌더링 오류: {e}")

# [속도 개선] 가장 가벼운 모델(1.5 Flash)을 최우선으로 사용
# [최종 해결] 이름을 추측하지 않고, 조회된 모델 중 하나를 골라 쓰는 함수
def analyze_connection(lecture_text, jokbo_text):
    if not api_key: return "AI 연결 필요"

@@ -103,32 +123,35 @@ def analyze_connection(lecture_text, jokbo_text):
    **분석:** (한 줄 요약)
    """

    # ⚡ 속도 최적화 모델 리스트 (가벼운 순서)
    candidate_models = [
        "gemini-1.5-flash",         # 1순위: 가장 빠름
        "models/gemini-1.5-flash",  # 2순위
        "gemini-1.5-flash-002",     # 3순위: 최신 최적화 버전
        "gemini-1.0-pro",           # 4순위: 구버전 (가벼움)
        "gemini-pro"
    ]
    try:
        # 1. 사용 가능한 모델 목록 다시 조회
        valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        if not valid_models:
            return "분석 실패: 사용 가능한 모델이 없습니다."

    last_error = ""
    
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text 
        except Exception as e:
            last_error = str(e)
            continue 
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

    return f"분석 실패 (에러: {last_error})"
    except Exception as e:
        return f"분석 에러 ({best_model} 사용 시도): {e}"

# =========================
# 2. 메인 UI
# =========================
st.title("⚡ Med-Study OS: 라이트 버전")
st.title("🩺 Med-Study OS: Final Ver.")

tab1, tab2 = st.tabs(["📂 데이터 학습 (준비)", "📖 강의 뷰어 (공부)"])

@@ -163,8 +186,7 @@ def analyze_connection(lecture_text, jokbo_text):
                                if emb:
                                    all_exams.append({"info": f"{f.name} p.{i+1}", "text": text})
                                    embeddings.append(emb)
                                # ⚡ 학습 속도도 높이기 위해 대기 시간 단축 (0.5 -> 0.2)
                                time.sleep(0.2)
                                time.sleep(0.3)
                        bar.progress((idx + 1) / total_files)

                    if embeddings:
@@ -188,7 +210,7 @@ def analyze_connection(lecture_text, jokbo_text):
            st.session_state.total_pages = len(reader.pages)

            if not st.session_state.lecture_done:
                if st.button("강의록 분석 시작 ⚡"):
                if st.button("강의록 분석 시작 🔍"):
                    if not st.session_state.jokbo_done:
                        st.error("족보 학습을 먼저 완료해주세요!")
                    else:
@@ -223,8 +245,6 @@ def analyze_connection(lecture_text, jokbo_text):
                            except Exception as e:
                                print(f"Error page {i}: {e}")

                            # ⚡ [중요] 분석 대기 시간을 1.0초 -> 0.3초로 대폭 단축!
                            # Flash 모델은 빨라서 이래도 괜찮습니다.
                            time.sleep(0.3)
                            bar2.progress((i+1)/len(lec_pages))
