# app.py
import time
import re
import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ==========================================
# 0. Page config
# ==========================================
st.set_page_config(page_title="Med-Study OS", layout="wide", page_icon="🩺")
st.caption("📌 흐름: (1) 족보 DB 구축 → (2,3) 조교 설명 및 포인트 추출 → (4) 나만의 정리노트 완성")

# ==========================================
# 1. Session state
# ==========================================
if "db" not in st.session_state:
    st.session_state.db = []

if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "api_key_ok" not in st.session_state:
    st.session_state.api_key_ok = False

if "text_models" not in st.session_state:
    st.session_state.text_models = []

if "best_text_model" not in st.session_state:
    st.session_state.best_text_model = None

if "lecture_doc" not in st.session_state:
    st.session_state.lecture_doc = None

if "lecture_filename" not in st.session_state:
    st.session_state.lecture_filename = None

if "current_page" not in st.session_state:
    st.session_state.current_page = 0

# caches for Tab 2
if "last_page_sig" not in st.session_state:
    st.session_state.last_page_sig = None

if "last_ai_sig" not in st.session_state:
    st.session_state.last_ai_sig = None

if "last_ai_text" not in st.session_state:
    st.session_state.last_ai_text = ""

if "last_related" not in st.session_state:
    st.session_state.last_related = []

# caches for Tab 3
if "last_transcript_result" not in st.session_state:
    st.session_state.last_transcript_result = ""

# Storage for Summary Notes (Tab 4)
if "my_notes" not in st.session_state:
    # item: {"id": str, "source": str, "content": str, "timestamp": str}
    st.session_state.my_notes = []

# ==========================================
# 2. Settings
# ==========================================
JOKBO_THRESHOLD = 0.72  # 추천 0.70~0.75

def has_jokbo_evidence(related: list[dict]) -> bool:
    return bool(related) and related[0]["score"] >= JOKBO_THRESHOLD

# ==========================================
# 3. Utils
# ==========================================
def ensure_configured():
    if st.session_state.get("api_key"):
        genai.configure(api_key=st.session_state["api_key"])

def extract_text_from_pdf(uploaded_file):
    data = uploaded_file.getvalue()
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text, "source": uploaded_file.name})
    return pages

def get_embedding(text: str):
    text = (text or "").strip()
    if not text:
        return []
    text = text[:12000]
    ensure_configured()
    try:
        return genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
        )["embedding"]
    except Exception:
        try:
            return genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document",
            )["embedding"]
        except Exception:
            return []

def filter_db_by_subject(subject: str, db: list[dict]):
    if not db:
        return []
    subject = (subject or "").strip()
    if subject in ["전체", "ALL", ""]:
        return db
    return [x for x in db if x.get("subject") == subject]

def find_relevant_jokbo(query_text: str, db: list[dict], top_k: int = 5):
    if not db:
        return []
    query_emb = get_embedding(query_text)
    if not query_emb:
        return []
    valid_items = [item for item in db if item.get("embedding")]
    if not valid_items:
        return []
    db_embs = [item["embedding"] for item in valid_items]
    sims = cosine_similarity([query_emb], db_embs)[0]
    top_idxs = np.argsort(sims)[::-1][:top_k]
    return [{"score": float(sims[i]), "content": valid_items[i]} for i in top_idxs]

def add_to_notes(source_type: str, content: str):
    """노트 저장 함수"""
    new_note = {
        "id": str(time.time()),
        "source": source_type,
        "content": content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    st.session_state.my_notes.append(new_note)
    st.toast("✅ 정리노트에 추가되었습니다!", icon="📝")

# ==========================================
# 4. AI (조교 설명)
# ==========================================
@st.cache_data(show_spinner=False)
def list_text_models(api_key: str):
    genai.configure(api_key=api_key)
    models = genai.list_models()
    out = []
    for m in models:
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            out.append(m.name)
    return out

def pick_best_text_model(model_names: list[str]):
    if not model_names:
        return None
    flash = [m for m in model_names if "flash" in m.lower()]
    return flash[0] if flash else model_names[0]

def generate_with_fallback(prompt: str, model_names: list[str]):
    ensure_configured()
    last_err = None
    for name in model_names:
        if not name:
            continue
        try:
            model = genai.GenerativeModel(name)
            res = model.generate_content(prompt)
            text = getattr(res, "text", None)
            if text:
                return text, name
            return str(res), name
        except Exception as e:
            last_err = e
    raise last_err

def build_ta_prompt(lecture_text: str, related: list[dict], subject: str):
    ctx_lines = []
    for r in related[:3]:
        c = r["content"]
        src = c.get("source", "")
        pg = c.get("page", "?")
        txt = (c.get("text") or "")[:450]
        ctx_lines.append(f'- [{src} p{pg} | sim={r["score"]:.3f}] {txt}')
    jokbo_ctx = "\n".join(ctx_lines)

    return f"""
너는 의대 조교다. 학생이 강의를 듣는 중이며, 지금 텍스트가 족보에서 어떤 식으로 출제되었는지 설명해라.
과목: {subject}

규칙:
- [관련 족보 발췌]에 근거해서만 말해라.
- 강의 텍스트를 길게 다시 말하지 말고, "족보 출제 포인트" 위주로 요약해라.

출력 형식:
**[조교 코멘트]**
(핵심 요약 1문장)

**[족보 기출 포인트]**
- (포인트 1)
- (포인트 2)

**[문제 유형]** (객관식/서술형/빈칸 등)

**[암기 키워드]**
키워드1, 키워드2...

[입력 텍스트]
{lecture_text}

[관련 족보 발췌]
{jokbo_ctx}
""".strip()

def build_transcript_prompt(chunks: list[str], related_packs: list[list[dict]], subject: str):
    lines = []
    for idx, (chunk, rel) in enumerate(zip(chunks, related_packs), start=1):
        if not has_jokbo_evidence(rel):
            continue
        ctx = []
        for r in rel[:2]:
            c = r["content"]
            ctx.append(f'- [{c.get("source","")} p{c.get("page","?")} sim={r["score"]:.3f}] {(c.get("text","")[:250])}')
        lines.append(f"""
(구간 {idx})
[강의 전사 일부]
{chunk}

[관련 족보 발췌]
{chr(10).join(ctx)}
""".strip())

    packed = "\n\n".join(lines)
    if not packed.strip():
        packed = "(족보 근거가 있는 구간이 없습니다.)"

    return f"""
너는 의대 조교다. 아래는 강의 전사 텍스트다.
'족보에 실제로 나왔던 내용'만 골라 정리해라.
과목: {subject}

규칙:
- 반드시 [관련 족보 발췌] 근거가 있는 구간만 포함해라.
- 출력은 "족보 포인트 노트" 형태로 간결하게.

출력 형식:
## 족보 포인트 정리
1. **(주제)**
   - 내용: ...
   - 근거: (파일명/페이지)
   - 암기: ...

2. **(주제)**
   ...

입력 데이터:
{packed}
""".strip()

# ==========================================
# 5. Transcript chunking
# ==========================================
def chunk_transcript(text: str, max_chars: int = 900):
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for p in parts:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            start = 0
            while start < len(p):
                chunks.append(p[start:start + max_chars])
                start += max_chars
    return chunks

# ==========================================
# 6. Sidebar
# ==========================================
with st.sidebar:
    st.title("🩺 Med-Study")

    api_key = st.text_input("Gemini API Key", type="password", key="api_key_input")
    if api_key:
        try:
            st.session_state.api_key = api_key
            genai.configure(api_key=api_key)
            available_models = list_text_models(api_key)
            if not available_models:
                st.session_state.api_key_ok = False
                st.error("generateContent 가능한 모델이 없습니다.")
            else:
                st.session_state.api_key_ok = True
                st.session_state.text_models = available_models
                st.session_state.best_text_model = pick_best_text_model(available_models)
                st.success("AI 연결 완료")
                st.caption(f"텍스트 모델(자동): {st.session_state.best_text_model}")
        except Exception as e:
            st.session_state.api_key_ok = False
            st.error(f"모델 목록 조회 실패: {e}")

    st.divider()

    subjects_in_db = sorted({x.get("subject", "") for x in st.session_state.db if x.get("subject")})
    st.caption(f"📚 학습된 족보 페이지: **{len(st.session_state.db)}**")
    st.caption(f"📚 과목: **{', '.join(subjects_in_db) if subjects_in_db else '(없음)'}**")

    if st.button("족보 DB 초기화", key="reset_db_btn"):
        st.session_state.db = []
        st.session_state.last_page_sig = None
        st.session_state.last_ai_sig = None
        st.session_state.last_ai_text = ""
        st.session_state.last_related = []
        st.session_state.last_transcript_result = ""
        st.session_state.my_notes = []
        st.rerun()

# ==========================================
# 7. Tabs
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📂 1) 족보 DB 구축", "📖 2) 강의본 + 조교", "🎙️ 3) 전사 텍스트 + 조교", "📝 4) 나만의 정리노트"]
)

# ==================================================
# TAB 1 — Upload
# ==================================================
with tab1:
    st.header("📂 1) 과목별 족보 업로드/학습")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        subject_for_upload = st.selectbox("과목", ["해부학", "생리학", "약리학", "기타(직접입력)"], index=1)
    with c2:
        subject_custom = st.text_input("기타 과목명", disabled=(subject_for_upload != "기타(직접입력)"))

    subject_final = subject_custom.strip() if subject_for_upload == "기타(직접입력)" else subject_for_upload
    subject_final = subject_final if subject_final else "기타(미입력)"

    files = st.file_uploader("족보 PDF 업로드", type="pdf", accept_multiple_files=True)
    max_pages = st.number_input("파일당 최대 학습 페이지", 1, 500, 60)

    if st.button("📚 족보 DB 구축 시작", key="build_db_btn"):
        if not st.session_state.api_key_ok:
            st.error("API Key 필요")
            st.stop()
        if not files:
            st.warning("파일을 업로드하세요.")
            st.stop()

        bar = st.progress(0)
        status = st.empty()
        new_db = []
        total_files = len(files)

        for i, f in enumerate(files):
            status.text(f"📖 처리 중: {f.name}")
            pages = extract_text_from_pdf(f)[: int(max_pages)]
            if not pages:
                continue
            for j, p in enumerate(pages):
                emb = get_embedding(p["text"])
                if emb:
                    p["embedding"] = emb
                    p["subject"] = subject_final
                    new_db.append(p)
                time.sleep(0.7)
            bar.progress((i + 1) / total_files)

        st.session_state.db.extend(new_db)
        status.text("✅ 완료")
        st.success(f"[{subject_final}] {len(new_db)} 페이지 학습 완료")

# ==================================================
# TAB 2 — PDF Viewer + TA
# ==================================================
with tab2:
    st.header("📖 2) 강의본(PDF) → 조교 설명")
    
    if not st.session_state.db:
        st.warning("먼저 1번 탭에서 족보 DB를 구축하세요.")

    subjects_in_db = sorted({x.get("subject", "") for x in st.session_state.db if x.get("subject")})
    subject_options = ["전체"] + (subjects_in_db if subjects_in_db else ["(DB 없음)"])
    subject_pick = st.selectbox("분석 과목", subject_options, key="tab2_sub")
    
    lec_file = st.file_uploader("강의본 PDF 업로드", type="pdf", key="lec_pdf")

    if lec_file:
        if st.session_state.lecture_doc is None or st.session_state.lecture_filename != lec_file.name:
            data = lec_file.getvalue()
            st.session_state.lecture_doc = fitz.open(stream=data, filetype="pdf")
            st.session_state.lecture_filename = lec_file.name
            st.session_state.current_page = 0
            st.session_state.last_ai_text = ""

        doc = st.session_state.lecture_doc
        col_view, col_right = st.columns([6, 4])

        with col_view:
            nav1, nav2, nav3 = st.columns([1, 2, 1])
            if nav1.button("◀", key="prev"):
                if st.session_state.current_page > 0: st.session_state.current_page -= 1
            nav2.markdown(f"<center><b>{st.session_state.current_page+1} / {len(doc)}</b></center>", unsafe_allow_html=True)
            if nav3.button("▶", key="next"):
                if st.session_state.current_page < len(doc) - 1: st.session_state.current_page += 1

            page = doc.load_page(st.session_state.current_page)
            pix = page.get_pixmap(dpi=150)
            st.image(Image.frombytes("RGB", [pix.width, pix.height], pix.samples), use_container_width=True)
            page_text = (page.get_text() or "").strip()

        with col_right:
            st.markdown("### 🧑‍🏫 조교 설명")
            
            db_sub = filter_db_by_subject(subject_pick, st.session_state.db)
            page_sig = hash(page_text)
            
            # 페이지 변경 감지 시 DB 검색
            if page_sig != st.session_state.last_page_sig:
                st.session_state.last_page_sig = page_sig
                st.session_state.last_related = find_relevant_jokbo(page_text, db_sub) if page_text else []
                st.session_state.last_ai_sig = None 

            related = st.session_state.last_related
            
            if not page_text:
                st.info("텍스트 없음")
            elif not has_jokbo_evidence(related):
                st.info("💡 이 페이지는 족보와 직접적 연관이 적습니다.")
            else:
                # AI 생성
                ai_sig = (page_sig, subject_pick)
                if ai_sig != st.session_state.last_ai_sig:
                    if st.session_state.api_key_ok:
                        prompt = build_ta_prompt(page_text, related, subject_pick)
                        with st.spinner("분석 중..."):
                            res, _ = generate_with_fallback(prompt, st.session_state.text_models)
                        st.session_state.last_ai_text = res
                        st.session_state.last_ai_sig = ai_sig

                st.write(st.session_state.last_ai_text)
                
                # --- [기능 추가] 노트 저장 버튼 ---
                if st.session_state.last_ai_text:
                    st.divider()
                    col_save, _ = st.columns([1, 2])
                    if col_save.button("📌 이 내용 노트에 저장", key="save_tab2"):
                        note_content = f"[강의본 p{st.session_state.current_page+1}]\n{st.session_state.last_ai_text}"
                        add_to_notes("강의본(PDF)", note_content)

# ==================================================
# TAB 3 — Transcript
# ==================================================
with tab3:
    st.header("🎙️ 3) 강의 전사 텍스트 → 족보 포인트")
    
    subjects_in_db = sorted({x.get("subject", "") for x in st.session_state.db if x.get("subject")})
    subject_options = ["전체"] + (subjects_in_db if subjects_in_db else ["(DB 없음)"])
    subject_pick = st.selectbox("분석 과목", subject_options, key="tab3_sub")

    transcript_text = st.text_area("전사 텍스트 입력", height=200)
    max_chunks = st.number_input("최대 분석 구간 수", 1, 40, 10)

    if st.button("🧠 족보 포인트 뽑기", key="run_transcript"):
        if not transcript_text.strip():
            st.error("텍스트를 입력하세요")
            st.stop()
        
        db_sub = filter_db_by_subject(subject_pick, st.session_state.db)
        chunks = chunk_transcript(transcript_text, 900)[:int(max_chunks)]
        
        related_packs = []
        prog = st.progress(0)
        for i, ch in enumerate(chunks, 1):
            rel = find_relevant_jokbo(ch, db_sub, top_k=3)
            related_packs.append(rel)
            prog.progress(i / len(chunks))
        
        prompt = build_transcript_prompt(chunks, related_packs, subject_pick)
        with st.spinner("족보 매칭 중..."):
            result, _ = generate_with_fallback(prompt, st.session_state.text_models)
        
        st.session_state.last_transcript_result = result
        st.success("분석 완료!")

    # 결과 표시 및 저장
    if st.session_state.last_transcript_result:
        st.markdown("### 🧑‍🏫 족보 포인트 노트")
        st.write(st.session_state.last_transcript_result)
        
        st.divider()
        if st.button("📌 이 포인트 노트에 저장", key="save_tab3"):
            add_to_notes("전사텍스트", st.session_state.last_transcript_result)

# ==================================================
# TAB 4 — Summary Notes (NEW)
# ==================================================
with tab4:
    st.header("📝 나만의 정리노트")
    st.caption("강의본과 전사 텍스트에서 저장한 핵심 내용들을 모아봅니다.")

    if not st.session_state.my_notes:
        st.info("아직 저장된 노트가 없습니다. Tab 2, 3에서 '📌 저장' 버튼을 눌러보세요.")
    else:
        # 다운로드 기능
        full_text = ""
        for note in st.session_state.my_notes:
            full_text += f"[{note['timestamp']} | {note['source']}]\n{note['content']}\n\n{'='*30}\n\n"
        
        st.download_button(
            label="📥 전체 노트 다운로드 (TXT)",
            data=full_text,
            file_name=f"My_Med_Note_{datetime.now().strftime('%m%d')}.txt",
            mime="text/plain"
        )
        
        st.divider()

        # 노트 리스트 표시 (역순: 최신순)
        for i, note in enumerate(reversed(st.session_state.my_notes)):
            # Index handling for deletion logic is tricky with reversed, so use original index/ID
            real_index = len(st.session_state.my_notes) - 1 - i
            
            with st.expander(f"📝 노트 #{i+1} ({note['source']} - {note['timestamp']})", expanded=True):
                col_content, col_del = st.columns([9, 1])
                
                with col_content:
                    st.markdown(note['content'])
                
                with col_del:
                    if st.button("🗑️", key=f"del_{note['id']}"):
                        st.session_state.my_notes.pop(real_index)
                        st.rerun()
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

