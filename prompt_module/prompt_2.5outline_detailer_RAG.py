import re
from pathlib import Path
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document  # ✅ 직접 Document 생성

# ===============================
# 경로 설정 (절대경로로 확정)
# ===============================
BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = (BASE_DIR / ".." / "rag_corpus").resolve()

OUTLINE_SRC = (RAG_DIR / "outline.txt")
LAST_STORY = (RAG_DIR / "last_story.txt")
OUT_PATH = (RAG_DIR / "outline_update.txt")

print("[DEBUG] CWD:", Path.cwd())
print("[DEBUG] LAST_STORY path:", LAST_STORY)
print("[DEBUG] exists:", LAST_STORY.exists())

# ===============================
# 안전한 파일 로딩
# ===============================
def safe_read_text(p: Path) -> str:
    # 인코딩 폴백: utf-8 → utf-8-sig → cp949
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    # 그래도 안되면 바이너리 읽고 대충 복구
    return p.read_bytes().decode("utf-8", errors="replace")

# ===============================
# 최신 아웃라인 1줄
# ===============================
def read_latest_outline(path: Path) -> str:
    if not path.exists():
        return ""
    lines = safe_read_text(path).strip().splitlines()
    return lines[-1].strip() if lines else ""

# ===============================
# RAG: last_story에서 관련 문맥 추출
# (TextLoader 대신 직접 읽어서 Document로 구성)
# ===============================
EMB_MODEL = "jhgan/ko-sroberta-multitask"
CHUNK_CHARS = 450
CHUNK_OVERLAP = 120
TOP_K = 5
MAX_CONTEXT_CHARS = 1400

def rag_search_last_story(query: str) -> List[str]:
    if not LAST_STORY.exists():
        print("[WARN] last_story.txt가 존재하지 않습니다:", LAST_STORY)
        return []

    text = safe_read_text(LAST_STORY)
    if not text.strip():
        print("[WARN] last_story.txt 내용이 비어 있습니다.")
        return []

    # 문서 → 청크
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP)
    # 직접 Document 구성
    docs = [Document(page_content=text)]
    pieces = splitter.split_documents(docs)

    # 인덱스
    emb = SentenceTransformerEmbeddings(model_name=EMB_MODEL)
    db = FAISS.from_documents(pieces, emb)

    # 검색
    hits = db.similarity_search_with_score(query, k=TOP_K)
    contents = [doc.page_content.strip() for doc, _ in hits if doc and doc.page_content]

    # 중복 제거 + 길이 제한
    uniq, seen = [], set()
    for c in contents:
        key = re.sub(r"\s+", " ", c)[:200]
        if key not in seen:
            uniq.append(c); seen.add(key)

    acc, total = [], 0
    for c in uniq:
        if total + len(c) + 2 <= MAX_CONTEXT_CHARS:
            acc.append(c); total += len(c) + 2
        else:
            break
    return acc

# ===============================
# A.X Light 로드/생성 함수 (기존 그대로)
# ===============================
LM_NAME = "skt/A.X-4.0-Light"
MAX_NEW_TOKENS = 280
TEMPERATURE = 0.7
TOP_P = 0.9

def load_lm():
    tok = AutoTokenizer.from_pretrained(LM_NAME, use_fast=False)
    mdl = AutoModelForCausalLM.from_pretrained(
        LM_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
        mdl.resize_token_embeddings(len(tok))
    return tok, mdl

def generate_updated_outline(latest_outline: str, related_contexts: List[str]) -> str:
    tok, mdl = load_lm()
    context_block = "\n\n---\n\n".join(related_contexts) if related_contexts else "관련 컨텍스트 없음"

    system = (
        "너는 소설의 아웃라인 편집자다. "
        "주어진 최신 아웃라인의 핵심 전개를 유지하되, 제공된 과거 스토리 문맥 중 "
        "진행과 일관성에 꼭 필요한 사실/단서만 선별하여 아웃라인에 반영하라. "
        "고유명사/관계/시간 흐름을 바꾸지 말고, 중언부언과 과도한 설정 확장을 피하라. "
        "최종 출력은 한 문단(2~3문장 이내)으로 간결하게 작성하라."
    )
    user = (
        f"[LATEST OUTLINE]\n{latest_outline}\n\n"
        f"[RELATED CONTEXT FROM LAST_STORY]\n{context_block}\n\n"
        "[REQUEST]\n"
        "- 최신 아웃라인의 핵심을 보존하되, 관련 컨텍스트에서 일치/보강되는 정보만 통합해 업데이트된 아웃라인을 작성.\n"
        "- 스포일러성 결말 확정 금지, 향후 전개 여지 남길 것.\n"
        "- 결과는 2~3문장 한 문단으로."
    )

    messages = [{"role":"system","content":system}, {"role":"user","content":user}]
    inp = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(mdl.device)

    with torch.no_grad():
        out = mdl.generate(
            inp,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id
        )
    gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
    gen = re.sub(r"[\*#]+", "", gen).strip()
    return gen

# ===============================
# 메인
# ===============================
if __name__ == "__main__":
    latest = read_latest_outline(OUTLINE_SRC)
    if not latest:
        print("[ERROR] outline.txt에서 최신 아웃라인을 찾지 못했습니다. 경로:", OUTLINE_SRC)
        raise SystemExit(1)

    print("[INFO] 최신 아웃라인:", latest)
    print("[INFO] last_story에서 관련 문맥 검색 중…")
    contexts = rag_search_last_story(latest)

    if contexts:
        print(f"[INFO] 관련 문맥 {len(contexts)}개 수집")
    else:
        print("[WARN] 관련 문맥을 찾지 못해, 최신 아웃라인만 기반으로 갱신을 시도합니다.")

    updated = generate_updated_outline(latest, contexts)

    # === outline.txt 마지막 줄 삭제 후 업데이트 ===
    outline_lines = safe_read_text(OUTLINE_SRC).splitlines()
    if outline_lines:
        outline_lines = outline_lines[:-1]  # 마지막 줄 제거
    outline_lines.append(updated)  # 새로운 아웃라인 추가

    OUTLINE_SRC.write_text("\n".join(outline_lines) + "\n", encoding="utf-8")
    print(f"outline.txt 마지막 줄이 갱신되었습니다.\n추가된 내용: {updated}")
