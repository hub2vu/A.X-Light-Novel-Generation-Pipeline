import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
import re
# --- 파일 경로 설정 ---
OUTLINE_FILE = "../rag_corpus/outline.txt"
WORLD_CONFIG_FILE = "../rag_corpus/world_config.txt"
LAST_STORY_FILE = "../rag_corpus/last_story.txt"
OUTPUT_SETTING_FILE = "../rag_corpus/setting.txt"

# --- 모델 및 토크나이저 설정 ---
MODEL_NAME = "skt/A.X-4.0-Light"


# --- 헬퍼 함수 ---
def get_last_line(file_path):
    """파일의 마지막 줄을 읽어 반환합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            return lines[-1] if lines else None
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return None


def read_file_content(file_path):
    """파일의 전체 내용을 읽어 반환합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return None


def append_to_file(file_path, content):
    """파일 끝에 내용을 추가합니다. 파일이 비어있지 않으면 먼저 빈 줄을 하나 추가합니다."""
    try:
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 파일 끝에 추가
        with open(file_path, 'a', encoding='utf-8') as f:
            f.seek(0, 2)  # 파일 끝으로 이동
            # 파일이 비어있을 경우, 바로 내용 추가
            if f.tell() == 0:
                f.write(content)
            else:
                # 파일이 비어있지 않으면, 항상 두 개의 줄바꿈을 추가하여 빈 줄을 만듦
                f.write('\n' + content + '\n--')

        print(f"\n성공적으로 '{file_path}' 파일에 최종 배경 설정을 추가했습니다.")
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")


# --- 메인 로직 ---
def main():
    # 모델 및 토크나이저 로드
    print("모델 및 토크나이저를 로드합니다...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        return

    # --- 1단계: LLM을 이용한 초기 배경 아이디어 생성 ---
    print("\n--- 1단계: LLM으로 초기 배경 아이디어를 생성합니다. ---")

    # 입력 데이터 로드
    latest_outline = get_last_line(OUTLINE_FILE)
    world_config = read_file_content(WORLD_CONFIG_FILE)

    if not latest_outline or not world_config:
        print("필요한 파일(아웃라인 또는 세계관)이 없거나 비어있어 실행을 중단합니다.")
        return

    # 프롬프트 구성
    prompt_step1 = f"""<s>[PROMPT]
[SYSTEM]
너는 주어진 세계관과 최신 스토리 아웃라인을 바탕으로, 이 아웃라인에 가장 어울리는 구체적인 공간 배경과 시간대를 생성하는 AI 작가야.

[세계관 정보]
{world_config}

[최신 스토리 아웃라인]
{latest_outline}

[요청]
위 정보를 바탕으로, 다음 이야기가 펼쳐질 장소와 시간대를 구체적으로 묘사해줘. 예를 들어 '비가 내리는 늦은 밤, 낡은 항구의 창고 안' 또는 '해가 저무는 저녁, 인적 드문 공원의 벤치'와 같이 분위기가 느껴지도록 한 문장으로 간결하게 만들어줘.
[/PROMPT]
[SETTING]
"""

    # 배경 아이디어 생성
    inputs1 = tokenizer(prompt_step1, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs1 = model.generate(
            inputs1.input_ids,
            max_new_tokens=100,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    generated_setting_query = tokenizer.decode(outputs1[0][len(inputs1.input_ids[0]):],
                                               skip_special_tokens=True).strip()

    if not generated_setting_query:
        print("LLM이 초기 배경 아이디어를 생성하지 못했습니다. 실행을 중단합니다.")
        return
    print(f"  [생성된 초기 아이디어]: {generated_setting_query}")

    # --- 2단계: RAG를 이용한 과거 유사 배경 검색 ---
    print("\n--- 2단계: 생성된 아이디어를 쿼리로 사용하여 과거 스토리에서 유사한 배경을 검색합니다. ---")
    retrieved_context = "참고할 만한 과거 장면을 찾지 못했습니다."  # 기본값
    try:
        loader = TextLoader(LAST_STORY_FILE, encoding="utf-8")
        docs = loader.load()
        if docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = text_splitter.split_documents(docs)
            embedding_model = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")
            vectorstore = FAISS.from_documents(split_docs, embedding_model)
            retrieved_docs = vectorstore.similarity_search(generated_setting_query, k=3)
            if retrieved_docs:
                retrieved_context = retrieved_docs[0].page_content
    except FileNotFoundError:
        print(f"  [정보] '{LAST_STORY_FILE}' 파일이 없어 RAG를 건너뜁니다.")
    except Exception as e:
        print(f"  [오류] RAG 수행 중 오류 발생: {e}")
    print(f"  [검색된 유사 장면]:\n{retrieved_context}")

    # --- 3단계: LLM을 이용한 최종 배경 설정 생성 (수정된 부분) ---
    print("\n--- 3단계: 초기 아이디어와 검색된 장면을 종합하여 최종 배경 설정을 생성합니다. ---")

    prompt_step2 = f"""<s>[PROMPT]
[SYSTEM]
너는 AI 작가야. 주어진 참고 자료를 바탕으로 다음 화의 배경 설정을 구체적인 항목으로 요약하는 역할을 맡았어.

[과거 스토리의 유사 장면 (참고 자료)]
{retrieved_context}

[초기 배경 아이디어 (분위기 참고용)]
{generated_setting_query}

[요청]
[과거 스토리의 유사 장면 (참고 자료)]을 바탕으로 다음 화의 배경이 될 [일자], [시간], [날씨], [공간], [공간의 이름], [분위기]를 각각 간결하게 요약해줘.
[초기 배경 아이디어]는 참고 자료에 정보가 부족할 경우, 분위기나 방향성을 잡는 데만 참고하고, 구체적인 정보는 참고 자료를 우선으로 사용해줘.

예시:
일자: 사건 발생 다음 날
시간: 저녁 8시경
날씨: 비가 부슬부슬 내리는
공간: 오래된 도서관의 서고
공간의 이름: 시립 중앙 도서관
분위기: 고요하지만 무언가 숨겨져 있는 듯한, 묵직한
[/PROMPT]
[FINAL SETTING]
"""

    inputs2 = tokenizer(prompt_step2, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs2 = model.generate(
            inputs2.input_ids,
            max_new_tokens=150,  # 더 많은 정보를 위해 토큰 수 증가
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1
        )
    final_setting_raw = tokenizer.decode(outputs2[0][len(inputs2.input_ids[0]):], skip_special_tokens=True).strip()
    final_setting = re.sub(r'[\*#]', '', final_setting_raw)
    print(f"  [생성된 최종 배경 설정]:\n{final_setting}")

    # 4. 결과 저장
    append_to_file(OUTPUT_SETTING_FILE, final_setting)


if __name__ == "__main__":
    main()
