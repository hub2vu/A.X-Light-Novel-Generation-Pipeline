import os
import re
import itertools
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 설정
character_path = "../rag_corpus/character.txt"
output_path = "../rag_corpus/relation.txt"
model_name = "skt/A.X-4.0-Light"

# 모델 로드
# NOTE: 아래 모델 로드 부분은 실행 환경에 맞게 설정해야 합니다.
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

#이름추출
def extract_character_names(path):
    """character.txt에서 인물 이름 목록을 추출합니다."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    names = []
    for line in lines:
        # 숫자. 뒤의 공백 이후, 줄 끝까지를 이름으로 인식
        match = re.match(r"^\d+\.\s*(.+)$", line.strip())
        if match:
            names.append(match.group(1))
    return names

# 숫자, 이름 패턴 기준으로 파싱 character.txt -> {이름 : 설명}
def parse_character_blocks(path):
    """character.txt를 파싱하여 {이름: 설명} 형태의 딕셔너리로 반환합니다."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # 숫자. 이름을 기준으로 분할하되, 번호와 이름을 다시 붙이기 위해 lookahead 사용
    blocks = re.split(r'(?=\d+\.\s*\S+)', text.strip())
    character_dict = {}  # 이름 -> 전체 설명 매핑
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        match = re.match(r"^\d+\.\s*(\S+)", lines[0])
        if match:
            name = match.group(1)
            character_dict[name] = block.strip()
    return character_dict

# char_dict는 스크립트 실행 시 한 번만 로드합니다.
# char_dict = parse_character_blocks(character_path)

# 두 인물char1, char2의 설명만 추출
def load_selected_character_info(char_dict, char1, char2):
    """두 인물의 설명 텍스트를 딕셔너리에서 불러옵니다."""
    parts = []
    for name in [char1, char2]:
        if name in char_dict:
            parts.append(char_dict[name])
    return "\n\n".join(parts)

def generate_relation_description(char_dict, char1, char2, max_retries=3, seed=42):
    char1, char2 = sorted([char1, char2])
    context = load_selected_character_info(char_dict, char1, char2)

    system = (
        "너는 한국어 소설 설계 도우미다. 두 인물의 성격/배경을 바탕으로 "
        #"현대 한국 배경의 추리물 톤으로"
        "관계를 요약한다."
    )
    user = (
        f"[인물1: {char1}] [인물2: {char2}]\n"
        f"[프로필]\n{context}\n\n"
        "요청:\n"
        "- 두 인물의 관계를 3~4문장으로, 단일 문단으로 서술.\n"
        "- 고유명사/설정 왜곡 금지, 사실 추정은 ‘가능성’/‘정황’ 어조로.\n"
        "- 마침표로 끝나는 문장만 사용.\n"
        "- 출력은 오직 관계 설명만."
    )

    messages = [{"role":"system","content":system},{"role":"user","content":user}]
    inp = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            inp,
            max_new_tokens=220,
            do_sample=True, temperature=0.7, top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
        )
    text = tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
    # 특수기호 제거 + 문장 수 제한
    text = re.sub(r"[＊*#]", "", text).strip()
    sents = re.split(r"(?<=[.!?。])\s+", text)
    text = " ".join(sents[:4]).strip()  # 최대 4문장
    return f"{char1} - {char2}: {text}"

# --- 실행 부분 ---
def main():
    """메인 실행 함수"""
    if not os.path.exists(character_path):
        print(f"오류: {character_path} 에서 캐릭터 파일을 찾을 수 없습니다.")
        return

    # 파일이 이미 존재할 경우, 덮어쓴다는 메시지를 출력합니다.
    if os.path.exists(output_path):
        print(f"알림: 기존 '{output_path}' 파일에 새로운 내용을 덮어씁니다.")

    print("캐릭터 데이터 로딩 중...")
    char_dict = parse_character_blocks(character_path)
    characters = list(char_dict.keys())
    pairs = list(itertools.combinations(characters, 2))
    print(f"총 {len(characters)}명의 인물을 찾았습니다. {len(pairs)}개의 관계를 생성합니다.")

    # 모델과 토크나이저가 로드되었는지 확인
    # if 'tokenizer' not in globals() or 'model' not in globals():
    #     print("오류: 모델과 토크나이저가 로드되지 않았습니다. 로드 부분을 확인해주세요.")
    #     return

    with open(output_path, "w", encoding="utf-8") as f:
        for i, (char1, char2) in enumerate(pairs):
            print(f"\n관계 {i+1}/{len(pairs)} 처리 중...")
            # char_dict를 인자로 전달합니다.
            description = generate_relation_description(char_dict, char1, char2)
            if "관계 정보 생성 실패" not in description:
                print(f"성공: {description}")
                f.write(description + "\n\n")
            else:
                print(f"실패: {description}")


    print(f"\n생성 완료. 총 {len(pairs)}개의 관계를 처리했으며 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    # 이 스크립트를 직접 실행할 때 main 함수가 호출됩니다.
    # 모델 로딩과 같은 무거운 작업은 이 블록 안이나 main 함수 시작 부분에 두는 것이 좋습니다.
    print("모델과 토크나이저 로딩 중...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        main()
    except Exception as e:
        print(f"설정 또는 실행 중 오류가 발생했습니다: {e}")
        print("필요한 라이브러리가 설치되어 있는지, 필요하다면 허깅페이스에 로그인되어 있는지 확인해주세요.")