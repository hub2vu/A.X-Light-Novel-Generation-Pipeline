import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import re
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# === 모델 로드 (QLoRA 없이 순수 A.X 사용) ===
model_name = "skt/A.X-4.0-Light"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# === 입력 데이터 로드 ===
character_text = Path("../rag_corpus/character.txt").read_text(encoding="utf-8").strip()
relation_text = Path("../rag_corpus/relation.txt").read_text(encoding="utf-8").strip()
outline_lines = Path("../rag_corpus/outline.txt").read_text(encoding="utf-8").strip().splitlines()
last_outline = outline_lines[-1].strip()

# === 프롬프트 메시지 생성 ===
messages = [
    {
        "role": "system",
        "content": "너는 소설을 이해하고 다음 화의 전개를 기획하는 소설 기획 어시스턴트야. 등장인물들의 성격, 관계, 이전 스토리의 흐름을 바탕으로 다음 화에 가장 자연스럽고 필연적으로 등장해야 할 인물 1~2명을 선정해줘."
    },
    {
        "role": "user",
        "content":
        f"[CHARACTERS]\n{character_text}\n\n"
        f"[RELATIONS]\n{relation_text}\n\n"
        f"[LAST OUTLINE]\n{last_outline}\n\n"
        "이 정보를 바탕으로 다음 화에 반드시 등장해야 할 인물 1~2명을 뽑고, 그 이유를 간단히 설명해줘."
    }
]

# === 토크나이즈
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

# === 생성
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1
    )

# === 디코딩 및 특수문자 제거
generated_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
generated_text = generated_text.replace("*", "")
generated_text = generated_text.replace("#", "")
# === 출력
print("다음 화 등장 인물 추천 결과:\n")
print(generated_text)

lines = generated_text.strip().splitlines()
if len(lines) >= 3:
    trimmed_text = "\n".join(lines[1:-1])  # 첫 줄과 마지막 줄 제거
else:
    trimmed_text = ""  # 너무 짧으면 공백 저장

print(trimmed_text)
# === 파일 저장
output_path = Path("../rag_corpus/outline_name_extract.txt")
output_path.write_text(trimmed_text, encoding="utf-8")
print(f"\n 결과가 {output_path} 에 저장되었습니다.")




