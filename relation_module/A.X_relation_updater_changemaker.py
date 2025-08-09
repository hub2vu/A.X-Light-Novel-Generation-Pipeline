import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === 설정 ===
base_model_name = "skt/A.X-4.0-Light"
story_file = "../new_story_edited.txt"
relation_file = "../rag_corpus/relation.txt"
character_file = "../rag_corpus/character.txt"
relation_update_file = "./rag_corpus/relation_update.txt"

# === character.txt에서 이름 추출 ===
def extract_character_names(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    names = []
    for line in lines:
        match = re.match(r"^\d+\.\s*(\S+)", line.strip())
        if match:
            names.append(match.group(1))
    return names

character_list = extract_character_names(character_file)

# === 모델 로딩 ===
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.bfloat16)
model.eval()

# === 소설 불러오기 ===
with open(story_file, encoding="utf-8") as f:
    story_text = f.read()

# === 프롬프트 구성 ===
system_prompt = "너는 추리소설 분석가야. 주어진 소설 내용을 기반으로 등장인물 간의 관계 변화를 분석해."
prompt = f"""
[SYSTEM]
{system_prompt}

[USER]
다음은 이번 화 소설 내용이야. 등장인물은 {', '.join(character_list)} 이고, 등장인물 간의 관계 변화가 있으면 요약해줘.
기존에 없던 관계가 새로 드러나거나, 갈등/협력이 생긴 경우 구체적으로 서술해줘. 형식은 "A - B: 관계내용" 형태로 써줘.

[STORY]
{story_text}
"""

# === 모델 입력 ===
input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

# === 생성 ===
with torch.no_grad():
    output = model.generate(
        input_ids.input_ids,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95
    )

generated = tokenizer.decode(output[0][input_ids.input_ids.shape[1]:], skip_special_tokens=True)
print("🔁 추출된 관계 변화 [최초생성]:\n", generated.strip())

# === 관계 없는 내용 제거 ==
def filter_meaningful_relations(text):
    lines = text.strip().split("\n")
    filtered = []
    for line in lines:
        if any(kw in line for kw in ["두 인물 간의 관계 변화를 추론하기 어렵다"]):
            continue
        if "-" in line and ":" in line:
            filtered.append(line.strip())
    return "\n".join(filtered)

filtered_generated = filter_meaningful_relations(generated)
print("🔁 추출된 관계 변화 [무관계필터]:\n", filtered_generated.strip())

# === 관계 변화 저장 (이름순 정렬 적용) ===
with open(relation_update_file, "w", encoding="utf-8") as f:
    for line in filtered_generated.strip().split("\n"):
        line = line.strip()
        match = re.match(r"^(\S+)\s*-\s*(\S+)\s*:\s*(.+)", line)
        if match:
            char1, char2, desc = match.groups()
            char1, char2 = sorted([char1, char2])  # 이름 순 정렬
            f.write(f"{char1} - {char2}: {desc.strip()}\n")


print(f"✅ 새로운 관계 변화 저장 완료 → {relation_update_file}")
