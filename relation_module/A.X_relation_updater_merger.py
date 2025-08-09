import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === 설정 ===
base_model_name = "skt/A.X-4.0-Light"
relation_file = "./rag_corpus/relation.txt"
relation_update_file = "./rag_corpus/relation_update.txt"
relation_merged_file = "./rag_corpus/relation_merged.txt"

# === 모델 로딩 ===
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.bfloat16)
model.eval()

# === 기존 관계 로드 ===
def load_existing_relations(path):
    rel_dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            match = re.match(r"\s*(\S+)\s*-\s*(\S+)\s*(.+)", line.strip())
            if match:
                a, b, desc = match.groups()
                a = a.rstrip(":")
                b = b.rstrip(":")
                key = frozenset([a, b])  # 순서 무시
                rel_dict[key] = {"A": a, "B": b, "desc": desc.strip()}
    return rel_dict

# === 새로운 관계 변화 로드 ===
def load_relation_updates(path):
    updates = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            match = re.match(r"(\S+)\s*-\s*(\S+)\s*:\s*(.+)", line.strip())
            if match:
                a, b, change = match.groups()
                a = a.strip().rstrip(":")
                b = b.strip().rstrip(":")
                change = change.strip()
                key = frozenset([a, b])
                updates[key] = {"A": a, "B": b, "change": change.strip()}
    return updates

# === 통합 프롬프트 기반 재생성 ===
def regenerate_relation_description(a, b, past, new):
    prompt = f"""
[SYSTEM]
너는 소설 등장인물 간의 관계를 분석하는 조력자야.

[USER]
기존 관계 설명: "{past}"
이번 화에서 드러난 새로운 관계 변화: "{new}"
이 정보를 바탕으로 {a}와 {b}의 관계를 약 300자 이내로 통합해서 재작성해줘.
"""

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids.input_ids,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.95
        )
    result = tokenizer.decode(output[0][input_ids.input_ids.shape[1]:], skip_special_tokens=True)
    return result.strip()

# === 실행 ===
existing_rel = load_existing_relations(relation_file)
updates = load_relation_updates(relation_update_file)
merged = {}

for key in updates:
    a, b = updates[key]["A"], updates[key]["B"]
    a, b = sorted([a, b])
    past_desc = existing_rel.get(key, {}).get("desc", "없음")
    new_change = updates[key]["change"]
    new_desc = regenerate_relation_description(a, b, past_desc, new_change)
    merged[key] = f"{a} - {b}: {new_desc}"
print(merged) ###
for key in merged:
    print("update :", key)###
# === 기존 관계 중 업데이트되지 않은 것 보존 ===
for key in existing_rel:
    if key not in merged:
        print("Not update :", key) ###
        a, b = existing_rel[key]["A"], existing_rel[key]["B"]
        a, b = sorted([a, b]) #정렬
        desc = existing_rel[key]["desc"]
        merged[key] = f"{a} - {b}: {desc}"

# === 저장 ===
with open(relation_merged_file, "w", encoding="utf-8") as f:
    for line in merged.values():
        f.write(line.strip() + "\n")

print(f"✅ 관계 통합 완료 → {relation_merged_file} (총 {len(merged)}개)")
