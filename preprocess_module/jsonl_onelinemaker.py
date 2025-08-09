import json

input_path = "./train_data.jsonl"  # 사실 .json 형식임
output_path = "./train_data_fixed.jsonl"

# 전체 파일을 읽어서 여러 JSON 객체 파싱
with open(input_path, "r", encoding="utf-8") as f:
    raw_data = f.read()

# 여러 JSON 객체가 붙어 있는 경우 분리
chunks = raw_data.split("}\n{")
cleaned = []

for i, chunk in enumerate(chunks):
    # JSON 객체 복원
    if not chunk.startswith("{"):
        chunk = "{" + chunk
    if not chunk.endswith("}"):
        chunk = chunk + "}"
    try:
        item = json.loads(chunk)
        # message에서 user와 assistant만 추출
        prompt = ""
        story = ""
        for m in item.get("messages", []):
            if m["role"] == "user":
                prompt = m["content"]
            elif m["role"] == "assistant":
                story = m["content"]
        if prompt and story:
            cleaned.append({"prompt": prompt, "story": story})
    except Exception as e:
        print(f"❌ JSON 파싱 실패 (객체 {i}): {e}")
        continue

# JSONL로 저장
with open(output_path, "w", encoding="utf-8") as f:
    for item in cleaned:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 총 {len(cleaned)}개의 항목이 저장되었습니다.")
