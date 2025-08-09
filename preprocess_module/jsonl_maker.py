import os
import json

# === 설정 ===
input_folder = "./json_folder"  # JSON 파일들이 있는 폴더 경로
output_path = "./train_data.jsonl"  # 출력할 JSONL 파일 경로

# === JSONL 생성 ===
with open(output_path, "w", encoding="utf-8") as jsonl_file:
    for fname in sorted(os.listdir(input_folder)):
        if not fname.endswith(".json"):
            continue

        full_path = os.path.join(input_folder, fname)
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 메시지 리스트가 있어야 유효
        if "messages" in data:
            json_line = json.dumps({"messages": data["messages"]}, ensure_ascii=False)
            jsonl_file.write(json_line + "\n")

print("✅ jsonl 파일 생성 완료:", output_path)
