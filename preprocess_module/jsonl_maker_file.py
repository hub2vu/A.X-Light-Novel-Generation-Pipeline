import os
import json

# 설정
root_folder = "./converted_json"  # 상위 폴더 경로
output_file = "./train_data.jsonl"  # 결과 저장 경로

# 결과 파일 열기
with open(output_file, "w", encoding="utf-8") as outfile:
    # 하위 폴더 순회
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".json"):
                file_path = os.path.join(foldername, filename)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # 메시지 항목이 있는 경우만 추출
                    if isinstance(data, dict) and "messages" in data:
                        json_line = json.dumps({"messages": data["messages"]}, ensure_ascii=False)
                        outfile.write(json_line + "\n")
                        print(f"✅ 처리 완료: {file_path}")
                    else:
                        print(f"⚠️ messages 키 없음: {file_path}")
                except Exception as e:
                    print(f"❌ 에러 발생: {file_path} - {e}")
