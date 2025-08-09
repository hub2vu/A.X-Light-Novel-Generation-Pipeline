import os

# 기본 경로 설정
base_dir = r"C:\Users\hub2v\Downloads\코난도일"   # 작품별 폴더들이 들어 있는 최상위 폴더
output_dir = r"C:\Users\hub2v\Downloads\output_chunks"  # 모든 분할 결과가 저장될 폴더

os.makedirs(output_dir, exist_ok=True)

def split_text(text, chunk_size=4000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# 작품별 하위 폴더 순회
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):
            input_path = os.path.join(folder_path, filename)

            # 인코딩 자동 감지 또는 euc-kr 시도
            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(input_path, "r", encoding="euc-kr", errors="ignore") as f:
                    text = f.read()

            chunks = split_text(text, 4000)
            name_wo_ext = os.path.splitext(filename)[0]

            for i, chunk in enumerate(chunks):
                output_filename = f"{folder_name}_{name_wo_ext}_part{i+1}.txt"
                output_path = os.path.join(output_dir, output_filename)
                with open(output_path, "w", encoding="utf-8") as out_f:
                    out_f.write(chunk)

            print(f"✅ 분할 완료: {folder_name}/{filename} → {len(chunks)}개")
