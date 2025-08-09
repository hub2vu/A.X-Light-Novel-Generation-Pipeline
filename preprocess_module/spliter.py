import os

input_folder = "./txt_origin"
output_folder = "./output_chunks"
os.makedirs(output_folder, exist_ok=True)

# 줄 단위로 길이 맞춰 분할
def split_text_lines(text, max_len=2000):
    lines = text.splitlines()
    chunks = []
    current_chunk = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(current_chunk) + len(line) + 1 <= max_len:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# 모든 txt 파일 처리
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_folder, filename)

        # ✅ 이제부터는 utf-8로 읽기!
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = split_text_lines(text, max_len=2000)

        base_name = os.path.splitext(filename)[0]
        for i, chunk in enumerate(chunks):
            output_filename = f"{base_name}_part{i+1}.txt"
            output_path = os.path.join(output_folder, output_filename)
            with open(output_path, "w", encoding="utf-8") as out_f:
                out_f.write(chunk)

        print(f"✅ 분할 완료: {filename} → {len(chunks)}개 파일 생성")
