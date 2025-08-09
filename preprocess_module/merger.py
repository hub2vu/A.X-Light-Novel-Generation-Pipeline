import os
import re

# === 설정 ===
input_folder = "./folder"  # 폴더 경로를 실제 폴더 위치로 바꿔주세요
output_file = "merged_output.txt"  # 병합된 결과 파일 이름

# === 파일 목록 가져오기 ===
files = os.listdir(input_folder)

# === output_n.txt 형식 필터링 및 숫자 추출 ===
pattern = re.compile(r"output_(\d+)\.txt")
numbered_files = []
for fname in files:
    match = pattern.match(fname)
    if match:
        number = int(match.group(1))
        numbered_files.append((number, fname))

# === 숫자 기준 정렬 ===
numbered_files.sort(key=lambda x: x[0])

# === 파일 내용 병합 ===
with open(output_file, "w", encoding="utf-8") as outfile:
    for number, fname in numbered_files:
        path = os.path.join(input_folder, fname)
        with open(path, "r", encoding="utf-8") as infile:
            outfile.write(infile.read())
            outfile.write("\n")  # 각 파일 끝에 개행 추가 (선택사항)
