import os
import zipfile

# 압축 파일이 모여있는 폴더 경로를 지정하세요!
target_folder = r"C:\Users\hub2v\Desktop\zipfile"  # ← 원하는 경로로 변경

# 폴더 내 모든 zip 파일 반복
for filename in os.listdir(target_folder):
    if filename.lower().endswith(".zip"):
        zip_path = os.path.join(target_folder, filename)
        # zip 파일명과 동일한 폴더 생성 (이미 있으면 그대로)
        extract_dir = os.path.join(target_folder, os.path.splitext(filename)[0])
        os.makedirs(extract_dir, exist_ok=True)
        print(f"⏳ 압축 해제 중: {filename} → {extract_dir}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

print("✅ 모든 압축 해제 완료!")

