import os
import shutil

# 1. 상위 폴더 경로 (drive로 시작하는 폴더들이 모여 있는 곳)
root_folder = r"C:\Users\hub2v\Desktop\zipfile"  # ← 네 상황에 맞게 수정!
# 2. 복사(이동) 대상 폴더
target_folder = r"C:\Users\hub2v\Desktop\datafol"  # 원하는 경로로!

os.makedirs(target_folder, exist_ok=True)

# 3. drive로 시작하는 폴더만 반복
for drive_folder in os.listdir(root_folder):
    if drive_folder.startswith("drive"):
        drive_path = os.path.join(root_folder, drive_folder)
        if os.path.isdir(drive_path):
            # 그 안에 있는 "모든 하위 폴더" 반복
            for subfolder in os.listdir(drive_path):
                subfolder_path = os.path.join(drive_path, subfolder)
                if os.path.isdir(subfolder_path):
                    dest_path = os.path.join(target_folder, subfolder)
                    # 중복 폴더명 방지
                    i = 1
                    orig_dest = dest_path
                    while os.path.exists(dest_path):
                        dest_path = f"{orig_dest}_{i}"
                        i += 1
                    shutil.copytree(subfolder_path, dest_path)
                    print(f"✅ 복사됨: {subfolder_path} → {dest_path}")

print(f"\n🎉 모든 하위 폴더가 '{target_folder}'에 복사 완료!")
