import os
import re

# 설정
input_folder = "./ntr용사json"  # ✅ JSON 파일들이 들어있는 폴더 경로

# 모든 .json 파일 순회
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)

        # 파일 읽기
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # ===== 📌 1. 형식 정리 =====
        # \n이 2번 이상 반복되면 \n 하나로
        content = re.sub(r"\n{2,}", "\n", content)

        # '-'이 2번 이상 반복되면 '-' 하나로
        content = re.sub(r"-{2,}", "-", content)

        # '='이 2번 이상 반복되면 '=' 하나로
        content = re.sub(r"={2,}", "=", content)

        # '**' 제거
        content = re.sub(r"*{1,}", "", content)

        # "작품 후기" 등 지정된 문구 제거
        content = content.replace("작품 후기", "")
        content = content.replace("선작. 추천. 코멘. 쿠폰. 평점 주신 분들 감사합니다.", "")

        # ===== 📌 2. 긴 영문 제거 =====
        # (1) base64 유사 문자열 제거 (공백 없이 40자 이상)
        content = re.sub(r'\b[A-Za-z0-9+/=]{20,}\b', '', content)

        # (2) 공백 없는 영문/숫자 30자 이상 제거
        content = re.sub(r'\b[A-Za-z0-9]{20,}\b', '', content)

        # (3) 공백 포함된 영문/숫자 30자 이상 제거
        content = re.sub(r'\b[A-Za-z0-9\s]{20,}\b', '', content)

        # 결과 덮어쓰기
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"✅ 수정 완료: {filename}")