import os

# 경로 설정
new_story_path = "../new_story_edited.txt"
last_story_path = "../rag_corpus/last_story.txt"

# 파일 존재 여부 확인
if not os.path.exists(new_story_path):
    raise FileNotFoundError(f"새로운 스토리 파일을 찾을 수 없습니다: {new_story_path}")

# 디렉토리 생성 (없으면)
os.makedirs(os.path.dirname(last_story_path), exist_ok=True)

# 새로운 스토리 읽기
with open(new_story_path, "r", encoding="utf-8") as f:
    new_story_content = f.read().strip()

if not new_story_content:
    raise ValueError("새로운 스토리 파일이 비어 있습니다.")

# 기존 last_story.txt에 추가
with open(last_story_path, "a", encoding="utf-8") as f:
    # 기존 내용이 있다면 줄바꿈 추가
    if os.path.getsize(last_story_path) > 0:
        f.write("\n")
    f.write(new_story_content)

print(f"'{new_story_path}' 내용을 '{last_story_path}'의 끝에 추가 완료.")
