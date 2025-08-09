from pathlib import Path

# 경로 설정
base_path = Path("../rag_corpus")
outline_path = base_path / "outline.txt"
character_path = base_path / "outline_name_extract.txt"
setting_path = base_path / "setting.txt"
last_prompt_path = base_path / "last_prompt.txt"

# outline.txt 마지막 줄 읽기
outline_lines = outline_path.read_text(encoding="utf-8").strip().splitlines()
last_outline = outline_lines[-1] if outline_lines else ""

# outline_name_extract.txt 내용 읽기
character_content = character_path.read_text(encoding="utf-8").strip()

# setting.txt에서 마지막 두 개의 '--' 사이 내용 읽기
setting_text = setting_path.read_text(encoding="utf-8")
setting_parts = setting_text.split("--")
setting_content = ""
if len(setting_parts) >= 3:
    setting_content = setting_parts[-2].strip()

# messages 형식 생성
messages = [
    {
        "role": "system",
        "content": "너는 아웃라인과 캐릭터 및 세계관 정보에 따라, 한국어 소설 원문을 자연스럽게 이어 써주는 소설 생성 인공지능이다. 문체는 정제되어 있고 몰입감 있게 구성한다."
    },
    {
        "role": "user",
        "content": f"[OUTLINE] {last_outline}\n[CHARACTERS] {character_content}\n[SETTING] {setting_content}"
    }
]

# last_prompt.txt에 마지막 줄 다음 줄에 저장
if last_prompt_path.exists():
    with last_prompt_path.open("a", encoding="utf-8") as f:
        f.write("\n" + str(messages))
else:
    last_prompt_path.write_text(str(messages), encoding="utf-8")

print(f"저장 완료 → {last_prompt_path}")