import subprocess
import os

# prompt_module 폴더 경로
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_module")

# 실행할 파일 목록
scripts = [
    "prompt_1outline_name_extractor.py",
    "prompt_2outline_generator.py",
    "prompt_3setting_maker.py",
    "prompt_4source_merger.py"
]

# 순차 실행
for script in scripts:
    script_path = os.path.join(base_dir, script)
    print(f"Prompt_generater 실행 중: {script_path}")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)

    # 출력 확인
    print(result.stdout)
    if result.stderr:
        print("오류 발생:")
        print(result.stderr)

    # 오류 시 중단
    if result.returncode != 0:
        print(f"실행 실패: {script}")
        break

print("모든 스크립트 실행 완료")
