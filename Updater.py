import subprocess

scripts = [
    "./relation_module/A.X_relation_updater_changemaker.py",
    "./relation_module/A.X_relation_updater_merger.py",
    "./updater_module/story_stacker.py"
]

for script in scripts:
    print(f"[실행 중] {script}")
    result = subprocess.run(["python", script], capture_output=True, text=True)

    # 실행 결과 출력
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # 오류 발생 시 중단
    if result.returncode != 0:
        print(f"[오류 발생] {script} 실행 실패 (코드: {result.returncode})")
        break
