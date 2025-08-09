import subprocess

# 실행할 스크립트 경로
scripts = [
    "./character_module/A.X_generate_character_profiles.py",
    "./relation_module/A.X_relation_maker_2parcing.py"
]

for script in scripts:
    print(f"▶ 실행 중: {script}")
    try:
        subprocess.run(["python", script], check=True)
        print(f"✅ 완료: {script}\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ 오류 발생: {script}")
        print(e)
        break
