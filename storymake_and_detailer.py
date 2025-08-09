import subprocess
import os

# === 경로 설정 ===
story_script = os.path.join("story_generater_module", "A.X_LORA_generater.py")
detailer_script = os.path.join("detailer_module", "loop_detailer.py")

# === 첫 번째 스크립트 실행 ===
print(f"[1/2] 실행 중: {story_script}")
result1 = subprocess.run(["python", story_script], capture_output=True, text=True)

if result1.returncode != 0:
    print(f"[오류] {story_script} 실행 실패\n{result1.stderr}")
    exit(1)
else:
    print(result1.stdout)
    print(f"[완료] {story_script} 실행 성공\n")

# === 두 번째 스크립트 실행 ===
print(f"[2/2] 실행 중: {detailer_script}")
result2 = subprocess.run(["python", detailer_script], capture_output=True, text=True)

if result2.returncode != 0:
    print(f"[오류] {detailer_script} 실행 실패\n{result2.stderr}")
    exit(1)
else:
    print(result2.stdout)
    print(f"[완료] {detailer_script} 실행 성공\n")
