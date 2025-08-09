import json
import re

input_path = "train_data.txt"       # 원본 txt 경로
output_path = "train_data.jsonl"   # 결과 jsonl 경로

# 전체 파일 읽기
with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 중괄호로 시작하고 끝나는 JSON 블록만 추출 (재귀적 중괄호 매칭은 어려워서 단순한 방법 사용)
# 패턴: {"messages": [...]} 형태의 JSON 객체 전체
pattern = r'\{[\s\S]*?"messages"\s*:\s*\[[\s\S]*?\]\s*\}'

matches = re.findall(pattern, text)

valid_jsons = []
for match in matches:
    try:
        obj = json.loads(match)
        valid_jsons.append(obj)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 파싱 실패: {e}")
        print(match[:150], "...\n")

# JSONL로 저장
with open(output_path, 'w', encoding='utf-8') as f:
    for obj in valid_jsons:
        json.dump(obj, f, ensure_ascii=False)
        f.write('\n')

print(f"✅ 변환 완료! 총 {len(valid_jsons)}개의 JSON 객체가 저장되었습니다 → {output_path}")
