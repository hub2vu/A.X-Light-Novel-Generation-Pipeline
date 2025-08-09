import torch
#print(torch.cuda.is_available())
#print(torch.version.cuda)

#@title TEST

from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 저장된 모델/토크나이저 경로 지정
model_dir = "./kogpt2-finetuned-novel"  # 학습한 경로에 맞게 수정

# 2. 모델/토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir).to("cuda")

# 3. 테스트 프롬프트 입력 (요약문)
user_prompt = "프롬프트를 입력하세요"  # 예시

input_text = f"<s>[PROMPT] {user_prompt} [/PROMPT]\n[STORY]"

# 4. 토크나이즈 및 입력 생성
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

# 5. 생성!
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        max_length=1000,         # 생성할 최대 토큰 길이 (조절 가능)
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,         # 샘플링 여부
        top_k=50,               # 다양성 조절
        top_p=0.95,             # 다양성 조절
        temperature=0.8,        # 창의성/무작위성 조절
        repetition_penalty=1.1  # 반복 방지
    )

# 6. 결과 디코딩
result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# 7. 출력
print("=== 생성 결과 ===")
print(result)