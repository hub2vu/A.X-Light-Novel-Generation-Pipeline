import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ast
import os

# === 설정 ===
base_model_name = "skt/A.X-4.0-Light"
lora_adapter_path = "../Qlora/ax-qlora-adapter_hot"
lora_alpha_scaling = 0.65  # LoRA 반영 비율
USE_LORA = True  #  True면 LoRA 적용, False면 베이스 모델만 사용
output_file = "../rag_corpus/new_story.txt"  # 출력 저장 파일 경로

# === 모델 & 토크나이저 로드 ===
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,     # fp16 사용
    device_map="auto",
)

# pad 토큰 안전장치
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))

# === LoRA 적용 여부 ===
if USE_LORA:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, lora_adapter_path)

    # LoRA 반영 비율 조절
    for name, module in model.named_modules():
        if hasattr(module, "base_layer") and hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            module.scale = lora_alpha_scaling

model.eval()

# === 프롬프트 불러오기 ===
prompt_path = "../rag_corpus/last_prompt.txt"
with open(prompt_path, "r", encoding="utf-8") as f:
    content = f.read()
    messages = ast.literal_eval(content)  # 안전한 파싱

# === 인풋 토크나이즈 ===
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# === 생성 ===
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        top_k=80,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

# === 결과 디코딩 ===
len_input = input_ids.shape[1]
generated_text = tokenizer.decode(output[0][len_input:], skip_special_tokens=True)

# === 결과 출력 ===
print(generated_text)

# === 결과 저장 ===
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    f.write(generated_text)

print(f"\n[저장 완료] 생성된 스토리가 '{output_file}'에 저장되었습니다.")
