import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === 설정 ===
input_folder = "./texts"
output_folder = "./output_json"
os.makedirs(output_folder, exist_ok=True)

# 모델 로드
model_name = "skt/A.X-4.0-Light"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# === 항목별 프롬프트 ===
def create_prompt(text, info_type):
    if info_type == "OUTLINE":
        instruction = "다음 소설 원문에서 중심 사건이나 줄거리를 명확히 요약한 아웃라인을 3~4 문장으로 작성해줘."
    elif info_type == "CHARACTERS":
        system_prompt = (
            "너는 소설 원문에서 등장인물들의 이름과 각 인물의 역할을 간략히 정리해주는 어시스턴트야. "
            "각 인물의 역할을 2~3 문장으로 작성해야 해."
        )
        user_prompt = f"다음 소설 원문에서 등장하는 인물들과 각 인물의 간략한 역할을 작성해줘.\n\n[소설 원문]\n{text}\n\n[CHARACTERS]\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages  # characters는 messages 형식으로 반환
    elif info_type == "SETTING":
        instruction = "다음 판타지 소설 원문에서 시대적 배경과 공간적 배경을 2~3 문장으로 작성해줘."
    else:
        raise ValueError("잘못된 info_type입니다.")

    prompt = f"{instruction}\n\n[소설 원문]\n{text}\n\n[{info_type}]\n"
    return prompt

# === 정보 추출 함수 ===
def generate_info(text, info_type, max_tokens=512):
    prompt = create_prompt(text, info_type)

    if info_type == "CHARACTERS":
        # apply_chat_template → chat 기반 프롬프트 처리
        input_ids = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        # ⬇️ attention mask 명시적으로 생성
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }  # dict 형태로 변환
    else:
        # 일반 텍스트 기반 프롬프트 처리
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.4,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    return result

# === JSON 생성 ===
def make_json(original_text, outline, characters, setting):
    return {
        "messages": [
            {
                "role": "system",
                "content": "너는 아웃라인과 캐릭터 및 세계관 정보에 따라, 한국어 소설 원문을 자연스럽게 이어 써주는 소설 생성 인공지능이다. 문체는 자연스럽고 시대적 배경에 맞게 쓴다."
            },
            {
                "role": "user",
                "content": f"[OUTLINE] {outline}\n[CHARACTERS] {characters}\n[SETTING] {setting}"
            },
            {
                "role": "assistant",
                "content": original_text
            }
        ]
    }


# === 전체 처리 루프 ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".txt"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".txt", ".json"))

        # 원본 소설 로드
        with open(input_path, encoding="utf-8") as f:
            original_text = f.read().strip()

        print(f"\n🔍 {filename} 처리 시작...")

        # 각 항목별로 독립적으로 생성
        print("📌 Outline 생성 중...")
        outline = generate_info(original_text, "OUTLINE")

        print("📌 Characters 생성 중...")
        characters = generate_info(original_text, "CHARACTERS", max_tokens=300)

        print("📌 Setting 생성 중...")
        setting = generate_info(original_text, "SETTING")

        # JSON 생성 및 저장
        json_data = make_json(original_text, outline, characters, setting)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        print(f"✅ {filename} → {output_path} 저장 완료.")
