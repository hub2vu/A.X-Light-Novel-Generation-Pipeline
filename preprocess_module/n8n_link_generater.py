from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import uvicorn

app = FastAPI()

# 모델 로드
base_model = AutoModelForCausalLM.from_pretrained("skt/A.X-4.0-Light", torch_dtype=torch.float16, device_map="auto")
peft_model = PeftModel.from_pretrained(base_model, "./Qlora/ax-qlora-adapter_700")
tokenizer = AutoTokenizer.from_pretrained("skt/A.X-4.0-Light")


@app.post("/generate")
async def generate(req: Request):
    data = await req.json()
    messages = data.get("messages", [])

    # prompt 구성
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"[SYSTEM]\n{content}\n\n"
        elif role == "user":
            prompt += f"[USER]\n{content}\n\n"
        elif role == "assistant":
            prompt += f"[ASSISTANT]\n{content}\n\n"

    #파라미터
    temperature = data.get("temperature", 0.5)#랜덤성 - 낮을수록 프롬프트를 따르는 경향 ; 엄격 : 0.2~0.5 ; 창의 : 0.7~1.0
    top_p = data.get("top_p", 0.95)#0.8~0.9 낮을수록 프롬프트를 따르는 경향
    top_k = data.get("top_k", 50)#20~50 낮을수록 프롬프트를 따르는 경향
    max_new_tokens = data.get("max_new_tokens", 1024) #길이 짧을수록 프롬프트를 따르는 경향

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
        output = peft_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
    generated = output[0][input_ids.shape[1]:]  # 프롬프트 길이만큼 자르기
    result = tokenizer.decode(generated, skip_special_tokens=True)
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
