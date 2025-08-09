# A.X Light Novel Generation Pipeline
이 저장소는 SKT의 **A.X Light** 한국어 언어 모델을 활용하여 웹소설/창작 소설을 자동으로 생성하는 파이프라인 예제입니다.  
데이터 전처리, 프롬프트 구성, 모델 호출, 결과 저장까지 전체 흐름을 포함합니다.

---

## Features
- Hugging Face Transformers 기반 A.X Light 로딩
- 대화형 프롬프트 구조로 소설 생성
- RAG(검색+생성) 또는 순수 생성 모드 지원
- 결과 자동 저장 및 로그 관리

---

## Installation
```bash
git clone https://github.com/hub2vu/ax-light-novel-pipeline.git
cd ax-light-novel-pipeline
pip install -r requirements.txt

---

## License

### Code
- 본 저장소의 **코드**는 [MIT License](LICENSE)를 따릅니다.

### Models
- **A.X Light** (skt/A.X-4.0-Light)  
  - License: CC BY-NC 4.0 (비영리적 사용만 가능)  
  - 출처: [Hugging Face - skt/A.X-4.0-Light](https://huggingface.co/skt/A.X-4.0-Light)

- **ko-sroberta-multitask** (jhgan/ko-sroberta-multitask)  
  - License: Apache License 2.0 (상업적 사용 가능, 저작권 표시 필요)  
  - 출처: [Hugging Face - jhgan/ko-sroberta-multitask](https://huggingface.co/jhgan/ko-sroberta-multitask)

---

## Notice
- 모델 가중치는 이 저장소에 포함되지 않으며, Hugging Face에서 직접 다운로드해야 합니다.
- 각 모델의 라이선스 조건을 반드시 준수해야 하며,  
  특히 **A.X Light** 모델은 상업적 사용이 금지됩니다.
