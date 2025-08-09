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
