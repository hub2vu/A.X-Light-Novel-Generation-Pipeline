# A.X Light Novel Generation Pipeline
이 저장소는 SKT의 **A.X Light** 한국어 언어 모델을 활용하여 웹소설/창작 소설을 자동으로 생성하는 파이프라인 예제입니다.  
데이터 전처리, 프롬프트 구성, 모델 호출, 결과 저장까지 전체 흐름을 포함합니다.

---

## Acknowledgement
This project was supported by the Research Fund of the Sogang University LINC+ Program (University Innovation Support Project).

본 프로젝트는 AI+인문융합 LAB(탐구공동체) 프로그램의 결과물로, 서강대학교 대학혁신사업단(대학혁신지원사업)의 연구비 지원을 받아 수행되었습니다.

---

## Features
- Hugging Face Transformers 기반 A.X Light 로딩
- 대화형 프롬프트 구조로 소설 생성
- RAG(검색+생성) 또는 순수 생성 모드 지원
- 결과 자동 저장 및 로그 관리

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

---

## Installation
```bash
git clone https://github.com/hub2vu/ax-light-novel-pipeline.git
cd ax-light-novel-pipeline
pip install -r requirements.txt

```
---

## RUN
 1. ./rag_corpus/world_config.txt 에 배경, 분위기, 시대, 장르 등의 정보를 입력하세요.
 2. base_info_maker.py 실행
 3. Prompt_generater.py 실행
 4. storymake_and_detailer.py 실행
 5. Updater.py 실행

---

## Info
```bash
├── LORA_train_module			LORA학습 모듈
│   ├── A.X_QLORA_3.py			QLORA학습
│   └── continue_A.X_QLORA.py	QLORA점진학습		
	
├── Qlora
│   ├── ax-qlora-adapter

├── character_module			character 프로필생성기
│   └── A.X_generate_character_profiles.py

├── detailer_module				텍스트 디테일러
│   └── loop_detailer.py			3회 반복으로 신규 스토리 텍스트 수정

├── preprocess_module			전처리 코드[수정중]

├── prompt_module			프롬프트생성모듈
│   ├── prompt_1outline_name_extractor.py	다음화 등장인물 생성	
│   ├── prompt_2outline_generator.py		다음화 아웃라인 생성
│   ├── prompt_2.5outline_detailer_RAG.py		아웃라인에 기존 소설정보 반영
│   ├── prompt_3setting_maker.py			배경 생성. 기존 등장 장소 반영
│   └── prompt_4source_merger.py			프롬프트 생성

├── rag_corpus						생성 정보
│   ├── character.txt					캐릭터 프로필
│   ├── last_prompt.txt					마지막 프롬프트
│   ├── last_story.txt					지금까지의 전체 소설 생성본
│   ├── new_story.txt					1차 생성결과
│   ├── outline.txt						최신화 스토리 아웃라인
│   ├── outline_name_extract.txt			다음화 등장할 인물 및 등장이유
│   ├── relation.txt						인물 간 관계				
│   ├── relation_update.txt				신규 생성물을 관계에 반영
│   ├── relation_merged.txt				기존 관계에 신규 관계 반영
│   ├── setting.txt						최신화 시간, 배경, 분위기 정보
│   └── world_config.txt					세계관, 장르, 전체 분위기 등

├── relation_module					관계모듈
│   ├── A.X_realtion_maker_2parcing.py		관계생성	
│   ├── A.X_relation_updater_changemaker.py	최신화에 따른 관계 변동 감지 및 변화 생성
│   └── A.X_relation_updater_merger.py		기존 관계정보에 신규 관계변화 반영

├── reputation_module					평가모듈[진행중]
│   └── requtation_bert.py				

├── story_generater_module				소설생성기모듈
│   └── A.X_LORA_generater.py				LORA반영(코드 내 반영스위치 있음) 신규화 생성	

└── updater_module					업데이트 모듈
    └── story_stacker.py					신규 화, 기존 소설에 추가

// 모듈에서 py 불러와 파이프라인 구성
├── base_info_maker.py					캐릭터정보, 관계정보 생성 파이프라인
├── Prompt_generater.py					프롬프트 파이프라인
├── storymake_and_detailer.py				스토리 생성 및 디테일러 파이프라인
├── Updater.py						관계갱신 및 최신화 반영 파이프라인

// 텍스트
├── new_story_edited.txt					최신화

```



