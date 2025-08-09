import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import itertools  # nC2 조합 생성을 위해 추가


# ================== 이름 리스트 추출 ==================
def extract_names_from_outline(input_filename):
    """
    주어진 파일에서 '숫자. 이름' 패턴을 찾아 이름 리스트를 반환합니다.
    """
    extracted_names = []
    pattern = re.compile(r"^\s*\d+\.\s+(.+)")
    try:
        with open(input_filename, 'r', encoding='utf-8') as infile:
            for line in infile:
                match = pattern.match(line)
                if match:
                    name = match.group(1).strip()
                    extracted_names.append(name)
        return extracted_names
    except FileNotFoundError:
        print(f"오류: '{input_filename}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return []
    except Exception as e:
        print(f"이름 추출 중 오류 발생: {e}")
        return []


# ================== 이전 아웃라인 파싱 ==================
def parse_past_outline(file_path):
    """
    지정된 파일의 가장 마지막 줄을 파싱해서 가져옵니다.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            if lines:
                return lines[-1]
            else:
                return "창의적으로 스토리 아웃라인을 생성하라."
    except FileNotFoundError:
        return "창의적으로 스토리 아웃라인을 생성하라."
    except Exception as e:
        return f"파싱 중 오류 발생: {e}"


# ================== 전체 캐릭터 정보 파싱 ==================
def parse_all_characters(file_path):
    """
    character.txt 파일에서 모든 등장인물 정보를 파싱하여 딕셔너리로 반환합니다.
    """
    characters = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            character_blocks = content.split('--')
            for block in character_blocks:
                block = block.strip()
                if block:
                    name_match = re.search(r'^\d+\.\s*(.*)', block, re.MULTILINE)
                    if name_match:
                        name = name_match.group(1).strip()
                        characters[name] = block
    except FileNotFoundError:
        print(f"오류: {file_path} 파일을 찾을 수 없습니다.")
    return characters


# ================== 인물 관계 정보 파싱 (새로 추가된 함수) ==================
def parse_relations(file_path):
    """
    relation.txt 파일에서 인물 간의 관계 정보를 파싱합니다.
    A - B 순서는 이름의 가나다순으로 정렬하여 키로 사용합니다.
    """
    relations = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # "이름1 - 이름2: 관계설명" 형식의 라인을 파싱
                match = re.match(r'^(.*?) - (.*?):(.*)', line)
                if match:
                    name1 = match.group(1).strip()
                    name2 = match.group(2).strip()
                    description = match.group(3).strip()
                    # 이름을 정렬하여 일관된 키를 생성
                    sorted_names = tuple(sorted([name1, name2]))
                    relations[sorted_names] = description
    except FileNotFoundError:
        print(f"경고: 관계 파일 '{file_path}'을(를) 찾을 수 없습니다. 관계 정보 없이 진행합니다.")
    return relations


# ================== 다음 아웃라인 생성 (프롬프트 수정) ==================
def generate_next_outline(past_outline, selected_characters_info, relations_info, model, tokenizer):
    """
    skt/A.X-4.0-Light 모델을 사용하여 다음 스토리 아웃라인을 생성합니다.
    인물 관계 정보가 프롬프트에 추가되었습니다.
    """
    if not selected_characters_info:
        raise ValueError("다음 아웃라인을 생성할 캐릭터 정보가 없습니다.")

    # 모델에 전달할 프롬프트를 구성합니다.
    prompt_template = """<s>[PROMPT]
[SYSTEM]
너는 이전 스토리, 주어진 등장인물, 그리고 그들의 관계를 바탕으로 다음 스토리를 흥미진진하게 이어가는 아웃라인을 생성하는 AI 작가야.

[이전 스토리 아웃라인]
{past_outline}

[등장인물 정보]
{selected_characters_info}

[인물 관계 정보]
{relations_info}

[요청]
위 모든 정보를 종합적으로 고려하여, 다음 사건이 벌어질 새로운 아웃라인을 생성해줘.
특히 인물들의 관계가 스토리에 어떻게 영향을 미치는지 잘 드러나야 해.
새로운 아웃라인은 독자의 궁금증을 유발하고, 인물 간의 관계가 더 깊어지거나 새로운 갈등이 생겨나는 방향으로 전개되어야 해.
결과는 특수문자나 줄바꿈 없이, 하나의 연속된 문장으로 만들어줘.
[/PROMPT]
[STORY]
"""
    prompt = prompt_template.format(
        past_outline=past_outline,
        selected_characters_info=selected_characters_info,
        relations_info=relations_info if relations_info else "제공된 관계 정보 없음."  # 관계 정보가 없을 경우를 대비
    )

    # 토큰화 및 디바이스 할당
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)

    # 모델을 사용하여 텍스트 생성
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
    )

    # 결과 디코딩
    generated_text = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

    # 생성된 텍스트 후처리
    cleaned_text = re.sub(r'[\*#\-:]', '', generated_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text


# ================== 메인 실행 함수 (로직 수정) ==================
def main():
    """
    메인 실행 함수
    """
    # Hugging Face 모델 및 토크나이저 로드
    model_name = "skt/A.X-4.0-Light"
    print(f"Loading model: {model_name}...")
    print("GPU 사용을 권장합니다. 로딩에 몇 분 정도 소요될 수 있습니다.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            model.resize_token_embeddings(len(tokenizer))
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        print("transformers, torch, accelerate 라이브러리가 올바르게 설치되었는지 확인해주세요.")
        return

    # 파일 경로 설정
    outline_source_file = '../rag_corpus/outline.txt'
    character_file = '../rag_corpus/character.txt'
    name_extract_file = '../rag_corpus/outline_name_extract.txt'
    relation_file = '../rag_corpus/relation.txt'  # 관계 파일 경로 추가

    # 1. outline_name_extract.txt에서 필요한 인물 이름 목록 추출
    required_names = extract_names_from_outline(name_extract_file)
    if not required_names:
        print(f"'{name_extract_file}'에서 인물 이름을 추출하지 못했습니다. 스크립트를 종료합니다.")
        return

    print("\n--- 다음 스토리에 등장할 인물 ---")
    for name in required_names:
        print(f"- {name}")

    # 2. character.txt에서 모든 인물 정보 로드
    all_characters = parse_all_characters(character_file)
    if not all_characters:
        print(f"'{character_file}'에서 인물 정보를 가져오지 못했습니다. 스크립트를 종료합니다.")
        return

    # 3. 필요한 인물들의 정보만 필터링
    selected_character_details = []
    missing_characters = []
    for name in required_names:
        if name in all_characters:
            selected_character_details.append(all_characters[name])
        else:
            missing_characters.append(name)

    if missing_characters:
        print(f"\n경고: '{character_file}' 파일에 다음 인물 정보가 없습니다: {', '.join(missing_characters)}")

    if not selected_character_details:
        print("\n오류: 다음 스토리를 생성하는 데 필요한 인물 정보를 찾지 못했습니다. 스크립트를 종료합니다.")
        return

    selected_characters_info = "\n--\n".join(selected_character_details)

    # 4. 관계 정보 파싱 및 추출 (새로 추가된 로직)
    all_relations = parse_relations(relation_file)
    relevant_relations_info = []

    # 등장인물 리스트에서 2명씩 짝을 지음 (nC2)
    if len(required_names) >= 2:
        for name_pair in itertools.combinations(required_names, 2):
            # 튜플을 정렬하여 일관된 키 생성
            sorted_pair = tuple(sorted(name_pair))
            if sorted_pair in all_relations:
                # 관계 정보가 존재하면, "이름1 - 이름2: 관계설명" 형식으로 추가
                relation_text = f"{sorted_pair[0]} - {sorted_pair[1]}: {all_relations[sorted_pair]}"
                relevant_relations_info.append(relation_text)

    relations_info_str = "\n".join(relevant_relations_info)
    print("\n--- 추출된 인물 관계 정보 ---")
    print(relations_info_str if relations_info_str else "추출된 관계 정보가 없습니다.")

    # 5. 이전 아웃라인 파싱
    past_outline = parse_past_outline(outline_source_file)

    if not past_outline.startswith("파싱 중"):
        print("\n--- 이전 스토리 아웃라인 ---")
        print(past_outline)
        print("\n" + "=" * 30 + "\n")

        # 6. 다음 스토리 아웃라인 생성
        print("Generating next outline...")
        try:
            next_outline = generate_next_outline(past_outline, selected_characters_info, relations_info_str, model,
                                                 tokenizer)
        except ValueError as e:
            print(f"오류: {e}")
            return

        print("--- 새롭게 생성된 다음 스토리 아웃라인 ---")
        print(next_outline)

        # 7. 생성된 아웃라인을 outline.txt 파일에 추가
        output_file_path = '../rag_corpus/outline.txt'
        try:
            output_dir = os.path.dirname(output_file_path)
            os.makedirs(output_dir, exist_ok=True)

            with open(output_file_path, 'a', encoding='utf-8') as f:
                f.seek(0, 2)
                if f.tell() > 0:
                    f.write('\n')
                f.write(next_outline)

            print(f"\n결과가 '{output_file_path}' 파일에 성공적으로 추가되었습니다.")
        except Exception as e:
            print(f"\n파일에 쓰는 중 오류가 발생했습니다: {e}")
    else:
        print(f"오류: 스토리 아웃라인을 생성하는 데 필요한 정보를 가져오지 못했습니다. (past_outline: {past_outline})")


if __name__ == "__main__":
    main()
