import torch
import torch.nn as nn
import transformers
from longformer_experiments.longformer_model import Model
from longformer_experiments.data_handling import TrainingBatch, LabelSet
from transformers import LongformerTokenizerFast
import numpy as np
import argparse
import sys
import os
from byte_offset_utils import correct_offset_mapping, adjust_span_offsets, correct_offset_mapping_utf16, adjust_span_offsets_utf16, convert_utf16_offset_to_char_range
import json

# spaCy 및 Annotator 임포트
try:
    import spacy
    from scripts.annotate import Annotator
except ImportError:
    spacy = None
    Annotator = None

def parse_arguments():
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='Text anonymization inference using Longformer model')
    parser.add_argument('input_file', help='Path to input PANORAMA jsonl file')
    parser.add_argument('-o', '--output', help='Output file path (optional)', default=None)
    parser.add_argument('--format', choices=['json', 'txt'], default='json',
                       help='Output format (json for TAB format, txt for masked text)')
    return parser.parse_args()

def load_model_and_tokenizer():
    """모델과 토크나이저를 로드합니다."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert = "allenai/longformer-base-4096"

    # 토크나이저와 레이블 세트 초기화
    tokenizer = LongformerTokenizerFast.from_pretrained(bert)
    label_set = LabelSet(labels=["MASK"])

    # 모델 객체 생성
    model = Model(model=bert, num_labels=len(label_set.ids_to_label.values()))
    model = model.to(device)

    # 가중치 불러오기
    checkpoint = torch.load('longformer_experiments/long_model_922.pt', map_location=device)
    model.load_state_dict(checkpoint)

    # 평가 모드 설정
    model.eval()

    return model, tokenizer, device

def tokenize_and_predict(text, model, tokenizer, device):
    """텍스트를 토큰화하고 예측을 수행합니다."""
    # 텍스트 토큰화 (훈련과 동일한 방식)
    tokens = tokenizer(
        text,
        max_length=4096,
        truncation=True,
        padding=True,
        return_offsets_mapping=True,
        add_special_tokens=True
    )

    # 배치 형태로 변환
    batch = {
        "input_ids": torch.tensor([tokens["input_ids"]]).to(device),
        "attention_masks": torch.tensor([tokens["attention_mask"]]).to(device)
    }

    # 추론 실행
    with torch.no_grad():
        output = model(batch)
        # 출력 차원 조정 (훈련과 동일)
        output = output.permute(0, 2, 1)
        predictions = output.argmax(dim=1).cpu().numpy()[0]

    # 신뢰도 점수 계산
    confidence_scores = torch.softmax(output, dim=1).max(dim=1)[0].cpu().numpy()[0]

    return tokens, predictions, confidence_scores

def infer_entity_types(text, masked_spans, use_annotator=True):
    """마스킹된 구간의 실제 엔티티 타입을 추정합니다."""
    if not masked_spans:
        return masked_spans

    if use_annotator and Annotator is not None:
        # 프로젝트의 Annotator 사용 (권장)
        annotator = Annotator(spacy_model="en_core_web_md")
        # doc = annotator.annotate(text)
        # ent_spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        for span in masked_spans:
            span["entity_type"] = annotator.annotate2(span["span_text"])
            print("entity", span["span_text"], span["entity_type"])
        return masked_spans
    elif spacy is not None:
        # 기본 spaCy NER 사용
        nlp = spacy.load("en_core_web_md")
        doc = nlp(text)
        ent_spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    else:
        # 타입 추정 불가능한 경우 MASK로 유지
        return masked_spans

    print("ent_spans", ent_spans)
    # IoU 기반 매칭으로 엔티티 타입 할당
    for span in masked_spans:
        ss, se = span["start_offset"], span["end_offset"]
        best_label = "MISC"
        best_iou = 0.0

        for es, ee, lab in ent_spans:
            # 겹침 계산
            inter = max(0, min(se, ee) - max(ss, es))
            if inter == 0:
                continue
            union = (se - ss) + (ee - es) - inter
            iou = inter / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_label = lab if isinstance(lab, str) else str(lab)

        # IoU가 0.3 이상일 때만 타입 변경
        if best_iou >= 0.0:
            span["entity_type"] = best_label
        else:
            span["entity_type"] = "MISC"

    return masked_spans

def find_masked_spans_tab_format(predictions, offset_mapping, confidence_scores, text, doc_id="DOC1", use_utf16_code_units=False):
    """TAB 데이터셋 형식으로 마스킹 구간을 찾습니다."""
    # offset_mapping을 선택된 기준으로 수정
    if use_utf16_code_units:
        # Windows 메모장 기준 (UTF-16 코드 유닛)
        offset_mapping = correct_offset_mapping_utf16(text, offset_mapping)
    else:
        # UTF-8 바이트 기준 (기본값)
        offset_mapping = correct_offset_mapping(text, offset_mapping)
    
    masked_spans = []
    current_span = None
    entity_counter = 1
    mention_counter = 1

    for i, (pred, offset, conf) in enumerate(zip(predictions, offset_mapping, confidence_scores)):
        # 예측이 MASK이고 실제 텍스트에 해당하는 토큰인 경우
        # if pred > 0:
        #     print("i", i)
        #     print("pred", pred)
        #     print("offset", offset)
        #     print("conf", conf)
        #     print("text", text[offset[0]:offset[1]])
        # print("pred", pred, conf, text[offset[0]:offset[1]], i, offset)
        if pred > 0 and offset[0] is not None and offset[1] is not None:
            if current_span is None and conf >= 0.55:
                # 새로운 마스킹 구간 시작
                current_span = {
                    "entity_type": "MISC",
                    "entity_mention_id": f"{doc_id}_em{mention_counter}",
                    "start_offset": offset[0],
                    "end_offset": offset[1],
                    "span_text": "",
                    "edit_type": "insert",
                    "confidential_status": "NOT_CONFIDENTIAL",
                    "identifier_type": "DIRECT",
                    "entity_id": f"{doc_id}_e{entity_counter}",
                    "label": pred,
                    "confidence": float(conf),
                    "tokens": [i],
                    "start_token": i,
                    "end_token": i
                }
            elif current_span is not None:
                # 기존 구간 확장
                current_span["end_offset"] = offset[1]
                current_span["end_token"] = i
                current_span["confidence"] = max(current_span["confidence"], float(conf))
                current_span["tokens"].append(i)
        else:
            # 마스킹 구간 종료
            if current_span is not None:
                if conf >= 0.55 and predictions[i-1] > 0 and predictions[i+1 if i < len(predictions) - 1 else i] > 0:
                    print("extend", i)
                    # 기존 구간 확장
                    current_span["end_offset"] = offset[1]
                    current_span["end_token"] = i
                    current_span["confidence"] = max(current_span["confidence"], float(conf))
                    current_span["tokens"].append(i)
                else:
                    # UTF-16 코드 유닛 기준 offset을 문자 offset으로 변환하여 span_text 추출
                    if use_utf16_code_units:
                        char_start, char_end = convert_utf16_offset_to_char_range(
                            text, current_span["start_offset"], current_span["end_offset"]
                        )
                        current_span["span_text"] = text[char_start:char_end]
                    else:
                        current_span["span_text"] = text[current_span["start_offset"]:current_span["end_offset"]]
                    
                    spans = []
                    # | 분할
                    if "|" in current_span["span_text"]:
                        span1 = current_span.copy()
                        span2 = current_span.copy()
                        
                        span1["span_text"] = span1["span_text"].split("|")[0]
                        span1["end_offset"] = span1["start_offset"] + len(span1["span_text"])
                        spans.append(span1)
                        
                        span2["span_text"] = span2["span_text"].split("|")[1]
                        span2["start_offset"] = span2["end_offset"] - len(span2["span_text"])
                        spans.append(span2)
                        
                    else:
                        spans.append(current_span)
                    
                    for span in spans:
                        if span["entity_type"] == "CODE" and ":" in span["span_text"]:
                            span["span_text"] = span["span_text"].split(":")[1]
                            span["start_offset"] = span["end_offset"] - len(span["span_text"])
                        
                        # 마스킹 구간 종료 후 트림 및 구간 변경
                        tm_text = span["span_text"].lstrip()
                        tm_text = tm_text.lstrip("'@.,;:!?\"()[]{}<>")
                        span["start_offset"] = span["start_offset"] + (len(span["span_text"]) - len(tm_text))
                        span["span_text"] = tm_text
                        
                        tm_text = span["span_text"].rstrip()
                        tm_text = tm_text.rstrip("'@.,;:!?\"()[]{}<>")
                        span["end_offset"] = span["end_offset"] - (len(span["span_text"]) - len(tm_text))
                        span["span_text"] = tm_text
                        
                        masked_spans.append(span)
                        print("end", i, span["span_text"])
                    current_span = None
                    entity_counter += 1
                    mention_counter += 1


    # 마지막 구간 처리
    if current_span is not None:
        # UTF-16 코드 유닛 기준 offset을 문자 offset으로 변환하여 span_text 추출
        if use_utf16_code_units:
            char_start, char_end = convert_utf16_offset_to_char_range(
                text, current_span["start_offset"], current_span["end_offset"]
            )
            current_span["span_text"] = text[char_start:char_end]
        else:
            current_span["span_text"] = text[current_span["start_offset"]:current_span["end_offset"]]
        masked_spans.append(current_span)

    return masked_spans

def process_entities(text, predictions, offset_mapping, confidence_scores, doc_id="TEST_DOC", use_utf16_code_units=True):
    """엔티티 처리 전체 파이프라인을 실행합니다."""
    # 마스킹 구간 찾기
    masked_spans = find_masked_spans_tab_format(
        predictions, offset_mapping, confidence_scores, text, doc_id=doc_id, use_utf16_code_units=use_utf16_code_units
    )

    # 엔티티 타입 추정
    try:
        masked_spans = infer_entity_types(text, masked_spans, use_annotator=True)
        print("✅ Entity types inferred using spaCy/Annotator")
    except Exception as e:
        print(f"⚠️  spaCy/Annotator not available: {e}")
        print("Using default 'MASK' entity types")

    return masked_spans

def create_tab_format_json(json_data, masked_spans):
    """TAB 형식의 JSON 구조를 생성합니다."""
    entities = []
    for span in masked_spans:
        if span["span_text"] == "" or span["end_offset"] == 0:
            continue
        entities.append({
            "span_text": span["span_text"],
            "entity_type": span["entity_type"],
            "start_offset": span["start_offset"],
            "end_offset": span["end_offset"],
            "span_id": "",
            "entity_id": "",
            "annotator": "",
            "identifier_type": ""
        })

    return {
        "metadata": json_data['metadata'],
        "text": json_data['text'],
        "entities": entities
    }

def apply_masking(text, masked_spans):
    """텍스트에 마스킹을 적용합니다."""
    if not masked_spans:
        return text

    # 구간을 역순으로 정렬하여 뒤에서부터 마스킹
    sorted_spans = sorted(masked_spans, key=lambda x: x["start_offset"], reverse=True)
    masked_text = text
    for span in sorted_spans:
        start, end = span["start_offset"], span["end_offset"]
        masked_text = masked_text[:start] + f"[{span['entity_type']}]" + masked_text[end:]

    return masked_text

def print_results(text, masked_spans, tab_json, masked_text):
    """결과를 출력합니다."""
    print(f"\nFound {len(masked_spans)} masked spans (TAB format):")
    # for i, span in enumerate(masked_spans):
    #     print(f"Entity {i+1}:")
    #     print(f"  - entity_type: {span['entity_type']}")
    #     print(f"  - span_text: '{span['span_text']}'")
    #     print(f"  - start_offset: {span['start_offset']}")
    #     print(f"  - end_offset: {span['end_offset']}")
    #     print(f"  - confidence: {span['confidence']:.3f}")
    #     print()

    print("TAB format JSON structure:")
    print(f"Document ID: {tab_json[0]['doc_id']}")
    print(f"Number of entity mentions: {len(tab_json[0]['annotations']['annotator_1']['entity_mentions'])}")

    print("\nText masking results:")
    print("Original text preview:", text[:200] + "..." if len(text) > 200 else text)
    print("Masked text preview:", masked_text[:200] + "..." if len(masked_text) > 200 else masked_text)

def save_results(args, tab_json_list):
    """결과를 파일로 저장합니다."""
    # 출력 파일 경로 결정
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(args.input_file)[0]
        output_file = f"{base_name}_predictions.jsonl"

    # 결과 저장
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tab_json_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"\nTAB format results saved to '{output_file}'")
        print(f"Processing completed successfully!")

    except Exception as e:
        print(f"Error saving output file '{output_file}': {e}")
        sys.exit(1)

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    args = parse_arguments()

    # 입력 파일 존재 확인
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        sys.exit(1)

    json_list = []
    # 텍스트 파일 읽기
    try:
        if args.input_file.endswith('.jsonl'):
            with open(args.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    json_list.append(json.loads(line))
        else:
            # 오류처리 
            print(f"Error: Input file '{args.input_file}' is not a json file!")
            sys.exit(1)
        print(f"Loaded text from: {args.input_file}")
        print(f"Text length: {len(json_list)} documents")
    except Exception as e:
        print(f"Error reading file '{args.input_file}': {e}")
        sys.exit(1)

    # 모델과 토크나이저 로드
    print("Loading model and tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer()
    print(f"Model loaded on device: {device}")

    tab_json_list = []
    masked_text_list = []
    masked_spans_list = []
    for json_data in json_list:
        meta = json_data['metadata']
        # 토큰화 및 예측
        print("Tokenizing text and running inference...")
        tokens, predictions, confidence_scores = tokenize_and_predict(json_data['text'], model, tokenizer, device)

        # 엔티티 처리
        print("Processing entities...")
        masked_spans = process_entities(json_data['text'], predictions, tokens["offset_mapping"], confidence_scores)


        # 마스킹된 텍스트 생성
        masked_text = apply_masking(json_data['text'], masked_spans)

        tab_json_list.append(create_tab_format_json(json_data, masked_spans))
        masked_text_list.append(masked_text)
        masked_spans_list.append(masked_spans)


    # 결과 저장
    save_results(args, tab_json_list)

if __name__ == "__main__":
    main()