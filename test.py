import torch
import torch.nn as nn
import transformers
from longformer_experiments.longformer_model import Model
from longformer_experiments.data_handling import TrainingBatch, LabelSet
from transformers import LongformerTokenizerFast
import numpy as np

# spaCy 및 Annotator 임포트
try:
    import spacy
    from scripts.annotate import Annotator
except ImportError:
    spacy = None
    Annotator = None

# load text from data.txt
with open('data.txt', 'r') as f:
    text = f.read()

# 1. 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bert = "allenai/longformer-base-4096"

# 2. 토크나이저와 레이블 세트 초기화
tokenizer = LongformerTokenizerFast.from_pretrained(bert)
label_set = LabelSet(labels=["MASK"])

# 3. 모델 객체 생성
model = Model(model=bert, num_labels=len(label_set.ids_to_label.values()))
model = model.to(device)

# 4. 가중치 불러오기
checkpoint = torch.load('longformer_experiments/long_model.pt', map_location=device)
model.load_state_dict(checkpoint)

# 5. 평가 모드 설정
model.eval()

# 6. 텍스트 토큰화 (훈련과 동일한 방식)
tokens = tokenizer(
    text,
    max_length=4096,
    truncation=True,
    padding=True,
    return_offsets_mapping=True,
    add_special_tokens=True
)

# 7. 배치 형태로 변환
batch = {
    "input_ids": torch.tensor([tokens["input_ids"]]).to(device),
    "attention_masks": torch.tensor([tokens["attention_mask"]]).to(device)
}

# 8. 추론 실행
with torch.no_grad():
    output = model(batch)
    # 출력 차원 조정 (훈련과 동일)
    output = output.permute(0, 2, 1)
    predictions = output.argmax(dim=1).cpu().numpy()[0]

print("Predictions shape:", predictions.shape)
print("Predictions:", predictions)

# 9. 엔티티 타입 추정 함수
def infer_entity_types(text, masked_spans, use_annotator=True):
    """마스킹된 구간의 실제 엔티티 타입을 추정합니다."""
    if not masked_spans:
        return masked_spans
    
    if use_annotator and Annotator is not None:
        # 프로젝트의 Annotator 사용 (권장)
        annotator = Annotator(spacy_model="en_core_web_md")
        doc = annotator.annotate(text)
        ent_spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    elif spacy is not None:
        # 기본 spaCy NER 사용
        nlp = spacy.load("en_core_web_md")
        doc = nlp(text)
        ent_spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    else:
        # 타입 추정 불가능한 경우 MASK로 유지
        return masked_spans
    
    # IoU 기반 매칭으로 엔티티 타입 할당
    for span in masked_spans:
        ss, se = span["start_offset"], span["end_offset"]
        best_label = "MASK"  # 기본값
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
        
        # IoU가 0.3 이상일 때만 타입 변경 (너무 낮으면 MASK 유지)
        if best_iou >= 0.3:
            span["entity_type"] = best_label
        else:
            span["entity_type"] = "MASK"
    
    return masked_spans

def link_entities_by_text_similarity(masked_spans):
    """텍스트 유사도 기반으로 같은 엔티티를 연결합니다."""
    if not masked_spans:
        return masked_spans
    
    # 엔티티 타입별로 그룹화
    entity_groups = {}
    for span in masked_spans:
        entity_type = span["entity_type"]
        if entity_type not in entity_groups:
            entity_groups[entity_type] = []
        entity_groups[entity_type].append(span)
    
    # 각 타입별로 엔티티 연결
    entity_counter = 1
    for entity_type, spans in entity_groups.items():
        if entity_type == "MASK":
            # MASK 타입은 각각 별도 엔티티로 처리
            for span in spans:
                span["entity_id"] = f"TEST_DOC_e{entity_counter}"
                entity_counter += 1
        else:
            # 같은 타입의 엔티티들을 텍스트 유사도로 연결
            linked_groups = link_spans_by_similarity(spans)
            for group in linked_groups:
                group_entity_id = f"TEST_DOC_e{entity_counter}"
                for span in group:
                    span["entity_id"] = group_entity_id
                entity_counter += 1
    
    return masked_spans

def link_spans_by_similarity(spans):
    """텍스트 유사도 기반으로 스팬들을 그룹화합니다."""
    if len(spans) <= 1:
        return [spans]
    
    groups = []
    used = set()
    
    for i, span1 in enumerate(spans):
        if i in used:
            continue
            
        current_group = [span1]
        used.add(i)
        
        for j, span2 in enumerate(spans[i+1:], i+1):
            if j in used:
                continue
                
            # 텍스트 유사도 계산 (간단한 방법)
            similarity = calculate_text_similarity(span1["span_text"], span2["span_text"])
            
            if similarity > 0.7:  # 70% 이상 유사하면 같은 엔티티로 간주
                current_group.append(span2)
                used.add(j)
        
        groups.append(current_group)
    
    return groups

def calculate_text_similarity(text1, text2):
    """두 텍스트의 유사도를 계산합니다 (간단한 구현)."""
    text1_lower = text1.lower().strip()
    text2_lower = text2.lower().strip()
    
    # 완전 일치
    if text1_lower == text2_lower:
        return 1.0
    
    # 부분 문자열 포함
    if text1_lower in text2_lower or text2_lower in text1_lower:
        return 0.8
    
    # 단어 기반 유사도
    words1 = set(text1_lower.split())
    words2 = set(text2_lower.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

# 10. 마스킹 구간 찾기 (TAB 데이터셋 형식)
def find_masked_spans_tab_format(predictions, offset_mapping, confidence_scores, doc_id="DOC1"):
    """TAB 데이터셋 형식으로 마스킹 구간을 찾습니다."""
    masked_spans = []
    current_span = None
    entity_counter = 1
    mention_counter = 1
    
    for i, (pred, offset, conf) in enumerate(zip(predictions, offset_mapping, confidence_scores)):
        # 예측이 MASK이고 실제 텍스트에 해당하는 토큰인 경우
        if pred > 0 and offset[0] is not None and offset[1] is not None:
            if current_span is None:
                # 새로운 마스킹 구간 시작 (B-MASK)
                current_span = {
                    "entity_type": "MASK",  # 나중에 실제 타입으로 변경
                    "entity_mention_id": f"{doc_id}_em{mention_counter}",
                    "start_offset": offset[0],
                    "end_offset": offset[1],
                    "span_text": "",  # 나중에 채움
                    "edit_type": "insert",  # 모델이 새로 예측한 엔티티 (check/insert/correct 중 insert가 적절)
                    "confidential_status": "NOT_CONFIDENTIAL",
                    "identifier_type": "DIRECT",  # 모델이 찾은 것은 모두 DIRECT로 간주
                    "entity_id": f"{doc_id}_e{entity_counter}",
                    "label": pred,
                    "confidence": float(conf),
                    "tokens": [i],
                    "start_token": i,
                    "end_token": i
                }
            else:
                # 기존 구간 확장 (I-MASK)
                current_span["end_offset"] = offset[1]
                current_span["end_token"] = i
                current_span["confidence"] = max(current_span["confidence"], float(conf))
                current_span["tokens"].append(i)
        else:
            # 마스킹 구간 종료
            if current_span is not None:
                # span_text 채우기
                current_span["span_text"] = text[current_span["start_offset"]:current_span["end_offset"]]
                masked_spans.append(current_span)
                current_span = None
                entity_counter += 1
                mention_counter += 1
    
    # 마지막 구간 처리
    if current_span is not None:
        current_span["span_text"] = text[current_span["start_offset"]:current_span["end_offset"]]
        masked_spans.append(current_span)
    
    return masked_spans

# 신뢰도 점수 계산
confidence_scores = torch.softmax(output, dim=1).max(dim=1)[0].cpu().numpy()[0]

# 상세한 마스킹 구간 찾기 (TAB 형식)
masked_spans = find_masked_spans_tab_format(
    predictions, 
    tokens["offset_mapping"], 
    confidence_scores,
    doc_id="TEST_DOC"
)

# 엔티티 타입 추정 및 연결 (spaCy/Annotator 사용)
try:
    masked_spans = infer_entity_types(text, masked_spans, use_annotator=True)
    print("✅ Entity types inferred using spaCy/Annotator")
except Exception as e:
    print(f"⚠️  spaCy/Annotator not available: {e}")
    print("Using default 'MASK' entity types")


print(f"\nFound {len(masked_spans)} masked spans (TAB format):")
for i, span in enumerate(masked_spans):
    print(f"Entity {i+1}:")
    print(f"  - entity_type: {span['entity_type']}")
    print(f"  - entity_mention_id: {span['entity_mention_id']}")
    print(f"  - entity_id: {span['entity_id']}")
    print(f"  - start_offset: {span['start_offset']}")
    print(f"  - end_offset: {span['end_offset']}")
    print(f"  - span_text: '{span['span_text']}'")
    print(f"  - identifier_type: {span['identifier_type']}")
    print(f"  - confidential_status: {span['confidential_status']}")
    print(f"  - edit_type: {span['edit_type']}")
    print(f"  - confidence: {span['confidence']:.3f}")
    print(f"  - token range: {span['start_token']}-{span['end_token']}")
    print()

# 10. TAB 형식 JSON 출력 생성
def create_tab_format_json(doc_id, text, masked_spans, dataset_type="dev"):
    """TAB 형식의 JSON 구조를 생성합니다."""
    entity_mentions = []
    for span in masked_spans:
        entity_mentions.append({
            "entity_type": span["entity_type"],
            "entity_mention_id": span["entity_mention_id"],
            "start_offset": span["start_offset"],
            "end_offset": span["end_offset"],
            "span_text": span["span_text"],
            "edit_type": span["edit_type"],
            "confidential_status": span["confidential_status"],
            "identifier_type": span["identifier_type"],
            "entity_id": span["entity_id"]
        })
    
    return [{
        "doc_id": doc_id,
        "text": text,
        "dataset_type": dataset_type,
        "meta": {},
        "quality_checked": False,
        "task": None,
        "annotations": {
            "annotator_1": {
                "entity_mentions": entity_mentions
            }
        }
    }]

# TAB 형식 JSON 생성
tab_json = create_tab_format_json("TEST_DOC", text, masked_spans)
print("TAB format JSON structure:")
print(f"Document ID: {tab_json[0]['doc_id']}")
print(f"Dataset type: {tab_json[0]['dataset_type']}")
print(f"Number of entity mentions: {len(tab_json[0]['annotations']['annotator_1']['entity_mentions'])}")
print(f"First entity mention: {tab_json[0]['annotations']['annotator_1']['entity_mentions'][0] if tab_json[0]['annotations']['annotator_1']['entity_mentions'] else 'None'}")

# 11. 마스킹된 텍스트 생성
def apply_masking(text, masked_spans, mask_char="[MASK]"):
    """텍스트에 마스킹을 적용합니다."""
    if not masked_spans:
        return text
    
    # 구간을 역순으로 정렬하여 뒤에서부터 마스킹 (인덱스 변화 방지)
    sorted_spans = sorted(masked_spans, key=lambda x: x["start_offset"], reverse=True)
    
    masked_text = text
    for span in sorted_spans:
        start, end = span["start_offset"], span["end_offset"]
        masked_text = masked_text[:start] + mask_char + masked_text[end:]
    
    return masked_text

masked_text = apply_masking(text, masked_spans)
print("\nText masking results:")
print("Original text preview:", text[:200] + "..." if len(text) > 200 else text)
print("Masked text preview:", masked_text[:200] + "..." if len(masked_text) > 200 else masked_text)

# 12. JSON 파일로 저장 (선택사항)
import json
with open('test_predictions.json', 'w', encoding='utf-8') as f:
    json.dump(tab_json, f, ensure_ascii=False, indent=2)
print(f"\nTAB format results saved to 'test_predictions.json'")