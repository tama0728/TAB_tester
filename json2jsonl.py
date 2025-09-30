import json

# argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', type=str, required=True)
parser.add_argument('--output_file', '-o', type=str, required=True)
args = parser.parse_args()

with open(args.input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 데이터가 리스트인지 확인
if isinstance(data, list):
    items = data
elif isinstance(data, dict):
    # 딕셔너리인 경우 values() 사용하거나 직접 처리
    if 'documents' in data:
        items = data['documents']
    elif 'data' in data:
        items = data['data']
    else:
        # 딕셔너리 자체를 리스트로 감싸기
        items = [data]
else:
    print(f"Error: Unsupported data type: {type(data)}")
    exit(1)

with open(args.output_file, 'w', encoding='utf-8') as f:
    for item in items:
        # 각 항목이 딕셔너리인지 확인
        if isinstance(item, dict):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            print(f"Warning: Skipping non-dict item: {type(item)}")
            # 딕셔너리가 아닌 경우에도 JSON으로 직렬화 시도
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Converted {len(items)} items to JSONL format")
        