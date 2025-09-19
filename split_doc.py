import json

with open('doc_ids.json', 'r') as f:
    doc_ids = json.load(f)
    
with open('echr_1268.json', 'r') as f:
    data = json.load(f)
    new_train = []
    new_test = []
    for item in data:
        if item['doc_id'] in doc_ids:
            new_test.append(item)
        else:
            new_train.append(item)
    with open('echr_train_1124.json', 'w', encoding='utf-8') as f:
        json.dump(new_train, f, ensure_ascii=False, indent=2)
    with open('echr_test_144.json', 'w', encoding='utf-8') as f:
        json.dump(new_test, f, ensure_ascii=False, indent=2)
    exit()