import json

with open('doc_ids.json', 'r') as f:
    doc_ids = json.load(f)
    
with open('echr_1268.json', 'r') as f:
    data = json.load(f)
    new_train = []
    new_test = []
    new_dev = []
    for item in data:
        if item['doc_id'] not in doc_ids:
            if item['dataset_type'] == 'train':
                new_train.append(item)
            elif item['dataset_type'] == 'test':
                new_test.append(item)
            elif item['dataset_type'] == 'dev':
                new_dev.append(item)
        else:
            if item['dataset_type'] == 'dev':
                new_dev.append(item)
            else:
                new_test.append(item)
    with open('echr_train_1001.json', 'w', encoding='utf-8') as f:
        json.dump(new_train, f, ensure_ascii=False, indent=2)
    with open('echr_test_1001.json', 'w', encoding='utf-8') as f:
        json.dump(new_test, f, ensure_ascii=False, indent=2)
    with open('echr_dev_1001.json', 'w', encoding='utf-8') as f:
        json.dump(new_dev, f, ensure_ascii=False, indent=2)
    exit()