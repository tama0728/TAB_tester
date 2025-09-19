import jsonlines
import json

with open('tab_entity_data_144.jsonl', 'r') as f:
    doc_ids = []
    for line in f:
        data = json.loads(line)
        doc_ids.append(data['metadata']['provenance']['doc_id'])

with open('doc_ids.json', 'w') as f:
    json.dump(doc_ids, f)
        