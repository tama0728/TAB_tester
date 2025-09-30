import json

predictions = []
with open('result.jsonl', "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        metadata = data['metadata']
        text = data['text']
        entities = data['entities']
        for entity in entities:
            if entity['span_text'] == "" or entity['end_offset'] == 0:
                entities.remove(entity)
        predictions.append( {"metadata": metadata, "text": text, "entities": entities} )


with open('result_no_null.jsonl', "w", encoding="utf-8") as f:
    for prediction in predictions:
        f.write(json.dumps(prediction, ensure_ascii=False) + '\n')
        
        
        