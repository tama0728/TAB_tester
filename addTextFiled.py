import json

texts = {}

with open('panorama_798_annotated_anthropic_claude-sonnet-4-20250514.jsonl', "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        texts[data['metadata']['data_id']] = data['text']

predictions = []
with open('panorama_798_annotated_anthropic_claude-sonnet-4-20250514_predictions.jsonl', "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        metadata = data['metadata']
        entities = data['entities']
        predictions.append( {"metadata": metadata, "text": texts[metadata['data_id']], "entities": entities} )


with open('result.jsonl', "w", encoding="utf-8") as f:
    for prediction in predictions:
        f.write(json.dumps(prediction, ensure_ascii=False) + '\n')
        
        
        