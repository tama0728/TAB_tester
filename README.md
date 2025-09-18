# Text Anonymization with Trained Longformer Model

> **Enhanced version** of the [Text Anonymization Benchmark (TAB)](https://github.com/NorskRegnesentral/text-anonymisation-benchmark) with ready-to-use inference capabilities.

This repository provides a **trained Longformer model** and **complete inference pipeline** for text anonymization tasks. Simply load your text and get anonymized results with detailed entity information in TAB format.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/tama0728/TAB_tester.git
cd TAB_tester

# Create conda environment (includes spaCy model)
conda env create -f environment.yml
conda activate tab
```

### 2. Download Model File

```bash
# Download the trained model (568MB)
python download_data.py
```

### 3. Run Text Anonymization

```bash
# Test with sample text
python test.py

# Or modify data.txt with your own text
echo "Your text here" > data.txt
python test.py
```

**That's it!** The script will:
- ✅ Load the trained Longformer model
- ✅ Detect personal identifiers in your text
- ✅ Classify entities (PERSON, ORG, LOC, etc.)
- ✅ Link co-references
- ✅ Generate TAB-format JSON output
- ✅ Create masked text with `[MASK]` tokens

## 📊 What You Get

### Console Output
```
Found 3 masked spans (TAB format):
Entity 1:
  - entity_type: PERSON
  - entity_mention_id: TEST_DOC_em1
  - entity_id: TEST_DOC_e1
  - start_offset: 0
  - end_offset: 10
  - span_text: 'John Smith'
  - identifier_type: DIRECT
  - confidential_status: NOT_CONFIDENTIAL
  - edit_type: insert
  - confidence: 0.892
```

### JSON Output (`test_predictions.json`)
```json
{
  "doc_id": "TEST_DOC",
  "text": "John Smith works at Microsoft Corporation...",
  "annotations": {
    "annotator_1": {
      "entity_mentions": [
        {
          "entity_type": "PERSON",
          "entity_mention_id": "TEST_DOC_em1",
          "start_offset": 0,
          "end_offset": 10,
          "span_text": "John Smith",
          "edit_type": "insert",
          "confidential_status": "NOT_CONFIDENTIAL",
          "identifier_type": "DIRECT",
          "entity_id": "TEST_DOC_e1"
        }
      ]
    }
  }
}
```

### Masked Text
```
Original: "John Smith works at Microsoft Corporation in Seattle."
Masked:   "[MASK] works at [MASK] in [MASK]."
```

## 🔧 Advanced Usage

### Custom Text Processing

```python
# Modify test.py to process multiple texts
texts = [
    "Alice Johnson is a lawyer in New York.",
    "The company Microsoft was founded by Bill Gates."
]

for i, text in enumerate(texts):
    # Save text to data.txt
    with open('data.txt', 'w') as f:
        f.write(text)
    
    # Run inference
    import subprocess
    subprocess.run(['python', 'test.py'])
    
    # Load results
    import json
    with open('test_predictions.json', 'r') as f:
        results = json.load(f)
    
    print(f"Text {i+1}: Found {len(results[0]['annotations']['annotator_1']['entity_mentions'])} entities")
```

### Model Configuration

The model uses these default settings:
- **Model**: `allenai/longformer-base-4096`
- **Max Length**: 4096 tokens
- **Labels**: MASK detection (B-MASK, I-MASK, O)
- **Device**: Auto-detects CUDA/CPU

To modify settings, edit `test.py`:
```python
# Change model
bert = "allenai/longformer-base-4096"  # or other Longformer model

# Adjust max length
tokens = tokenizer(
    text,
    max_length=2048,  # Reduce for faster processing
    truncation=True,
    padding=True,
    return_offsets_mapping=True,
    add_special_tokens=True
)
```

## 📁 Project Structure

```
text-anonymization-benchmark/
├── test.py                      # 🎯 Main inference script
├── environment.yml              # 📦 Complete conda environment
├── data.txt                     # 📝 Input text file
├── test_predictions.json        # 📊 Output results
├── longformer_experiments/
│   ├── long_model.pt           # 🧠 Trained model weights
│   ├── train_model.py          # 🏋️ Training script
│   ├── longformer_model.py     # 🏗️ Model architecture
│   └── data_handling.py        # 🔧 Data utilities
├── scripts/
│   └── annotate.py             # 🏷️ spaCy annotation utilities
├── echr_train.json             # 📚 Training dataset
├── echr_test.json              # 📚 Test dataset
└── README.md                   # 📖 This file
```

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: Longformer-base-4096 (4096 token context)
- **Task**: Token-level classification for anonymization
- **Labels**: 3 classes (O, B-MASK, I-MASK)
- **Training**: Fine-tuned on ECHR court cases

### Entity Processing Pipeline
1. **Tokenization**: Longformer tokenizer with offset mapping
2. **Inference**: Trained model predicts MASK labels
3. **Span Extraction**: Convert token predictions to character spans
4. **Entity Classification**: Use spaCy/Annotator for entity types
5. **Entity Linking**: Co-reference resolution by text similarity
6. **Output Generation**: TAB-format JSON with all metadata

### Dependencies
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face Longformer
- **spaCy**: Named Entity Recognition
- **NumPy/Pandas**: Data processing
- **scikit-learn**: ML utilities

All dependencies are automatically installed via `environment.yml`.

## 📈 Performance

The trained model provides:
- **Entity Detection**: Identifies personal identifiers
- **Entity Classification**: PERSON, ORG, LOC, MISC, etc.
- **Confidence Scores**: Reliability metrics for each prediction
- **Co-reference Resolution**: Links mentions of the same entity

## 🔍 Troubleshooting

### Common Issues

**1. CUDA out of memory**
```python
# In test.py, change device to CPU
device = 'cpu'  # instead of 'cuda'
```

**2. spaCy model not found**
```bash
# Already included in environment.yml, but if needed:
python -m spacy download en_core_web_md
```

**3. Model file missing**
```bash
# Ensure long_model.pt exists
ls longformer_experiments/long_model.pt
```

**4. Empty predictions**
- Check if your text contains personal identifiers
- Try with sample text: "John Smith works at Microsoft Corporation"

### Performance Tips

- **Faster processing**: Reduce `max_length` to 2048 or 1024
- **Better accuracy**: Use longer texts (more context)
- **Memory issues**: Process texts in smaller chunks

## 📚 Dataset Information

Based on the **Text Anonymization Benchmark (TAB)**:
- **Source**: European Court of Human Rights (ECHR) cases
- **Size**: 1,268 manually annotated documents
- **Format**: Standoff JSON with entity mentions
- **Categories**: PERSON, ORG, LOC, MISC, DATETIME, QUANTITY, CODE

## 🔗 References

- **Original Repository**: [NorskRegnesentral/text-anonymisation-benchmark](https://github.com/NorskRegnesentral/text-anonymisation-benchmark)
- **Original Paper**: [The Text Anonymization Benchmark (TAB)](https://arxiv.org/abs/2202.00443)
- **Longformer Paper**: [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)

## 📄 License

MIT License - see original repository for details.

## 🙏 Acknowledgments

- **Original Authors**: Ildikó Pilán, Pierre Lison, Lilja Øvrelid, Anthi Papadopoulou, David Sánchez, Montserrat Batet
- **Enhanced Version**: Ready-to-use inference pipeline and improved usability

---

**Ready to anonymize text?** Just run `python test.py` and you're done! 🎉