# Text Anonymization with Trained Longformer Model

> **Enhanced version** of the [Text Anonymization Benchmark (TAB)](https://github.com/NorskRegnesentral/text-anonymisation-benchmark) with ready-to-use inference capabilities.

This repository provides a **trained Longformer model** and **complete inference pipeline** for text anonymization tasks. Simply provide your text file and get anonymized results with detailed entity information in TAB format.

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
# Create a text file
echo "John Smith works at Microsoft Corporation in Seattle." > sample.txt

# Run anonymization with file path
python test.py sample.txt

# Specify output format
python test.py sample.txt --format json  # TAB format JSON (default)
python test.py sample.txt --format txt   # Masked text only

# Custom output file
python test.py sample.txt -o my_results.json
```

**That's it!** The script will:
- ✅ Load the trained Longformer model
- ✅ Detect personal identifiers in your text
- ✅ Classify entities (PERSON, ORG, LOC, etc.)
- ✅ Generate TAB-format JSON output
- ✅ Create masked text with entity-specific markers

## 📊 What You Get

### Console Output
```
Loaded text from: sample.txt
Text length: 53 characters
Loading model and tokenizer...
Model loaded on device: cuda

Found 3 masked spans (TAB format):
Document ID: sample
Number of entity mentions: 3

Text masking results:
Original text: John Smith works at Microsoft Corporation in Seattle.
Masked text: [PERSON] works at [ORG] in [GPE].
```

### JSON Output (`sample_predictions.json`)
```json
[{
  "doc_id": "sample",
  "text": "John Smith works at Microsoft Corporation in Seattle.",
  "masked_text": "[PERSON] works at [ORG] in [GPE].",
  "dataset_type": "dev",
  "meta": {},
  "quality_checked": false,
  "task": null,
  "annotations": {
    "annotator_1": {
      "entity_mentions": [
        {
          "entity_type": "PERSON",
          "entity_mention_id": "sample_em1",
          "start_offset": 0,
          "end_offset": 10,
          "span_text": "John Smith",
          "edit_type": "insert",
          "confidential_status": "NOT_CONFIDENTIAL",
          "identifier_type": "DIRECT",
          "entity_id": "sample_e1"
        },
        {
          "entity_type": "ORG",
          "entity_mention_id": "sample_em2",
          "start_offset": 20,
          "end_offset": 41,
          "span_text": "Microsoft Corporation",
          "edit_type": "insert",
          "confidential_status": "NOT_CONFIDENTIAL",
          "identifier_type": "DIRECT",
          "entity_id": "sample_e2"
        },
        {
          "entity_type": "GPE",
          "entity_mention_id": "sample_em3",
          "start_offset": 45,
          "end_offset": 52,
          "span_text": "Seattle",
          "edit_type": "insert",
          "confidential_status": "NOT_CONFIDENTIAL",
          "identifier_type": "DIRECT",
          "entity_id": "sample_e3"
        }
      ]
    }
  }
}]
```

### Masked Text Output (`sample_masked.txt`)
```
[PERSON] works at [ORG] in [GPE].
```

## 🔧 Advanced Usage

### Command Line Options

```bash
# Get help
python test.py -h

# Different output formats
python test.py input.txt --format json  # Full TAB format JSON
python test.py input.txt --format txt   # Masked text only

# Custom output file
python test.py input.txt -o results.json
python test.py input.txt -o masked.txt --format txt
```

### Batch Processing

```bash
# Process multiple files
for file in *.txt; do
    python test.py "$file"
done

# Or create a batch script
echo "Processing documents..."
python test.py document1.txt -o doc1_results.json
python test.py document2.txt -o doc2_results.json
python test.py document3.txt -o doc3_results.json
```

### Python Integration

```python
import subprocess
import json

def anonymize_text_file(input_file, output_file=None):
    """Anonymize a text file and return results."""
    cmd = ['python', 'test.py', input_file]
    if output_file:
        cmd.extend(['-o', output_file])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        # Load results
        output_file = output_file or f"{input_file.split('.')[0]}_predictions.json"
        with open(output_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Error: {result.stderr}")
        return None

# Usage
results = anonymize_text_file('my_document.txt')
if results:
    print(f"Found {len(results[0]['annotations']['annotator_1']['entity_mentions'])} entities")
```

## 📁 Project Structure

```
text-anonymization-benchmark/
├── test.py                      # 🎯 Main inference script (NEW!)
├── environment.yml              # 📦 Complete conda environment
├── download_data.py             # 📥 Model download script
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
1. **Input**: Text file specified via command line
2. **Tokenization**: Longformer tokenizer with offset mapping
3. **Inference**: Trained model predicts MASK labels
4. **Span Extraction**: Convert token predictions to character spans
5. **Entity Classification**: Use spaCy/Annotator for entity types (PERSON, ORG, GPE, etc.)
6. **Masking**: Replace entities with type-specific markers `[PERSON]`, `[ORG]`, etc.
7. **Output Generation**: TAB-format JSON with all metadata + masked text

### Key Features
- **Command Line Interface**: Easy file-based processing
- **Flexible Output**: JSON (TAB format) or plain masked text
- **Entity-Specific Masking**: Uses actual entity types instead of generic `[MASK]`
- **Automatic File Naming**: Generates output files based on input filename
- **Error Handling**: Comprehensive validation and error messages

### Dependencies
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face Longformer
- **spaCy**: Named Entity Recognition
- **NumPy**: Array processing
- **argparse**: Command line parsing

All dependencies are automatically installed via `environment.yml`.

## 📈 Performance

The trained model provides:
- **Entity Detection**: Identifies personal identifiers
- **Entity Classification**: PERSON, ORG, GPE, LOC, MISC, etc.
- **Confidence Scores**: Reliability metrics for each prediction
- **Smart Masking**: Type-aware anonymization

### Entity Types Supported
- **PERSON**: Personal names
- **ORG**: Organizations, companies
- **GPE**: Geopolitical entities (countries, cities)
- **LOC**: Locations, landmarks
- **MISC**: Miscellaneous entities
- **And more** (based on spaCy's entity recognition)

## 🔍 Troubleshooting

### Common Issues

**1. File not found**
```bash
# Ensure your input file exists
ls -la your_file.txt
python test.py your_file.txt
```

**2. CUDA out of memory**
```python
# The script automatically detects and uses available hardware
# For CPU-only processing, it will automatically fall back
```

**3. spaCy model not found**
```bash
# Already included in environment.yml, but if needed:
python -m spacy download en_core_web_md
```

**4. Model file missing**
```bash
# Download the trained model
python download_data.py
# Verify it exists
ls longformer_experiments/long_model.pt
```

**5. Permission errors**
```bash
# Ensure write permissions for output directory
chmod 755 .
```

### Performance Tips

- **Faster processing**: Use shorter texts or process in chunks
- **Better accuracy**: Provide more context (longer texts)
- **Memory optimization**: Script automatically handles device selection

## 📚 Dataset Information

Based on the **Text Anonymization Benchmark (TAB)**:
- **Source**: European Court of Human Rights (ECHR) cases
- **Size**: 1,268 manually annotated documents
- **Format**: Standoff JSON with entity mentions
- **Categories**: PERSON, ORG, LOC, MISC, DATETIME, QUANTITY, CODE

## 🆕 What's New

### Enhanced Features
- **File-based Processing**: No need to modify source code
- **Command Line Interface**: Professional CLI with argparse
- **Flexible Output Formats**: JSON (TAB) or plain text
- **Smart Masking**: Entity-type specific replacement (`[PERSON]`, `[ORG]`, etc.)
- **Automatic File Naming**: Intelligent output file generation
- **Better Error Handling**: Comprehensive validation and user feedback
- **Modular Design**: Clean function-based architecture

### Usage Improvements
- **Zero Code Changes**: Process any text file directly
- **Batch Processing Ready**: Easy integration into workflows
- **Professional Output**: Clean, structured results
- **Development Friendly**: Importable as a Python module

## 🔗 References

- **Original Repository**: [NorskRegnesentral/text-anonymisation-benchmark](https://github.com/NorskRegnesentral/text-anonymisation-benchmark)
- **Original Paper**: [The Text Anonymization Benchmark (TAB)](https://arxiv.org/abs/2202.00443)
- **Longformer Paper**: [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)

## 📄 License

MIT License - see original repository for details.

## 🙏 Acknowledgments

- **Original Authors**: Ildikó Pilán, Pierre Lison, Lilja Øvrelid, Anthi Papadopoulou, David Sánchez, Montserrat Batet
- **Enhanced Version**: Professional CLI interface with file-based processing and improved usability

---

**Ready to anonymize text?** Create a text file and run `python test.py your_file.txt` - it's that simple! 🎉