# Document Parser

A comprehensive PDF and PPTX parser designed for academic documents with math symbol support.

## 🚀 Quick Setup

```bash
# One-time setup
./setup.sh

# Or manually:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📖 Usage

```bash
# Activate virtual environment (each time)
source venv/bin/activate

# Parse documents
python parser.py path/to/document.pdf
python parser.py path/to/presentation.pptx

# Options
python parser.py document.pdf --no-clean  # Keep raw symbols
```

## 🧪 Test with Sample Files

```bash
# Test PDF parsing
python parser.py scanable_pdf_test_documents/02-the-basics-A1.pdf

# Test PPTX parsing  
python parser.py scanable_pdf_test_documents/Reinforcement_Learning.pptx
```

## 💻 Programmatic Usage

```python
from parser import parse_document, DocumentParser

# Simple usage
pages = parse_document("document.pdf")
print(f"First page: {pages[0]}")

# Advanced usage
parser = DocumentParser(clean_symbols=True)
info = parser.get_document_info("document.pdf")
pages = parser.parse_document("document.pdf")
```

## ✨ Features

- **PDF**: Multiple extraction methods with math symbol support
- **PPTX**: Comprehensive extraction (titles, content, tables, notes, charts)
- **Symbol Cleaning**: Converts misinterpreted math symbols to readable text
- **Virtual Environment**: Isolated dependencies
- **Error Handling**: Graceful fallbacks for problematic content
- **No Permission Issues**: Removed OCR to avoid macOS security prompts

## 🧹 Cleanup

```bash
# Remove virtual environment
./setup.sh clean
# or
rm -rf venv
``` 