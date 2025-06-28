#!/bin/bash

# Parser Setup Script
# This script sets up and manages the virtual environment for the document parser

set -e  # Exit on any error

echo "🚀 Document Parser Setup"
echo "======================="

# Function to create virtual environment
setup_venv() {
    echo "📦 Creating virtual environment..."
    if [ ! -d "venv" ]; then
        python -m venv venv
        echo "✅ Virtual environment created"
    else
        echo "ℹ️  Virtual environment already exists"
    fi
}

# Function to activate venv and install dependencies
install_deps() {
    echo "📥 Installing dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
}

# Function to test the parser
test_parser() {
    echo "🧪 Testing parser..."
    source venv/bin/activate
    echo "Testing PDF parsing..."
    python parser.py scanable_pdf_test_documents/02-the-basics-A1.pdf | head -10
    echo ""
    echo "Testing PPTX parsing..."
    python parser.py scanable_pdf_test_documents/Reinforcement_Learning.pptx | head -10
    echo "✅ Parser tests completed"
}

# Function to show usage instructions
show_usage() {
    echo ""
    echo "📖 Usage Instructions:"
    echo "====================="
    echo ""
    echo "To activate the virtual environment:"
    echo "  source venv/bin/activate"
    echo ""
    echo "To run the parser:"
    echo "  python parser.py <file_path>"
    echo ""
    echo "Examples:"
    echo "  python parser.py scanable_pdf_test_documents/02-the-basics-A1.pdf"
    echo "  python parser.py scanable_pdf_test_documents/Reinforcement_Learning.pptx"
    echo ""
    echo "Options:"
    echo "  --no-ocr     Disable OCR fallback"
    echo "  --no-clean   Disable symbol cleaning"
    echo ""
    echo "To deactivate when done:"
    echo "  deactivate"
}

# Main setup process
case "${1:-setup}" in
    "setup")
        setup_venv
        install_deps
        show_usage
        ;;
    "test")
        test_parser
        ;;
    "clean")
        echo "🧹 Cleaning up..."
        rm -rf venv
        echo "✅ Virtual environment removed"
        ;;
    *)
        echo "Usage: $0 [setup|test|clean]"
        echo "  setup (default) - Set up virtual environment and install dependencies"
        echo "  test           - Test the parser with sample files"
        echo "  clean          - Remove virtual environment"
        exit 1
        ;;
esac 