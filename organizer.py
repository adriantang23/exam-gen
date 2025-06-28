"""
Document Organizer for Academic Content

This module provides functionality to organize and categorize academic documents
(homework, lecture slides, previous exams) using the parser and storing them
in structured JSON files for easy retrieval and processing.
"""

import os
import json
import sys
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Add parser directory to path to import the parser
sys.path.append(str(Path(__file__).parent / 'parser'))
from parser import DocumentParser


class FileCategory(Enum):
    """Enumeration for different types of academic documents."""
    HOMEWORK = "homework"
    LECTURE_SLIDES = "lecture_slides"
    PREVIOUS_EXAMS = "previous_exams"


class DocumentOrganizer:
    """
    Organizes and categorizes academic documents by parsing and storing them
    in structured format based on their category.
    """
    
    def __init__(self, output_dir: str = "parsed_output"):
        """
        Initialize the DocumentOrganizer.
        
        Args:
            output_dir: Directory where organized files will be stored
        """
        self.output_dir = Path(output_dir)
        self.pending_files = []  # List of (file_path, category) tuples
        self.parser = DocumentParser(use_ocr_fallback=True, clean_symbols=True)
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize storage files if they don't exist
        self._initialize_storage_files()
    
    def _initialize_storage_files(self):
        """Create initial JSON files for each category if they don't exist."""
        for category in FileCategory:
            file_path = self.output_dir / f"{category.value}.json"
            if not file_path.exists():
                initial_data = {
                    "category": category.value,
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "documents": []
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(initial_data, f, indent=2, ensure_ascii=False)
    
    def add_file(self, file_path: str, category: FileCategory) -> bool:
        """
        Add a file to the processing queue.
        
        Args:
            file_path: Path to the file to be processed
            category: Category of the document
            
        Returns:
            True if file was added successfully, False otherwise
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return False
        
        if file_path.suffix.lower() not in ['.pdf', '.pptx', '.tex']:
            print(f"Error: Unsupported file format: {file_path.suffix}")
            return False
        
        self.pending_files.append((file_path, category))
        print(f"Added {file_path.name} to queue as {category.value}")
        return True
    
    def add_files_batch(self, file_category_pairs: List[tuple]) -> int:
        """
        Add multiple files at once.
        
        Args:
            file_category_pairs: List of (file_path, category) tuples
            
        Returns:
            Number of files successfully added
        """
        success_count = 0
        for file_path, category in file_category_pairs:
            if self.add_file(file_path, category):
                success_count += 1
        return success_count
    
    def process_files(self, use_ocr: bool = True, clean_symbols: bool = True) -> Dict[str, Any]:
        """
        Process all pending files and store them in organized format.
        
        Args:
            use_ocr: Whether to use OCR fallback for PDFs
            clean_symbols: Whether to apply symbol cleaning
            
        Returns:
            Dictionary with processing results and statistics
        """
        if not self.pending_files:
            print("No files in queue to process.")
            return {"processed": 0, "errors": 0, "results": []}
        
        # Update parser settings
        self.parser = DocumentParser(use_ocr_fallback=use_ocr, clean_symbols=clean_symbols)
        
        results = []
        processed_count = 0
        error_count = 0
        
        print(f"Processing {len(self.pending_files)} files...")
        print("=" * 50)
        
        for file_path, category in self.pending_files:
            try:
                print(f"Processing: {file_path.name} ({category.value})")
                
                # Parse the document
                sections = self.parser.parse_document(file_path)
                
                # Get document info
                doc_info = self.parser.get_document_info(file_path)
                
                # Create document record
                document_record = {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "category": category.value,
                    "processed_at": datetime.now().isoformat(),
                    "file_size": doc_info.get('file_size', 0),
                    "file_extension": doc_info.get('file_extension', ''),
                    "num_sections": len(sections),
                    "total_characters": sum(len(section) for section in sections),
                    "sections": sections,
                    "parsing_info": {
                        "use_ocr": use_ocr,
                        "clean_symbols": clean_symbols,
                        "non_empty_sections": len([s for s in sections if s.strip()])
                    }
                }
                
                # Store in appropriate category file
                self._store_document(document_record, category)
                
                results.append({
                    "file": file_path.name,
                    "category": category.value,
                    "status": "success",
                    "sections": len(sections)
                })
                
                processed_count += 1
                print(f"✅ Successfully processed {file_path.name} ({len(sections)} sections)")
                
            except Exception as e:
                error_msg = f"Failed to process {file_path.name}: {str(e)}"
                print(f"❌ {error_msg}")
                
                results.append({
                    "file": file_path.name,
                    "category": category.value,
                    "status": "error",
                    "error": str(e)
                })
                
                error_count += 1
        
        # Clear processed files
        self.pending_files.clear()
        
        summary = {
            "processed": processed_count,
            "errors": error_count,
            "total": processed_count + error_count,
            "results": results
        }
        
        print("=" * 50)
        print(f"Processing complete: {processed_count} successful, {error_count} errors")
        
        return summary
    
    def _store_document(self, document_record: Dict[str, Any], category: FileCategory):
        """
        Store a document record in the appropriate category file.
        
        Args:
            document_record: Document data to store
            category: Category of the document
        """
        file_path = self.output_dir / f"{category.value}.json"
        
        # Load existing data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Update metadata
        data["last_updated"] = datetime.now().isoformat()
        
        # Add document (check for duplicates by file name)
        existing_doc_names = {doc["file_name"] for doc in data["documents"]}
        if document_record["file_name"] not in existing_doc_names:
            data["documents"].append(document_record)
        else:
            # Update existing document
            for i, doc in enumerate(data["documents"]):
                if doc["file_name"] == document_record["file_name"]:
                    data["documents"][i] = document_record
                    break
        
        # Save updated data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_category_summary(self, category: FileCategory) -> Dict[str, Any]:
        """
        Get summary information for a specific category.
        
        Args:
            category: Category to summarize
            
        Returns:
            Summary dictionary with statistics
        """
        file_path = self.output_dir / f"{category.value}.json"
        
        if not file_path.exists():
            return {"error": f"No data found for category: {category.value}"}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = data["documents"]
        
        summary = {
            "category": category.value,
            "total_documents": len(documents),
            "total_sections": sum(doc["num_sections"] for doc in documents),
            "total_characters": sum(doc["total_characters"] for doc in documents),
            "file_types": {},
            "documents": [
                {
                    "name": doc["file_name"],
                    "sections": doc["num_sections"],
                    "characters": doc["total_characters"],
                    "processed_at": doc["processed_at"]
                }
                for doc in documents
            ],
            "created_at": data.get("created_at"),
            "last_updated": data.get("last_updated")
        }
        
        # Count file types
        for doc in documents:
            ext = doc["file_extension"]
            summary["file_types"][ext] = summary["file_types"].get(ext, 0) + 1
        
        return summary
    
    def get_all_summaries(self) -> Dict[str, Any]:
        """
        Get summary information for all categories.
        
        Returns:
            Dictionary with summaries for each category
        """
        summaries = {}
        for category in FileCategory:
            summaries[category.value] = self.get_category_summary(category)
        
        return summaries
    
    def search_content(self, query: str, category: Optional[FileCategory] = None) -> List[Dict[str, Any]]:
        """
        Search for content across documents.
        
        Args:
            query: Search term
            category: Optional category to limit search to
            
        Returns:
            List of matching results
        """
        results = []
        categories_to_search = [category] if category else list(FileCategory)
        
        for cat in categories_to_search:
            file_path = self.output_dir / f"{cat.value}.json"
            if not file_path.exists():
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for doc in data["documents"]:
                for i, section in enumerate(doc["sections"]):
                    if query.lower() in section.lower():
                        results.append({
                            "file_name": doc["file_name"],
                            "category": cat.value,
                            "section_index": i,
                            "section_preview": section[:200] + "..." if len(section) > 200 else section,
                            "processed_at": doc["processed_at"]
                        })
        
        return results
    
    def export_category(self, category: FileCategory, output_file: str, format: str = "json"):
        """
        Export a category's data to a file.
        
        Args:
            category: Category to export
            output_file: Output file path
            format: Export format ("json" or "txt")
        """
        file_path = self.output_dir / f"{category.value}.json"
        
        if not file_path.exists():
            print(f"No data found for category: {category.value}")
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if format.lower() == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format.lower() == "txt":
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Category: {category.value.title()}\n")
                f.write("=" * 50 + "\n\n")
                
                for doc in data["documents"]:
                    f.write(f"Document: {doc['file_name']}\n")
                    f.write(f"Processed: {doc['processed_at']}\n")
                    f.write(f"Sections: {doc['num_sections']}\n")
                    f.write("-" * 30 + "\n")
                    
                    for i, section in enumerate(doc['sections'], 1):
                        f.write(f"Section {i}:\n{section}\n\n")
                    
                    f.write("=" * 50 + "\n\n")
        
        print(f"Exported {category.value} to {output_file}")


# Example usage and testing functions
def main():
    """Example usage of the DocumentOrganizer."""
    organizer = DocumentOrganizer()
    
    print("Document Organizer - Example Usage")
    print("=" * 50)
    
    # Example: Add some files (you would replace these with actual file paths)
    sample_files = [
        ("scanable_pdf_test_documents/wa3.tex", FileCategory.HOMEWORK),
        ("scanable_pdf_test_documents/wa8.tex", FileCategory.HOMEWORK),
        ("scanable_pdf_test_documents/CS237hw5.tex", FileCategory.HOMEWORK),
        ("scanable_pdf_test_documents/Reinforcement_Learning.pptx", FileCategory.LECTURE_SLIDES),
        ("scanable_pdf_test_documents/02-the-basics-A1.pdf", FileCategory.PREVIOUS_EXAMS),
    ]
    
    # Add files to queue
    added = organizer.add_files_batch(sample_files)
    print(f"Added {added} files to processing queue")
    
    # Process all files
    results = organizer.process_files()
    
    # Show summaries
    print("\nCategory Summaries:")
    print("=" * 50)
    summaries = organizer.get_all_summaries()
    for category_name, summary in summaries.items():
        if "error" not in summary:
            print(f"{category_name.title()}: {summary['total_documents']} documents, "
                  f"{summary['total_sections']} sections")
    
    return organizer


if __name__ == "__main__":
    main()
