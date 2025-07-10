import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# Add parser directory to path
sys.path.append(str(Path(__file__).parent / 'parser'))


def get_file_paths_from_args() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Get file paths from command line arguments."""
    parser = argparse.ArgumentParser(description="LLM Exam Generator CLI")
    parser.add_argument("--slides", help="Path to lecture slides PDF")
    parser.add_argument("--exam", help="Path to past exam PDF") 
    parser.add_argument("--homework", help="Path to homework file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive or (not args.slides and not args.exam and not args.homework):
        return get_file_paths_interactive()
    
    return args.slides, args.exam, args.homework


def get_file_paths_interactive() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Prompt user for file paths interactively."""
    print("LLM Exam Generator CLI")
    print("=" * 50)
    
    slides_path = input("Enter path to lecture slides PDF (leave blank to skip): ").strip()
    exam_path = input("Enter path to past exam PDF (leave blank to skip): ").strip()
    homework_path = input("Enter path to homework file (leave blank to skip): ").strip()

    if not slides_path and not exam_path and not homework_path:
        print("Error: Need at least one file. Exiting.")
        return None, None, None

    return slides_path or None, exam_path or None, homework_path or None


def parse_documents(file_paths: Tuple[Optional[str], Optional[str], Optional[str]]) -> List[str]:
    """Parse documents using the parser module."""
    from parser.parser import DocumentParser
    
    print("🔍 Parsing input files...")
    parser = DocumentParser(use_ocr_fallback=True, clean_symbols=True)
    parsed_texts = []
    
    slides_path, exam_path, homework_path = file_paths
    
    if slides_path:
        print(f"Parsing lecture slides: {slides_path}")
        try:
            sections = parser.parse_document(slides_path)
            parsed_texts.extend(sections)
            print(f"✅ Parsed {len(sections)} sections from lecture slides")
        except Exception as e:
            print(f"❌ Error parsing lecture slides: {e}")
    
    if exam_path:
        print(f"Parsing exam: {exam_path}")
        try:
            sections = parser.parse_document(exam_path)
            parsed_texts.extend(sections)
            print(f"✅ Parsed {len(sections)} sections from exam")
        except Exception as e:
            print(f"❌ Error parsing exam: {e}")
    
    if homework_path:
        print(f"Parsing homework: {homework_path}")
        try:
            sections = parser.parse_document(homework_path)
            parsed_texts.extend(sections)
            print(f"✅ Parsed {len(sections)} sections from homework")
        except Exception as e:
            print(f"❌ Error parsing homework: {e}")
    
    return parsed_texts


def organize_content(file_paths: Tuple[Optional[str], Optional[str], Optional[str]]) -> dict:
    """Organize content using the organizer module."""
    from organizer import DocumentOrganizer, FileCategory
    
    print("📋 Organizing content...")
    organizer = DocumentOrganizer()
    
    slides_path, exam_path, homework_path = file_paths
    files_to_add = []
    
    if slides_path:
        files_to_add.append((slides_path, FileCategory.LECTURE_SLIDES))
    
    if exam_path:
        files_to_add.append((exam_path, FileCategory.PREVIOUS_EXAMS))
    
    if homework_path:
        files_to_add.append((homework_path, FileCategory.HOMEWORK))
    
    # Add files to queue
    added = organizer.add_files_batch(files_to_add)
    print(f"Added {added} files to processing queue")
    
    # Process all files
    results = organizer.process_files()
    print(f"✅ Organized content: {results['processed']} successful, {results['errors']} errors")
    
    return results


def build_and_send_prompts() -> dict:
    """Build prompts and send to LLM using prompt_builder module."""
    from prompt_builder import main as prompt_builder_main
    
    print("🤖 Building prompts and generating questions...")
    
    # Call the original prompt_builder main function
    # This will use the hardcoded sample files and generate questions
    prompt_builder_main()
    
    return {"status": "completed"}


def save_output() -> None:
    """Save and display the generated output."""
    print("💾 Saving output...")
    
    # Check for generated files
    output_dir = Path("LLM_output")
    if output_dir.exists():
        print("Generated files:")
        for file_path in output_dir.glob("*.json"):
            print(f"  📄 {file_path}")
    
    # Also check for the generated practice exam
    practice_exam_path = Path("generated/practice_exam.txt")
    if practice_exam_path.exists():
        print(f"📝 Practice exam saved to: {practice_exam_path}")
        with open(practice_exam_path, 'r') as f:
            content = f.read()
            print("\n" + "="*50)
            print("GENERATED PRACTICE EXAM")
            print("="*50)
            print(content[:500] + "..." if len(content) > 500 else content)
    
    print("✅ Generation complete!")


def main():
    """Main CLI function that follows the pipeline."""
    try:
        # 1. Get file paths from user (CLI args or interactive)
        file_paths = get_file_paths_from_args()
        if not any(file_paths):
            return
        
        # 2. Parse documents using parser.py
        parsed_texts = parse_documents(file_paths)
        if not parsed_texts:
            print("No content was successfully parsed. Exiting.")
            return
        
        # 3. Organize content using organizer.py
        organize_results = organize_content(file_paths)
        if organize_results['processed'] == 0:
            print("No files were successfully organized. Exiting.")
            return
        
        # 4. Build prompts and send to LLM using prompt_builder.py
        llm_results = build_and_send_prompts()
        
        # 5. Save and display output
        save_output()
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
