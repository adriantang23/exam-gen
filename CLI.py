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


def build_and_send_prompts(parsed_texts: List[str]) -> dict:
    """Build prompts and send to LLM using the new batch method."""
    from LLM import LLMExamGenerator
    
    print("🤖 Building prompts and generating questions...")
    
    # Create lecture context from parsed texts
    lecture_context = "\n\n".join(parsed_texts[:3])  # Use first 3 sections as context
    
    # Create exam examples from parsed texts (use different sections)
    exam_examples = parsed_texts[3:6] if len(parsed_texts) >= 6 else parsed_texts[1:3]
    
    # Instantiate the new LLMExamGenerator
    generator = LLMExamGenerator(model="gpt-4", temperature=0.2)
    result = generator.generate_exam(
        lecture_context=lecture_context,
        exam_examples=exam_examples,
        num_questions=5,
        include_answers=False,
    )
    
    # Print the result
    if "error" in result:
        print(f"❌ LLM call failed: {result['error']}")
        return {"status": "error", "message": result['error']}
    else:
        print("\n=== GENERATED PRACTICE EXAM ===\n")
        print(result["raw"])
        
        # Save to a text file for reference
        output_dir = Path("generated")
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / "practice_exam.txt", "w") as f:
            f.write(result["raw"])
        
        return {"status": "completed", "content": result["raw"]}


def save_output_and_generate_pdf() -> None:
    """Read the generated text file and create a PDF automatically."""
    from output import questions_to_pdf
    practice_exam_path = Path("generated/practice_exam.txt")
    if practice_exam_path.exists():
        print("📝 Reading from practice_exam.txt...")
        with open(practice_exam_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Extract questions from the content (simple parsing)
        questions = []
        lines = content.split('\n')
        current_question = ""
        for line in lines:
            if line.strip().startswith('Q') and '.' in line:
                if current_question.strip():
                    questions.append(current_question.strip())
                current_question = line
            elif line.strip():
                current_question += "\n" + line
        if current_question.strip():
            questions.append(current_question.strip())
        if questions:
            print(f"📝 Found {len(questions)} questions from practice_exam.txt")
            pdf_path = questions_to_pdf(questions, filename="new_practice_exam", title="New Practice Exam")
            if pdf_path:
                print(f"🎉 New practice exam PDF created at: {pdf_path}")
            else:
                print("❌ Failed to generate PDF from new practice exam")
        else:
            print("❌ No questions found in practice_exam.txt")
    else:
        print("❌ practice_exam.txt not found. Cannot generate PDF.")


def main():
    """Main CLI function that follows the pipeline end-to-end."""
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
        # 4. Build prompts and send to LLM using the new batch method
        llm_results = build_and_send_prompts(parsed_texts)
        # 5. Save and display output, and generate PDF
        save_output_and_generate_pdf()
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
