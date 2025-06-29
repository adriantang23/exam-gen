import os


def main():
    print("LLM Exam Generator CLI")
    
    slides_path = input("Enter path to lecture slides PDF (leave blank to skip): ").strip()
    exam_path = input("Enter path to past exam PDF (leave blank to skip): ").strip()

    if not slides_path and not exam_path:
        print("Error: Need at least one file. Exiting.")
        return

    # 2. Parse PDFs
    from parser.parser import DocumentParser
    print("🔍 Parsing input files...")
    parsed_text = parse_document(slides_path, exam_path)

    # 3. Organize text
    from organizer import main
    print("Organizing content...")
    structured_data = organize_text(parsed_text)

    # 4. Build prompt & call LLM
    from prompt_builder import main
    from LLM import LLMExamGenerator
    print("Generating practice questions with GPT...")
    questions = build_and_send_prompt(structured_data)

    # 5. Output
    print("Generation complete!\n")
    print(questions)
    with open("generated/practice_exam.txt", "w") as f:
        f.write(questions)

if __name__ == "__main__":
    main()
