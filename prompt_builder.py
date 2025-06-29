from organizer import DocumentOrganizer, FileCategory
from LLM import LLMExamGenerator


def main():
    organizer = DocumentOrganizer()
    #organizer to parse files and export to json
    sample_files = [
        # ("scanable_pdf_test_documents/wa3.tex", FileCategory.HOMEWORK),
        # ("scanable_pdf_test_documents/wa8.tex", FileCategory.HOMEWORK),
        ("scanable_pdf_test_documents/CS237hw5.tex", FileCategory.HOMEWORK),
        ("scanable_pdf_test_documents/CS237hw09.tex", FileCategory.HOMEWORK),
        ("scanable_pdf_test_documents/CS237hw10.tex", FileCategory.HOMEWORK),
        ("scanable_pdf_test_documents/cs237L24-annotated.pdf", FileCategory.LECTURE_SLIDES),
        ("scanable_pdf_test_documents/cs237L25-annotated.pdf", FileCategory.LECTURE_SLIDES),
        ("scanable_pdf_test_documents/cs237L27-annotated.pdf", FileCategory.LECTURE_SLIDES),
        ("scanable_pdf_test_documents/CS237_Practice_Midterm.pdf", FileCategory.PREVIOUS_EXAMS),
        ("scanable_pdf_test_documents/CS237_Practice_Final.pdf", FileCategory.PREVIOUS_EXAMS),
    ]
    
    # Add files to queue
    added = organizer.add_files_batch(sample_files)
    print(f"Added {added} files to processing queue")
    
    # Process all files
    results = organizer.process_files()

    # run through LLM.py to generate prompts
    generator = LLMExamGenerator()
    generator.generate_from_homeworks("""You are an instructor coming up with an exam. 
    You are looking through previous homework problems for inspiration.
    Generate questions based on this, and be pretty similar to them. 
    Feel free to have multiple parts. NOT EVEN SECTION NEEDS TO HAVE A QUESTION, 
    IF IT DOESNT FEEL APPROPRIATE DO NOT MAKE THE CALL TO GENERATE QUESTION. 
    This is for a probability class""", None)
    generator.generate_from_lecture_slides("You're an instructor. Generate questions for an exam.", None)
    generator.generate_from_previous_exams("You're an instructor. Generate questions for an exam.", None)



if __name__ == "__main__":
    main()