from organizer import DocumentOrganizer, FileCategory
from LLM import LLMExamGenerator


def main():
    """
    Build a single prompt for the LLM to generate a full practice exam in one shot.
    """
    # Example: Use the organizer to get lecture context and past exam examples
    organizer = DocumentOrganizer()
    # For demo, just use a few files (adjust as needed)
    sample_files = [
        ("scanable_pdf_test_documents/CS237hw5.tex", FileCategory.HOMEWORK),
        ("scanable_pdf_test_documents/CS237hw09.tex", FileCategory.HOMEWORK),
        ("scanable_pdf_test_documents/CS237hw10.tex", FileCategory.HOMEWORK),
        # Add more as needed
    ]
    organizer.add_files_batch(sample_files)
    results = organizer.process_files()

    # For this refactor, let's mock the lecture context and exam examples
    # In production, you would extract these from the processed/organized content
    lecture_context = """
    Probability theory is the branch of mathematics concerned with probability. It describes the likelihood of events occurring. Key concepts include random variables, expected value, independence, and conditional probability.
    """
    exam_examples = [
        "Let X be a random variable representing the outcome of a fair 6-sided die. What is the probability that X is even?",
        "Prove that the sum of two independent Poisson random variables is also Poisson distributed.",
        "Explain the difference between discrete and continuous random variables, with examples.",
    ]

    # Instantiate the new LLMExamGenerator
    generator = LLMExamGenerator(model="gpt-4", temperature=0.2)
    result = generator.generate_exam(
        lecture_context=lecture_context,
        exam_examples=exam_examples,
        num_questions=5,
        include_answers=False,
    )

    # Print or save the result
    if "error" in result:
        print(f"❌ LLM call failed: {result['error']}")
    else:
        print("\n=== GENERATED PRACTICE EXAM ===\n")
        print(result["raw"])


if __name__ == "__main__":
    main()
