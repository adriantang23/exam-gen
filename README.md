Exam Generator (CLI)

A local-based tool to generate practice exams from university course materials using an LLM  
Designed for CS/math students to upload lecture slides and past exams, then generate new practice tests mimicking actual exam format.

---

cli.py : entry point for CLI app. Handles input, file selection, and starts pipeline.
llm.py : handles requests to LLM for actual test generation. Sends prompts and recieves responses from LLM + error handling.
organizer.py : takes parsed text and organizes it into structure formats for cleaner prompt generation
parser.py : parses and extracts readable text from machine readable PDF files
output.py : handles formatting and saving generated questions for final output
prompt_generation : uses organized text to build final prompt for LLM.

requirements.txt : Python dependencies

---

1. Run the CLI and select your input files
2. Text is parsed and cleaned.
3. Organized into topic outlines and example questions.
4. A prompt is generated and sent to LLM.
5. LLM returns realistic practice questions matching your course.

---

...
