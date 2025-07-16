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

## Full Pipeline Overview & Usage

This project provides a **one-command pipeline** from your course files to a formatted PDF of LLM-generated practice questions.

### Pipeline Steps
1. **Input Selection:**
   - The CLI prompts you for lecture slides, past exams, or homework files (PDF, LaTeX, etc).
2. **Parsing:**
   - Files are parsed and cleaned for text content.
3. **Organization:**
   - Content is organized into lecture context and example questions.
4. **LLM Generation:**
   - A single GPT call generates a full set of new practice questions in one shot.
5. **Output:**
   - Questions are saved to a text file, then automatically formatted into a LaTeX document and compiled to a PDF.

### How to Use (Step-by-Step)

1. **Clone the Repo:**
   ```bash
   git clone https://github.com/adriantang23/exam-gen.git
   cd exam-gen
   ```
2. **Set Up Python Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Install a TeX Distribution:**
   - **Mac:** `brew install --cask mactex` (or install BasicTeX)
   - **Linux:** `sudo apt-get install texlive-latex-base`
   - **Windows:** Download and install from https://miktex.org/download
4. **Set Your OpenAI API Key:**
   - Create a `.env` file in the project root with:
     ```
     OPENAI_API_KEY=sk-...
     ```
5. **Run the CLI:**
   ```bash
   python3 CLI.py --interactive
   ```
   or specify files directly:
   ```bash
   python3 CLI.py --homework scanable_pdf_test_documents/CS237hw10.tex
   ```
6. **Get Your PDF:**
   - The generated PDF will be at `generated/new_practice_exam.pdf`.

### Notes
- The pipeline is fully automated: input files → PDF, no manual steps required.
- If you encounter LaTeX errors, check the `.log` file in the `generated/` directory for details.
- For best results, ensure your input files are clean and machine-readable.

---
