"""
Output Module for LLM Exam Generator

This module provides functionality to convert LLM-generated questions into
clean, readable PDF exams using LaTeX formatting.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional


def escape_latex(text: str) -> str:
    """
    Escape special LaTeX characters in the given text.
    
    Args:
        text: The text to escape
        
    Returns:
        The text with LaTeX special characters properly escaped
    """
    # Define LaTeX special characters and their escaped versions
    latex_escapes = {
        '\\': r'\textbackslash{}',
        '{': r'\{',
        '}': r'\}',
        '$': r'\$',
        '&': r'\&',
        '#': r'\#',
        '^': r'\textasciicircum{}',
        '_': r'\_',
        '~': r'\textasciitilde{}',
        '%': r'\%',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
        '|': r'\textbar{}',
    }
    
    # Apply all escapes
    escaped_text = text
    for char, escape in latex_escapes.items():
        escaped_text = escaped_text.replace(char, escape)
    
    return escaped_text


def create_latex_document(questions: List[str], title: str = "Practice Exam") -> str:
    """
    Create a complete LaTeX document from a list of questions.
    
    Args:
        questions: List of question strings
        title: Title for the exam
        
    Returns:
        Complete LaTeX document as a string
    """
    # Escape the title
    escaped_title = escape_latex(title)
    
    # Start the LaTeX document
    latex_content = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{geometry}",
        r"\geometry{margin=1in}",
        r"\usepackage{enumitem}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{6pt}",
        "",
        r"\begin{document}",
        "",
        r"\begin{center}",
        f"\\Large\\textbf{{{escaped_title}}}",
        r"\end{center}",
        "",
        r"\vspace{0.5cm}",
        "",
        r"\textbf{Instructions:}",
        r"\begin{itemize}",
        r"\item Read each question carefully.",
        r"\item Show all your work for full credit.",
        r"\item You may use a calculator unless otherwise specified.",
        r"\end{itemize}",
        "",
        r"\vspace{0.5cm}",
        "",
        r"\begin{enumerate}[label=\textbf{Q\arabic*.}]",
    ]
    
    # Add each question
    for question in questions:
        if question.strip():  # Only add non-empty questions
            escaped_question = escape_latex(question.strip())
            # Remove any existing question number prefix
            cleaned_question = re.sub(r'^Q\d+\.\s*', '', escaped_question)
            latex_content.append(f"    \\item {cleaned_question}")
            latex_content.append("")
    
    # Close the document
    latex_content.extend([
        r"\end{enumerate}",
        "",
        r"\end{document}"
    ])
    
    return "\n".join(latex_content)


def compile_latex_to_pdf(tex_file_path: Path, output_dir: Path) -> Optional[Path]:
    import os
    env = os.environ.copy()
    env["PATH"] += ":/Library/TeX/texbin"  # or the correct path for your system
    
    # Debug: Check if tex file exists before running pdflatex
    print(f"🔍 Checking if tex file exists: {tex_file_path}")
    if not tex_file_path.exists():
        print(f"❌ Tex file not found: {tex_file_path}")
        return None
    
    print(f"🔍 Tex file size: {tex_file_path.stat().st_size} bytes")
    print(f"🔍 Working directory: {os.getcwd()}")
    print(f"🔍 Output directory: {output_dir}")
    
    try:
        # Run pdflatex to compile the .tex file to PDF
        # When cwd=output_dir, we need to pass just the filename, not the full path
        cmd = ['pdflatex', '-interaction=nonstopmode', tex_file_path.name]
        print(f"🔍 Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=output_dir,
            env=env  # Pass the environment
        )
        
        print(f"🔍 pdflatex return code: {result.returncode}")
        print(f"🔍 pdflatex stdout: {result.stdout[:200]}...")
        print(f"🔍 pdflatex stderr: {result.stderr[:200]}...")
        
        pdf_path = output_dir / f"{tex_file_path.stem}.pdf"
        print(f"🔍 Expected PDF path: {pdf_path}")
        print(f"🔍 PDF exists: {pdf_path.exists()}")
        
        # If PDF exists, treat as success even if return code is nonzero (LaTeX warnings)
        if pdf_path.exists():
            if result.returncode != 0:
                print(f"⚠️ pdflatex finished with warnings or minor errors (return code: {result.returncode}) but PDF was created.")
                print(f"See log file for details: {output_dir}/{tex_file_path.stem}.log")
            return pdf_path
        else:
            print(f"❌ PDF file not found at expected location: {pdf_path}")
            print(f"Error output: {result.stderr}")
            return None
    except FileNotFoundError:
        print("❌ pdflatex not found. Please install LaTeX (e.g., MacTeX, TeX Live, or MiKTeX)")
        return None
    except Exception as e:
        print(f"❌ Error during LaTeX compilation: {e}")
        return None


def questions_to_pdf(
    questions: List[str], 
    output_dir: str = "generated/", 
    filename: str = "practice_exam",
    title: str = "Practice Exam"
) -> Optional[Path]:
    """
    Convert a list of questions to a PDF exam.
    
    Args:
        questions: List of question strings
        output_dir: Directory to save output files
        filename: Base filename (without extension)
        title: Title for the exam
        
    Returns:
        Path to the generated PDF file, or None if generation failed
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Create file paths
    tex_file_path = output_path / f"{filename}.tex"
    pdf_file_path = output_path / f"{filename}.pdf"
    
    try:
        # Generate LaTeX content
        print(f"📝 Generating LaTeX document...")
        latex_content = create_latex_document(questions, title)
        
        # Write LaTeX file
        with open(tex_file_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        print(f"✅ LaTeX file saved to: {tex_file_path}")
        
        # Compile to PDF
        print(f"🔄 Compiling PDF...")
        generated_pdf_path = compile_latex_to_pdf(tex_file_path, output_path)
        
        if generated_pdf_path:
            print(f"✅ PDF generated successfully: {generated_pdf_path}")
            return generated_pdf_path
        else:
            print(f"❌ Failed to generate PDF")
            return None
            
    except Exception as e:
        print(f"❌ Error during PDF generation: {e}")
        return None


def extract_questions_from_llm_output(llm_output_dir: str = "LLM_output/") -> List[str]:
    """
    Extract questions from LLM output JSON files.
    
    Args:
        llm_output_dir: Directory containing LLM output files
        
    Returns:
        List of extracted questions
    """
    import json
    
    questions = []
    output_dir = Path(llm_output_dir)
    
    if not output_dir.exists():
        print(f"❌ LLM output directory not found: {llm_output_dir}")
        return questions
    
    # Process each JSON file in the output directory
    for json_file in output_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract questions from each item
            for item in data:
                if "llm_output" in item and "arguments" in item["llm_output"]:
                    # Handle both string and dict formats for arguments
                    arguments = item["llm_output"]["arguments"]
                    if isinstance(arguments, str):
                        # Parse JSON string
                        try:
                            parsed_args = json.loads(arguments)
                            question = parsed_args.get("question", "")
                        except json.JSONDecodeError:
                            # If it's not valid JSON, treat it as the question directly
                            question = arguments
                    else:
                        # Arguments is already a dict
                        question = arguments.get("question", "")
                    
                    if question.strip():
                        questions.append(question.strip())
                elif "error" in item:
                    print(f"⚠️  Skipping item with error: {item['error']}")
                    
        except Exception as e:
            print(f"❌ Error reading {json_file}: {e}")
    
    return questions


def main():
    """
    Main function to demonstrate the output module.
    """
    print("🎯 LLM Exam Generator - Output Module")
    print("=" * 50)
    
    # First, try to read from the new text file format
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
    
    # Also try to extract questions from LLM output if available
    print(f"\n📁 Checking for LLM output files...")
    extracted_questions = extract_questions_from_llm_output()
    
    if extracted_questions:
        print(f"📝 Found {len(extracted_questions)} questions from LLM output")
        llm_pdf_path = questions_to_pdf(extracted_questions, filename="llm_generated_exam", title="LLM Generated Practice Exam")
        
        if llm_pdf_path:
            print(f"🎉 LLM-generated PDF created at: {llm_pdf_path}")
        else:
            print("❌ Failed to generate PDF from LLM output")
    else:
        print("📝 No questions found in LLM output files")


if __name__ == "__main__":
    main()
