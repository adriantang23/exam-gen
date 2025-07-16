"""
LLM Exam Generator
==================

This module provides a class (`LLMExamGenerator`) that leverages the OpenAI
ChatCompletion API to generate mock exam questions from previously parsed
academic documents (homeworks, lecture slides, previous exams).

Key Features
------------
1. Reads structured JSON files from the `parsed_output/` directory (produced by
   `DocumentOrganizer`).
2. Iterates through each section and calls OpenAI with a custom system prompt:
   - For homework and exams: one API call per question/section
   - For lecture slides: groups slides together (default 12 per group) for efficiency
3. Utilises the function-calling feature to ask ChatGPT to respond with a
   structured `create_question` call containing the generated mock question.
4. Separate generation methods for homework, lecture slides, and previous
   exams so that prompts can be tuned per source type.
5. Loads the OpenAI API key from a local `.env` file – keeps secrets out of git.
6. Designed to be cost-efficient and handle large numbers of slides effectively.

Requirements
------------
* python-dotenv (`pip install python-dotenv`)
* openai >= 0.28 (or compatible)

A sample `.env` (NOT checked into git):
```
OPENAI_API_KEY="sk-FAKE_KEY_FOR_DEMO_PURPOSES"
```

The default model is now `gpt-3.5-turbo-1106` (supports function calling). You
can pass a different model when instantiating `LLMExamGenerator` or via CLI
with `--model`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Attempt to import openai – if not available we raise at runtime when used
try:
    import openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore


class LLMExamGenerator:
    """
    Generate a full practice exam in a single GPT call using lecture context and past exam examples.
    """
    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if openai is None:
            raise ImportError("openai package not installed. `pip install openai` to use LLMExamGenerator.")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set. Please create a .env file with your key or export it.")
        openai.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_exam(
        self,
        lecture_context: str,
        exam_examples: List[str],
        num_questions: int = 5,
        include_answers: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a full practice exam in a single GPT call.

        Args:
            lecture_context: Cleaned and organized lecture material as a string.
            exam_examples: List of past exam question strings (few-shot examples).
            num_questions: Number of questions to generate (default 5).
            include_answers: If True, also request answers/solutions.

        Returns:
            Dict with keys 'questions' (list of str) and optionally 'answers' (list of str).
        """
        system_message = (
            "You are a university professor writing a final exam based on the provided lecture material and past exam examples. "
            f"Create {num_questions} new questions in the same style."
        )
        user_content = (
            f"LECTURE MATERIAL:\n{lecture_context}\n\n"
            f"PAST EXAM EXAMPLES:\n"
            + "\n---\n".join(exam_examples)
            + "\n\nINSTRUCTIONS:\n"
            f"- Write {num_questions} new, original questions for a practice exam.\n"
            "- Each question should be clear and self-contained.\n"
            "- Format as Q1, Q2, ...\n"
            + ("- For each question, also provide a detailed answer/solution.\n" if include_answers else "")
            + "- Do NOT copy the examples verbatim.\n"
            "- Return the questions in a readable, numbered format."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response["choices"][0]["message"]["content"]
            # Optionally, parse questions/answers if possible
            return {"raw": content}
        except Exception as e:
            return {"error": str(e)}


# ----------------------------------------------------------------------
# CLI helper for quick testing (optional)
# ----------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Generate mock questions with LLM.")
    parser.add_argument(
        "--source",
        choices=["homework", "slides", "exams"],
        required=True,
        help="Which parsed source to generate from",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Custom system prompt to steer ChatGPT",
    )
    parser.add_argument(
        "--out",
        required=False,
        default=None,
        help="Output JSON file (default auto)"
    )
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help="Override the default model"
    )
    parser.add_argument(
        "--slides-per-group",
        type=int,
        required=False,
        default=12,
        help="Number of slides to group together for lecture processing (default: 12)"
    )
    args = parser.parse_args()

    generator = LLMExamGenerator()

    if args.model:
        generator.model = args.model
        
    generator = LLMExamGenerator(slides_per_group=args.slides_per_group)

    if args.source == "homework":
        generator.generate_from_homeworks(args.prompt, args.out)
    elif args.source == "slides":
        generator.generate_from_lecture_slides(args.prompt, args.out)
    else:
        generator.generate_from_previous_exams(args.prompt, args.out)
