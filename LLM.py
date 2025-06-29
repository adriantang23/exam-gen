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
from typing import Generator, List, Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv

# Attempt to import openai – if not available we raise at runtime when used
try:
    import openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore


class LLMExamGenerator:
    """Generate mock questions from parsed academic documents using ChatGPT."""

    def __init__(
        self,
        parsed_output_dir: str | Path = "parsed_output",
        model: str = "gpt-3.5-turbo-1106",
        temperature: float = 0.7,
        max_tokens: int = 512,
        slides_per_group: int = 12,  # Group slides together for lecture processing
    ) -> None:
        self.parsed_output_dir = Path(parsed_output_dir)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.slides_per_group = slides_per_group

        # Load API key from .env or environment
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if openai is None:
            raise ImportError(
                "openai package not installed. `pip install openai` to use LLMExamGenerator."
            )
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Please create a .env file with your key or export it."
            )
        openai.api_key = api_key

        # Define the function schema for create_question
        self.function_schema = [
            {
                "name": "create_question",
                "description": "Create a mock exam question with a probability score.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The mock exam question text."
                        }
                    },
                    "required": ["question"]
                }
            }
        ]

        output_dir = Path("LLM_output")
        output_dir.mkdir(exist_ok=True)
        self.default_output_paths = {
            "homework": output_dir / "homework_questions.json",
            "lecture_slides": output_dir / "lecture_slide_questions.json",
            "previous_exams": output_dir / "previous_exam_questions.json",
        }

    # ------------------------------------------------------------------
    # Public generation helpers
    # ------------------------------------------------------------------
    def generate_from_homeworks(
        self,
        system_prompt: str,
        output_file: str | Path | None = None,
    ) -> List[Dict[str, Any]]:
        """Generate mock questions from homework JSON."""
        return self._generate_for_category(
            category_filename="homework.json",
            system_prompt=system_prompt,
            output_file=output_file or self.default_output_paths["homework"],
        )

    def generate_from_previous_exams(
        self,
        system_prompt: str,
        output_file: str | Path | None = None,
    ) -> List[Dict[str, Any]]:
        """Generate mock questions from previous exam JSON."""
        return self._generate_for_category(
            category_filename="previous_exams.json",
            system_prompt=system_prompt,
            output_file=output_file or self.default_output_paths["previous_exams"],
        )

    def generate_from_lecture_slides(
        self,
        system_prompt: str,
        output_file: str | Path | None = None,
    ) -> List[Dict[str, Any]]:
        """Generate mock questions from lecture slide JSON."""
        return self._generate_for_category(
            category_filename="lecture_slides.json",
            system_prompt=system_prompt,
            output_file=output_file or self.default_output_paths["lecture_slides"],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_for_category(
        self,
        category_filename: str,
        system_prompt: str,
        output_file: str | Path | None,
    ) -> List[Dict[str, Any]]:
        category_path = self.parsed_output_dir / category_filename
        if not category_path.exists():
            raise FileNotFoundError(f"Category file not found: {category_path}")

        with open(category_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Iterate over documents & sections
        all_sections: Generator[tuple[str, str, int, str], None, None] = self._iterate_sections(  # type: ignore
            data)
        
        # Detect if this is lecture slides and group accordingly
        is_lecture_slides = category_filename == "lecture_slides.json"
        if is_lecture_slides:
            all_sections = self._group_sections_for_slides(data)
        else:
            all_sections = self._iterate_sections(data)

        results: List[Dict[str, Any]] = []
        for doc_name, category, section_idx, section_text in all_sections:
            try:
                function_call = self._prompt_llm(section_text, system_prompt, is_lecture_slides)
                results.append(
                    {
                        "source_document": doc_name,
                        "source_category": category,
                        "section_index": section_idx,
                        "created_at": datetime.now().isoformat(),
                        "llm_output": function_call,
                    }
                )
            except Exception as e:  # pragma: no cover
                # Capture errors but keep going for cost-efficiency & robustness
                results.append(
                    {
                        "source_document": doc_name,
                        "source_category": category,
                        "section_index": section_idx,
                        "created_at": datetime.now().isoformat(),
                        "error": str(e),
                    }
                )

        # Write to output JSON
        output_path = Path(output_file) if output_file else self.default_output_paths[category_filename.split('.')[0]]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Generated {len(results)} mock questions → {output_path}")
        return results

    def _iterate_sections(
        self, data: Dict[str, Any]
    ) -> Generator[tuple[str, str, int, str], None, None]:
        """Yield (doc_name, category, section_idx, section_text) tuples."""
        category_name = data.get("category", "unknown")
        for doc in data.get("documents", []):
            doc_name = doc.get("file_name", "unknown")
            for idx, section in enumerate(doc.get("sections", [])):
                # Heuristic: only process non-empty sections that look like questions.
                if section.strip():
                    yield doc_name, category_name, idx, section.strip()

    def _group_sections_for_slides(
        self, data: Dict[str, Any]
    ) -> Generator[tuple[str, str, str, str], None, None]:
        """Group lecture slide sections together for more efficient processing."""
        category_name = data.get("category", "unknown")
        for doc in data.get("documents", []):
            doc_name = doc.get("file_name", "unknown")
            sections = [s.strip() for s in doc.get("sections", []) if s.strip()]
            
            # Group sections into chunks
            for i in range(0, len(sections), self.slides_per_group):
                group = sections[i:i + self.slides_per_group]
                if group:  # Only yield non-empty groups
                    combined_text = "\n\n=== SLIDE BREAK ===\n\n".join(group)
                    group_id = f"{i//self.slides_per_group + 1}"
                    yield doc_name, category_name, f"group_{group_id}", combined_text

    def _prompt_llm(self, section_text: str, system_prompt: str, is_grouped: bool = False) -> Dict[str, Any]:
        """Call the OpenAI ChatCompletion API for a single section."""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "You will be provided with a question from course materials. "
                    "Generate a NEW, similar-style mock exam question. Respond ONLY "
                    "via the `create_question` function call with the new question text."
                ),
            },
            {"role": "user", "content": section_text},
        ]
        
        if is_grouped:
            messages[1]["content"] = (
                "You will be provided with multiple slides from a lecture. "
                "Generate ONE comprehensive mock exam question that covers the key concepts from these slides. "
                "Respond via the `create_question` function call."
            )

        response = openai.ChatCompletion.create(  # type: ignore[attr-defined]
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
            functions=self.function_schema,
            function_call="auto",
        )
        
        choice = response["choices"][0]
        if choice.get("finish_reason") == "function_call":
            return choice["message"]["function_call"]  # type: ignore[index]
        else:
            # Fallback to raw content if function was not invoked
            return {
                "name": "create_question",
                "arguments": {
                    "question": choice["message"].get("content", ""),
                },
            }


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
