from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from math_question_ocr.data_models import (
    QuestionType,
    build_figure_question_example,
    build_fill_blank_example,
    build_single_choice_example,
    build_solution_example,
)


class DataModelTestCase(unittest.TestCase):
    def test_single_choice_example(self) -> None:
        question = build_single_choice_example()
        self.assertEqual(question.question_type, QuestionType.SINGLE_CHOICE)
        self.assertEqual(len(question.options), 4)
        self.assertFalse(question.has_figure)

    def test_fill_blank_example(self) -> None:
        question = build_fill_blank_example()
        self.assertEqual(question.question_type, QuestionType.FILL_BLANK)
        self.assertEqual(len(question.blanks), 1)

    def test_solution_example(self) -> None:
        question = build_solution_example()
        self.assertEqual(question.question_type, QuestionType.SOLUTION)
        self.assertEqual(len(question.subquestions), 2)

    def test_figure_example(self) -> None:
        question = build_figure_question_example()
        self.assertTrue(question.has_figure)
        self.assertEqual(len(question.figures), 1)
        self.assertEqual(question.to_dict()["figures"][0]["figure_type"], "geometry_diagram")


if __name__ == "__main__":
    unittest.main()
