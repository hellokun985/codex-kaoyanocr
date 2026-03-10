from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from math_question_ocr.data_models import BBox, LineCluster, OCRBlock, OCRBlockType, QuestionType
from math_question_ocr.parsers.solution_question_parser import SolutionQuestionParser


class SolutionQuestionParserTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = SolutionQuestionParser()

    def test_parse_bracket_subquestions_with_score(self) -> None:
        lines = [
            LineCluster("l1", BBox(40, 40, 980, 100), text="18. 已知等差数列 {a_n} 满足 a_1=2，a_3=6（12分）", reading_order=1),
            LineCluster("l2", BBox(70, 130, 980, 185), text="（1）求数列 {a_n} 的通项公式；", reading_order=2),
            LineCluster("l3", BBox(70, 200, 980, 255), text="（2）求前 n 项和 S_n。", reading_order=3),
        ]

        parsed = self.parser.parse(line_clusters=lines)

        self.assertEqual(parsed.question_no, "18")
        self.assertEqual(parsed.score, 12.0)
        self.assertEqual(parsed.question_type, QuestionType.SOLUTION)
        self.assertIn("已知等差数列", parsed.stem)
        self.assertEqual([sub.marker for sub in parsed.subquestions], ["（1）", "（2）"])
        self.assertEqual(parsed.subquestions[0].stem, "求数列 {a_n} 的通项公式；")

    def test_parse_roman_subquestions_and_multiline_formula(self) -> None:
        blocks = [
            OCRBlock("b1", "20.", BBox(40, 40, 80, 90), line_id="l1"),
            OCRBlock("b2", "已知矩阵 A=", BBox(100, 40, 300, 90), line_id="l1"),
            OCRBlock("b3", "[[1,2],[3,4]]", BBox(320, 40, 560, 90), line_id="l1", block_type=OCRBlockType.FORMULA),
            OCRBlock("b4", "I.", BBox(70, 130, 110, 180), line_id="l2"),
            OCRBlock("b5", "求 det(A)；", BBox(130, 130, 360, 180), line_id="l2"),
            OCRBlock("b6", "II.", BBox(70, 200, 120, 250), line_id="l3"),
            OCRBlock("b7", "若", BBox(130, 200, 170, 250), line_id="l3"),
            OCRBlock("b8", "A x = b", BBox(190, 200, 360, 250), line_id="l3", block_type=OCRBlockType.FORMULA),
            OCRBlock("b9", "，求 x。", BBox(380, 200, 500, 250), line_id="l3"),
            OCRBlock("b10", "其中 x=(x1,x2)^T，b=(1,0)^T。", BBox(130, 265, 700, 315), line_id="l4", block_type=OCRBlockType.FORMULA),
        ]

        parsed = self.parser.parse(ocr_blocks=blocks)

        self.assertEqual(parsed.question_no, "20")
        self.assertEqual(len(parsed.subquestions), 2)
        self.assertEqual(parsed.subquestions[0].marker, "I.")
        self.assertEqual(parsed.subquestions[1].marker, "II.")
        self.assertIn("A x = b", parsed.subquestions[1].stem)
        self.assertIn("x=(x1,x2)^T", parsed.subquestions[1].stem)
        self.assertTrue(any(region.metadata.get("has_formula_block") for region in parsed.subquestions[1].stem_segments))

    def test_marker_only_line_attaches_following_content(self) -> None:
        lines = [
            LineCluster("l1", BBox(40, 40, 980, 100), text="21. 已知函数 f(x) 满足某条件。", reading_order=1),
            LineCluster("l2", BBox(70, 130, 120, 180), text="III.", reading_order=2),
            LineCluster("l3", BBox(130, 130, 980, 180), text="证明 f(x) 在区间 [0,1] 上单调递增。", reading_order=3),
        ]

        parsed = self.parser.parse(line_clusters=lines)

        self.assertEqual(len(parsed.subquestions), 1)
        self.assertEqual(parsed.subquestions[0].marker, "III.")
        self.assertEqual(parsed.subquestions[0].stem, "证明 f(x) 在区间 [0,1] 上单调递增。")


if __name__ == "__main__":
    unittest.main()
