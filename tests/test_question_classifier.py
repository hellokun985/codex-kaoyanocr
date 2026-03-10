from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from math_question_ocr.data_models import BBox, LineCluster, OCRBlock, QuestionType
from math_question_ocr.question_classifier import RuleBasedQuestionClassifier


class QuestionClassifierTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = RuleBasedQuestionClassifier()

    def test_single_choice_by_option_labels(self) -> None:
        lines = [
            LineCluster("l1", BBox(40, 40, 980, 90), text="1. 已知集合 M={1,2}, N={2,3}，则 M∩N=", reading_order=1),
            LineCluster("l2", BBox(70, 120, 980, 180), text="A. {1}  B. {2}  C. {3}  D. {1,2}", reading_order=2),
        ]

        result = self.classifier.classify(line_clusters=lines)

        self.assertEqual(result.question_type, QuestionType.SINGLE_CHOICE)
        self.assertGreaterEqual(result.confidence, 0.8)
        self.assertTrue(any("选项标签" in reason for reason in result.reasons))

    def test_fill_blank_by_blank_slots(self) -> None:
        lines = [
            LineCluster("l1", BBox(40, 40, 980, 90), text="12. 函数 f(x)=x^2-2x+1 的最小值为 ____ 。", reading_order=1),
            LineCluster("l2", BBox(70, 120, 980, 180), text="若 x=2，则 f(x)=（  ）", reading_order=2),
        ]

        result = self.classifier.classify(line_clusters=lines)

        self.assertEqual(result.question_type, QuestionType.FILL_BLANK)
        self.assertGreaterEqual(result.confidence, 0.8)
        self.assertTrue(any("空白线" in reason or "答案槽位" in reason for reason in result.reasons))

    def test_solution_by_subquestions_full_score_and_keywords(self) -> None:
        lines = [
            LineCluster("l1", BBox(40, 40, 980, 95), text="18. 本题满分 12 分。已知数列 {a_n} 满足 a_1=1。", reading_order=1),
            LineCluster("l2", BBox(70, 120, 980, 175), text="（1）求数列 {a_n} 的通项公式；", reading_order=2),
            LineCluster("l3", BBox(70, 185, 980, 240), text="（2）证明数列 {b_n} 单调递增。", reading_order=3),
        ]

        result = self.classifier.classify(line_clusters=lines)

        self.assertEqual(result.question_type, QuestionType.SOLUTION)
        self.assertGreaterEqual(result.confidence, 0.75)
        self.assertTrue(any("小问标记" in reason for reason in result.reasons))
        self.assertTrue(any("满分" in reason for reason in result.reasons))

    def test_solution_by_long_body_and_matrix_hints(self) -> None:
        lines = [
            LineCluster("l1", BBox(40, 40, 980, 95), text="已知矩阵 A=[[1,2],[3,4]]，设向量 x 满足 Ax=b。", reading_order=1),
            LineCluster("l2", BBox(70, 120, 980, 175), text="求解线性方程组，并说明当参数变化时解的情况。", reading_order=2),
            LineCluster("l3", BBox(70, 185, 980, 240), text="请写出详细过程。", reading_order=3),
        ]

        result = self.classifier.classify(line_clusters=lines)

        self.assertEqual(result.question_type, QuestionType.SOLUTION)
        self.assertGreaterEqual(result.score_breakdown["solution"], result.score_breakdown["single_choice"])

    def test_fallback_to_ocr_blocks(self) -> None:
        blocks = [
            OCRBlock("b1", "A. 1", BBox(70, 120, 220, 170), line_id="l1"),
            OCRBlock("b2", "8. 2", BBox(240, 120, 390, 170), line_id="l1"),
            OCRBlock("b3", "C. 3", BBox(410, 120, 560, 170), line_id="l1"),
            OCRBlock("b4", "O. 4", BBox(580, 120, 730, 170), line_id="l1"),
        ]

        result = self.classifier.classify(ocr_blocks=blocks, line_clusters=[])

        self.assertEqual(result.question_type, QuestionType.SINGLE_CHOICE)
        self.assertGreaterEqual(result.confidence, 0.7)


if __name__ == "__main__":
    unittest.main()
