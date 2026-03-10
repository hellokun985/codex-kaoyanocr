from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from math_question_ocr.data_models import BBox, LineCluster
from math_question_ocr.parsers.fill_blank_parser import FillBlankQuestionParser


class FillBlankQuestionParserTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = FillBlankQuestionParser()

    def test_parse_end_blank(self) -> None:
        lines = [
            LineCluster("l1", BBox(40, 40, 980, 100), text="12. 函数 f(x)=x^2-2x+1 的最小值为 ____ 。", reading_order=1),
        ]

        parsed = self.parser.parse(line_clusters=lines)

        self.assertEqual(parsed.question_no, "12")
        self.assertEqual(parsed.question_type.value, "fill_blank")
        self.assertEqual(len(parsed.blanks), 1)
        self.assertEqual(parsed.blanks[0].blank_index, 1)
        self.assertEqual(parsed.blanks[0].placeholder, "____")
        self.assertEqual(parsed.blanks[0].blank_length, 4)
        self.assertGreaterEqual(parsed.blanks[0].char_start, 0)
        self.assertGreater(parsed.blanks[0].bbox.width, 0)

    def test_parse_middle_multiple_blanks(self) -> None:
        lines = [
            LineCluster("l1", BBox(40, 40, 980, 100), text="13. 若 x=__，y=（ ），则 x+y=____。", reading_order=1),
        ]

        parsed = self.parser.parse(line_clusters=lines)

        self.assertEqual(parsed.question_no, "13")
        self.assertEqual(len(parsed.blanks), 3)
        self.assertEqual([blank.blank_index for blank in parsed.blanks], [1, 2, 3])
        self.assertEqual(parsed.blanks[0].placeholder, "__")
        self.assertTrue(parsed.blanks[1].placeholder.startswith("（"))
        self.assertTrue(parsed.blanks[1].placeholder.endswith("）"))
        self.assertEqual(parsed.blanks[2].placeholder, "____")
        self.assertEqual(parsed.blanks[0].blank_length, 2)
        self.assertGreaterEqual(parsed.blanks[1].blank_length, 2)
        self.assertEqual(parsed.blanks[2].blank_length, 4)
        self.assertLess(parsed.blanks[0].char_start, parsed.blanks[1].char_start)
        self.assertLess(parsed.blanks[1].char_start, parsed.blanks[2].char_start)


if __name__ == "__main__":
    unittest.main()
