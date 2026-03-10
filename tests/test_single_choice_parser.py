from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from math_question_ocr.data_models import BBox, FigureRegion, LineCluster, OCRBlock, OCRBlockType, Region, RegionType
from math_question_ocr.parsers.single_choice_parser import SingleChoiceQuestionParser


class SingleChoiceQuestionParserTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = SingleChoiceQuestionParser()

    def test_parse_vertical_options_from_line_clusters(self) -> None:
        lines = [
            LineCluster("l1", BBox(40, 40, 980, 100), text="1. 已知函数 y=x^2，则其最小值为（5分）", reading_order=1),
            LineCluster("l2", BBox(70, 130, 420, 180), text="A. 0", reading_order=2),
            LineCluster("l3", BBox(70, 190, 420, 240), text="B. 1", reading_order=3),
            LineCluster("l4", BBox(70, 250, 420, 300), text="C. 2", reading_order=4),
            LineCluster("l5", BBox(70, 310, 420, 360), text="D. 3", reading_order=5),
        ]

        parsed = self.parser.parse(line_clusters=lines)

        self.assertEqual(parsed.question_no, "1")
        self.assertEqual(parsed.score, 5.0)
        self.assertEqual(parsed.stem, "已知函数 y=x^2，则其最小值为")
        self.assertEqual([option.label for option in parsed.options], ["A", "B", "C", "D"])
        self.assertEqual(parsed.options[1].text, "1")

    def test_parse_horizontal_options_from_regions(self) -> None:
        regions = [
            Region("r1", RegionType.STEM_TEXT, BBox(40, 40, 980, 110), text="2．已知集合 M={1,2}，N={2,3}，则 M∩N="),
            Region("r2", RegionType.STEM_TEXT, BBox(70, 130, 980, 190), text="A. {1}   B. {2}   C. {3}   D. {1,2}"),
        ]

        parsed = self.parser.parse(regions=regions)

        self.assertEqual(parsed.question_no, "2")
        self.assertEqual(len(parsed.options), 4)
        self.assertEqual(parsed.options[0].text, "{1}")
        self.assertEqual(parsed.options[3].text, "{1,2}")

    def test_parse_with_ocr_label_tolerance_from_regions(self) -> None:
        regions = [
            Region("r1", RegionType.STEM_TEXT, BBox(40, 40, 980, 110), text="3. 若 x>0，则 ln x 的定义域是"),
            Region("r2", RegionType.STEM_TEXT, BBox(70, 130, 980, 190), text="4. x>0   8. x>=0   C. x<0   O. R"),
        ]

        parsed = self.parser.parse(regions=regions)

        self.assertEqual([option.label for option in parsed.options], ["A", "B", "C", "D"])
        self.assertEqual(parsed.options[0].text, "x>0")
        self.assertEqual(parsed.options[1].text, "x>=0")
        self.assertEqual(parsed.options[3].text, "R")

    def test_parse_from_ocr_blocks_and_keep_figure(self) -> None:
        blocks = [
            OCRBlock("b1", "4.", BBox(40, 40, 80, 90), line_id="l1"),
            OCRBlock("b2", "如图，在△ABC中，AB=", BBox(100, 40, 500, 90), line_id="l1"),
            OCRBlock("b3", "A.", BBox(70, 500, 110, 550), line_id="l2", block_type=OCRBlockType.TEXT),
            OCRBlock("b4", "3", BBox(120, 500, 150, 550), line_id="l2", block_type=OCRBlockType.TEXT),
            OCRBlock("b5", "B.", BBox(260, 500, 300, 550), line_id="l2", block_type=OCRBlockType.TEXT),
            OCRBlock("b6", "4", BBox(310, 500, 340, 550), line_id="l2", block_type=OCRBlockType.TEXT),
            OCRBlock("b7", "C.", BBox(450, 500, 490, 550), line_id="l2", block_type=OCRBlockType.TEXT),
            OCRBlock("b8", "5", BBox(500, 500, 530, 550), line_id="l2", block_type=OCRBlockType.TEXT),
            OCRBlock("b9", "D.", BBox(640, 500, 680, 550), line_id="l2", block_type=OCRBlockType.TEXT),
            OCRBlock("b10", "6", BBox(690, 500, 720, 550), line_id="l2", block_type=OCRBlockType.TEXT),
        ]
        figure = FigureRegion(
            region_id="fig1",
            region_type=RegionType.FIGURE,
            bbox=BBox(280, 120, 720, 420),
            text="几何图",
            block_ids=["f1"],
            reading_order=2,
            figure_type="geometry_diagram",
            caption="三角形图",
        )

        parsed = self.parser.parse(ocr_blocks=blocks, regions=[figure])

        self.assertEqual(parsed.question_no, "4")
        self.assertTrue(parsed.has_figure)
        self.assertEqual(len(parsed.figures), 1)
        self.assertEqual([option.text for option in parsed.options], ["3", "4", "5", "6"])


if __name__ == "__main__":
    unittest.main()
