from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from math_question_ocr.minimal_pipeline import MinimalQuestionPipeline
from math_question_ocr.ocr_stub import SidecarJsonOCREngine


PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WnRk9sAAAAASUVORK5CYII="
)


class MinimalPipelineTestCase(unittest.TestCase):
    def test_pipeline_outputs_single_choice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "question.png"
            image_path.write_bytes(base64.b64decode(PNG_1X1))

            ocr_json = tmp_path / "question.png.ocr.json"
            ocr_json.write_text(
                json.dumps(
                    {
                        "blocks": [
                            {
                                "block_id": "b1",
                                "text": "1. 已知函数 y=x^2，则最小值为（5分）",
                                "bbox": {"x1": 40, "y1": 40, "x2": 980, "y2": 100},
                                "line_id": "l1",
                            },
                            {
                                "block_id": "b2",
                                "text": "A. 0",
                                "bbox": {"x1": 70, "y1": 140, "x2": 250, "y2": 190},
                                "line_id": "l2",
                            },
                            {
                                "block_id": "b3",
                                "text": "B. 1",
                                "bbox": {"x1": 280, "y1": 140, "x2": 460, "y2": 190},
                                "line_id": "l2",
                            },
                            {
                                "block_id": "b4",
                                "text": "C. 2",
                                "bbox": {"x1": 490, "y1": 140, "x2": 670, "y2": 190},
                                "line_id": "l2",
                            },
                            {
                                "block_id": "b5",
                                "text": "D. 3",
                                "bbox": {"x1": 700, "y1": 140, "x2": 880, "y2": 190},
                                "line_id": "l2",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            pipeline = MinimalQuestionPipeline(ocr_engine=SidecarJsonOCREngine())
            artifacts = pipeline.run(str(image_path))

            self.assertEqual(artifacts.parsed_question.question_no, "1")
            self.assertEqual(artifacts.parsed_question.score, 5.0)
            self.assertEqual(artifacts.parsed_question.question_type.value, "single_choice")
            self.assertEqual(len(artifacts.parsed_question.options), 4)
            self.assertEqual(artifacts.parsed_question.options[2].text, "2")
            region_types = [region.region_type.value for region in artifacts.parsed_question.layout_regions]
            self.assertIn("question_no", region_types)
            self.assertIn("score", region_types)


if __name__ == "__main__":
    unittest.main()
