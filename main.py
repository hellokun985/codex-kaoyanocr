from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from math_question_ocr.minimal_pipeline import MinimalQuestionPipeline
from math_question_ocr.ocr_stub import PlaceholderFigureDetector, SidecarJsonOCREngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="数学题截图结构化识别最小链路")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("--ocr-json", default="", help="OCR sidecar JSON 路径")
    parser.add_argument("--figure-json", default="", help="图形 sidecar JSON 路径")
    parser.add_argument("--preprocess-out", default="", help="预处理输出目录")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = MinimalQuestionPipeline(
        ocr_engine=SidecarJsonOCREngine(ocr_json_path=args.ocr_json or None),
        figure_detector=PlaceholderFigureDetector(figure_json_path=args.figure_json or None),
    )
    artifacts = pipeline.run(
        args.image,
        preprocess_output_dir=args.preprocess_out or None,
    )
    print(
        json.dumps(
            {
                "preprocess": {
                    "original_path": artifacts.preprocess.original_path,
                    "processed_path": artifacts.preprocess.processed_path,
                    "width": artifacts.preprocess.width,
                    "height": artifacts.preprocess.height,
                    "metadata": artifacts.preprocess.metadata,
                },
                "ocr_blocks": [block.to_dict() for block in artifacts.ocr_blocks],
                "line_clusters": [line.to_dict() for line in artifacts.line_clusters],
                "parsed_question": artifacts.parsed_question.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
