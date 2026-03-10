# DEPRECATED
# 该文件属于历史链路，已不再是当前推荐路径的一部分。
#
# 当前唯一推荐入口：
# - main.py
#
# 当前推荐实现路径：
# - src/math_question_ocr/data_models.py
# - src/math_question_ocr/image_preprocessor.py
# - src/math_question_ocr/ocr_stub.py
# - src/math_question_ocr/line_clusterer.py
# - src/math_question_ocr/question_classifier.py
# - src/math_question_ocr/rule_classifier.py
# - src/math_question_ocr/minimal_pipeline.py
# - src/math_question_ocr/parsers/single_choice_parser.py
# - src/math_question_ocr/parsers/fill_blank_parser.py
# - src/math_question_ocr/parsers/solution_question_parser.py
#
# 该文件仅作为历史实现保留，不属于当前推荐主链路。
# 不要在该文件上继续新增功能。

from __future__ import annotations

import argparse
import json

from .adapters import PaddleOCRAdapter
from .pipeline import MathQuestionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="数学题截图结构化识别")
    parser.add_argument("image", help="输入截图路径")
    parser.add_argument("--debug-dir", default="", help="调试输出目录")
    parser.add_argument("--use-paddleocr", action="store_true", help="启用 PaddleOCR")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = MathQuestionPipeline(ocr_engine=PaddleOCRAdapter() if args.use_paddleocr else None)
    document = pipeline.parse(
        image_path=args.image,
        debug_output_dir=args.debug_dir or None,
    )
    print(json.dumps(document.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
