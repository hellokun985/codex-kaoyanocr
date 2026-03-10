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

from dataclasses import dataclass

from .adapters import FigureDetector, FormulaRecognizer, NullFormulaRecognizer, OCREngine, OpenCVFigureDetector
from .classifier import QuestionTypeClassifier
from .parsers import FillBlankParser, SingleChoiceParser, SolutionParser
from .schemas import DetectedBlock, QuestionDocument, QuestionType, SegmentationResult
from .segmenter import LayoutSegmenter
from .utils import sort_blocks
from .visualizer import DebugVisualizer


@dataclass(slots=True)
class ParseArtifacts:
    blocks: list[DetectedBlock]
    segmentation: SegmentationResult
    document: QuestionDocument


class MathQuestionPipeline:
    def __init__(
        self,
        *,
        ocr_engine: OCREngine | None = None,
        formula_recognizer: FormulaRecognizer | None = None,
        figure_detector: FigureDetector | None = None,
    ) -> None:
        self.ocr_engine = ocr_engine
        self.formula_recognizer = formula_recognizer or NullFormulaRecognizer()
        self.figure_detector = figure_detector or OpenCVFigureDetector()
        self.classifier = QuestionTypeClassifier()
        self.segmenter = LayoutSegmenter()
        self.visualizer = DebugVisualizer()
        self.parsers = {
            QuestionType.SINGLE_CHOICE: SingleChoiceParser(),
            QuestionType.FILL_BLANK: FillBlankParser(),
            QuestionType.SOLUTION: SolutionParser(),
        }

    def parse(
        self,
        *,
        image_path: str | None = None,
        blocks: list[DetectedBlock] | None = None,
        debug_output_dir: str | None = None,
    ) -> QuestionDocument:
        artifacts = self.parse_with_artifacts(
            image_path=image_path,
            blocks=blocks,
            debug_output_dir=debug_output_dir,
        )
        return artifacts.document

    def parse_with_artifacts(
        self,
        *,
        image_path: str | None = None,
        blocks: list[DetectedBlock] | None = None,
        debug_output_dir: str | None = None,
    ) -> ParseArtifacts:
        collected_blocks = self._collect_blocks(image_path=image_path, blocks=blocks)
        question_type = self.classifier.classify(collected_blocks)
        figures = self._collect_figures(image_path=image_path, blocks=collected_blocks)
        segmentation = self.segmenter.segment(collected_blocks, figures, question_type)
        parser = self.parsers[question_type]
        document = parser.parse(segmentation)

        if debug_output_dir:
            self.visualizer.dump(debug_output_dir, collected_blocks, segmentation, document)

        return ParseArtifacts(blocks=collected_blocks, segmentation=segmentation, document=document)

    def _collect_blocks(
        self,
        *,
        image_path: str | None,
        blocks: list[DetectedBlock] | None,
    ) -> list[DetectedBlock]:
        if blocks:
            ordered = sort_blocks(blocks)
        elif image_path and self.ocr_engine:
            text_blocks = self.ocr_engine.detect(image_path)
            formula_blocks = self.formula_recognizer.recognize(image_path, text_blocks)
            ordered = sort_blocks(text_blocks + formula_blocks)
        else:
            raise ValueError("必须提供 blocks，或同时提供 image_path 和 ocr_engine。")

        return ordered

    def _collect_figures(self, *, image_path: str | None, blocks: list[DetectedBlock]):
        if image_path:
            return self.figure_detector.detect(image_path, blocks)
        return [self._to_figure(block) for block in blocks if block.kind == "figure"]

    @staticmethod
    def _to_figure(block: DetectedBlock):
        from .schemas import FigureRegion

        return FigureRegion(
            figure_id=block.block_id,
            bbox=block.bbox,
            figure_type=block.metadata.get("figure_type", "diagram"),
            caption=block.text,
            source=block.source,
        )
