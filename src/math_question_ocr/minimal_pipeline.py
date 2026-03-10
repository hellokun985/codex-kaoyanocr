from __future__ import annotations

from dataclasses import dataclass

from .data_models import LineCluster, OCRBlock, ParsedQuestion
from .image_preprocessor import ImagePreprocessor, PreprocessResult
from .line_clusterer import YLineClusterer
from .ocr_stub import FigureDetector, FormulaRecognizer, OCREngine, PlaceholderFigureDetector, PlaceholderFormulaRecognizer
from .parsers.fill_blank_parser import FillBlankQuestionParser
from .parsers.single_choice_parser import SingleChoiceQuestionParser
from .parsers.solution_question_parser import SolutionQuestionParser
from .rule_classifier import RuleBasedQuestionTypeClassifier


@dataclass(slots=True)
class PipelineArtifacts:
    preprocess: PreprocessResult
    ocr_blocks: list[OCRBlock]
    line_clusters: list[LineCluster]
    parsed_question: ParsedQuestion


class MinimalQuestionPipeline:
    def __init__(
        self,
        *,
        preprocessor: ImagePreprocessor | None = None,
        ocr_engine: OCREngine | None = None,
        formula_recognizer: FormulaRecognizer | None = None,
        figure_detector: FigureDetector | None = None,
    ) -> None:
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.ocr_engine = ocr_engine
        self.formula_recognizer = formula_recognizer or PlaceholderFormulaRecognizer()
        self.figure_detector = figure_detector or PlaceholderFigureDetector()
        self.line_clusterer = YLineClusterer()
        self.classifier = RuleBasedQuestionTypeClassifier()
        self.single_choice_parser = SingleChoiceQuestionParser()
        self.fill_blank_parser = FillBlankQuestionParser()
        self.solution_parser = SolutionQuestionParser()

    def run(self, image_path: str, *, preprocess_output_dir: str | None = None) -> PipelineArtifacts:
        preprocess = self.preprocessor.preprocess(image_path, output_dir=preprocess_output_dir)
        ocr_blocks = self.ocr_engine.detect(preprocess.processed_path) if self.ocr_engine else []
        formula_blocks = self.formula_recognizer.recognize(preprocess.processed_path, ocr_blocks)
        merged_blocks = sorted(
            ocr_blocks + formula_blocks,
            key=lambda block: (block.bbox.y1, block.bbox.x1, block.block_id),
        )
        line_clusters = self.line_clusterer.cluster(merged_blocks)
        question_type = self.classifier.classify(line_clusters)
        figures = self.figure_detector.detect(preprocess.processed_path, merged_blocks)

        if question_type.value == "single_choice":
            parsed = self.single_choice_parser.parse(line_clusters=line_clusters, regions=figures)
        elif question_type.value == "fill_blank":
            parsed = self.fill_blank_parser.parse(lines=line_clusters, figures=figures)
        else:
            parsed = self.solution_parser.parse(lines=line_clusters, figures=figures)

        return PipelineArtifacts(
            preprocess=preprocess,
            ocr_blocks=merged_blocks,
            line_clusters=line_clusters,
            parsed_question=parsed,
        )
