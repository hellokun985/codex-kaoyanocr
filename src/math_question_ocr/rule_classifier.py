from __future__ import annotations

from .data_models import LineCluster, OCRBlock, QuestionType
from .question_classifier import ClassificationResult, RuleBasedQuestionClassifier


class RuleBasedQuestionTypeClassifier:
    def __init__(self) -> None:
        self._classifier = RuleBasedQuestionClassifier()

    def classify(self, lines: list[LineCluster], ocr_blocks: list[OCRBlock] | None = None) -> QuestionType:
        return self.classify_with_details(lines=lines, ocr_blocks=ocr_blocks).question_type

    def classify_with_details(
        self,
        *,
        lines: list[LineCluster],
        ocr_blocks: list[OCRBlock] | None = None,
    ) -> ClassificationResult:
        return self._classifier.classify(line_clusters=lines, ocr_blocks=ocr_blocks or [])
