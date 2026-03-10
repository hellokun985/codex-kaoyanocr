"""Current public exports for the active main pipeline."""

from .data_models import (
    BBox,
    BlankRegion,
    FigureRegion,
    JsonSerializable,
    LineCluster,
    OCRBlock,
    OCRBlockType,
    ParsedOption,
    ParsedQuestion,
    QuestionType,
    Region,
    RegionType,
    SubQuestion,
)
from .image_preprocessor import ImagePreprocessor, PreprocessResult
from .line_clusterer import YLineClusterer
from .minimal_pipeline import MinimalQuestionPipeline, PipelineArtifacts
from .ocr_stub import (
    FigureDetector,
    FormulaRecognizer,
    OCREngine,
    PlaceholderFigureDetector,
    PlaceholderFormulaRecognizer,
    SidecarJsonOCREngine,
)
from .question_classifier import ClassificationResult, RuleBasedQuestionClassifier
from .rule_classifier import RuleBasedQuestionTypeClassifier

__all__ = [
    "BBox",
    "BlankRegion",
    "ClassificationResult",
    "FigureDetector",
    "FigureRegion",
    "FormulaRecognizer",
    "ImagePreprocessor",
    "JsonSerializable",
    "LineCluster",
    "MinimalQuestionPipeline",
    "OCRBlock",
    "OCRBlockType",
    "OCREngine",
    "ParsedOption",
    "ParsedQuestion",
    "PipelineArtifacts",
    "PlaceholderFigureDetector",
    "PlaceholderFormulaRecognizer",
    "PreprocessResult",
    "QuestionType",
    "Region",
    "RegionType",
    "RuleBasedQuestionClassifier",
    "RuleBasedQuestionTypeClassifier",
    "SidecarJsonOCREngine",
    "SubQuestion",
    "YLineClusterer",
]
