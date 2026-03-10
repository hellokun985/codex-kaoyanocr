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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuestionType(str, Enum):
    SINGLE_CHOICE = "single_choice"
    FILL_BLANK = "fill_blank"
    SOLUTION = "solution"


@dataclass(slots=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    def to_dict(self) -> dict[str, int]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass(slots=True)
class DetectedBlock:
    block_id: str
    kind: str
    text: str
    bbox: BBox
    score: float = 1.0
    source: str = "rule"
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self, *, text: str | None = None, metadata: dict[str, Any] | None = None) -> "DetectedBlock":
        return DetectedBlock(
            block_id=self.block_id,
            kind=self.kind,
            text=self.text if text is None else text,
            bbox=self.bbox,
            score=self.score,
            source=self.source,
            metadata=dict(self.metadata if metadata is None else metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "kind": self.kind,
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class FigureRegion:
    figure_id: str
    bbox: BBox
    figure_type: str = "unknown"
    caption: str = ""
    source: str = "opencv"

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "bbox": self.bbox.to_dict(),
            "figure_type": self.figure_type,
            "caption": self.caption,
            "source": self.source,
        }


@dataclass(slots=True)
class StemSegment:
    segment_id: str
    kind: str
    content: str
    bbox: BBox | None = None
    order: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = {
            "segment_id": self.segment_id,
            "kind": self.kind,
            "content": self.content,
            "order": self.order,
        }
        if self.bbox is not None:
            data["bbox"] = self.bbox.to_dict()
        return data


@dataclass(slots=True)
class OptionItem:
    label: str
    text: str
    bbox: BBox | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {"label": self.label, "text": self.text}
        if self.bbox is not None:
            data["bbox"] = self.bbox.to_dict()
        return data


@dataclass(slots=True)
class BlankItem:
    blank_id: str
    placeholder: str
    bbox: BBox | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {"blank_id": self.blank_id, "placeholder": self.placeholder}
        if self.bbox is not None:
            data["bbox"] = self.bbox.to_dict()
        return data


@dataclass(slots=True)
class SubQuestionItem:
    marker: str
    stem: str
    stem_segments: list[StemSegment] = field(default_factory=list)
    bbox: BBox | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "marker": self.marker,
            "stem": self.stem,
            "stem_segments": [segment.to_dict() for segment in self.stem_segments],
        }
        if self.bbox is not None:
            data["bbox"] = self.bbox.to_dict()
        return data


@dataclass(slots=True)
class QuestionDocument:
    question_no: str = ""
    score: float | None = None
    question_type: str = ""
    has_figure: bool = False
    figures: list[FigureRegion] = field(default_factory=list)
    stem_segments: list[StemSegment] = field(default_factory=list)
    stem: str = ""
    options: list[OptionItem] = field(default_factory=list)
    blanks: list[BlankItem] = field(default_factory=list)
    subquestions: list[SubQuestionItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_no": self.question_no,
            "score": self.score,
            "question_type": self.question_type,
            "has_figure": self.has_figure,
            "figures": [figure.to_dict() for figure in self.figures],
            "stem_segments": [segment.to_dict() for segment in self.stem_segments],
            "stem": self.stem,
            "options": [option.to_dict() for option in self.options],
            "blanks": [blank.to_dict() for blank in self.blanks],
            "subquestions": [subquestion.to_dict() for subquestion in self.subquestions],
        }


@dataclass(slots=True)
class SegmentationResult:
    question_type: QuestionType
    question_no: str = ""
    score: float | None = None
    cleaned_blocks: list[DetectedBlock] = field(default_factory=list)
    stem_blocks: list[DetectedBlock] = field(default_factory=list)
    option_groups: list[tuple[str, list[DetectedBlock]]] = field(default_factory=list)
    blank_blocks: list[DetectedBlock] = field(default_factory=list)
    subquestion_groups: list[tuple[str, list[DetectedBlock]]] = field(default_factory=list)
    figures: list[FigureRegion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type.value,
            "question_no": self.question_no,
            "score": self.score,
            "cleaned_blocks": [block.to_dict() for block in self.cleaned_blocks],
            "stem_blocks": [block.to_dict() for block in self.stem_blocks],
            "option_groups": [
                {"label": label, "blocks": [block.to_dict() for block in blocks]}
                for label, blocks in self.option_groups
            ],
            "blank_blocks": [block.to_dict() for block in self.blank_blocks],
            "subquestion_groups": [
                {"marker": marker, "blocks": [block.to_dict() for block in blocks]}
                for marker, blocks in self.subquestion_groups
            ],
            "figures": [figure.to_dict() for figure in self.figures],
        }
