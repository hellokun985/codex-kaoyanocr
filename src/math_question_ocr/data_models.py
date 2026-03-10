from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any


class QuestionType(str, Enum):
    SINGLE_CHOICE = "single_choice"
    FILL_BLANK = "fill_blank"
    SOLUTION = "solution"


class OCRBlockType(str, Enum):
    TEXT = "text"
    FORMULA = "formula"
    FIGURE = "figure"


class RegionType(str, Enum):
    QUESTION_NO = "question_no"
    SCORE = "score"
    STEM_TEXT = "stem_text"
    STEM_FORMULA = "stem_formula"
    OPTION = "option"
    BLANK = "blank"
    SUBQUESTION = "subquestion"
    FIGURE = "figure"
    UNKNOWN = "unknown"


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


@dataclass(slots=True)
class JsonSerializable:
    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(slots=True)
class BBox(JsonSerializable):
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
    def area(self) -> int:
        return self.width * self.height


@dataclass(slots=True)
class OCRBlock(JsonSerializable):
    block_id: str
    text: str
    bbox: BBox
    confidence: float = 1.0
    block_type: OCRBlockType = OCRBlockType.TEXT
    line_id: str = ""
    source: str = "paddleocr"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LineCluster(JsonSerializable):
    line_id: str
    bbox: BBox
    blocks: list[OCRBlock] = field(default_factory=list)
    text: str = ""
    reading_order: int = 0

    def __post_init__(self) -> None:
        if not self.text and self.blocks:
            self.text = " ".join(block.text for block in self.blocks if block.text).strip()


@dataclass(slots=True)
class Region(JsonSerializable):
    region_id: str
    region_type: RegionType
    bbox: BBox
    text: str = ""
    block_ids: list[str] = field(default_factory=list)
    reading_order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FigureRegion(Region):
    figure_type: str = "unknown"
    caption: str = ""

    def __post_init__(self) -> None:
        if self.region_type == RegionType.UNKNOWN:
            self.region_type = RegionType.FIGURE


@dataclass(slots=True)
class BlankRegion(Region):
    blank_index: int = 1
    placeholder: str = "__"
    blank_length: int = 0
    char_start: int = -1
    char_end: int = -1
    expected_answer_type: str = ""

    def __post_init__(self) -> None:
        if self.region_type == RegionType.UNKNOWN:
            self.region_type = RegionType.BLANK


@dataclass(slots=True)
class ParsedOption(JsonSerializable):
    label: str
    text: str
    regions: list[Region] = field(default_factory=list)
    is_correct: bool | None = None
    explanation: str = ""


@dataclass(slots=True)
class SubQuestion(JsonSerializable):
    marker: str
    stem: str
    stem_segments: list[Region] = field(default_factory=list)
    layout_regions: list[Region] = field(default_factory=list)
    score: float | None = None


@dataclass(slots=True)
class ParsedQuestion(JsonSerializable):
    question_no: str
    score: float | None
    question_type: QuestionType
    has_figure: bool = False
    figures: list[FigureRegion] = field(default_factory=list)
    stem_segments: list[Region] = field(default_factory=list)
    stem: str = ""
    options: list[ParsedOption] = field(default_factory=list)
    blanks: list[BlankRegion] = field(default_factory=list)
    subquestions: list[SubQuestion] = field(default_factory=list)
    layout_regions: list[Region] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.question_type, str):
            self.question_type = QuestionType(self.question_type)
        if self.figures and not self.has_figure:
            self.has_figure = True


def build_single_choice_example() -> ParsedQuestion:
    stem_region = Region(
        region_id="stem_1",
        region_type=RegionType.STEM_TEXT,
        bbox=BBox(40, 40, 960, 120),
        text="已知集合 A={1,2,3}，B={2,3,4}，则 A∩B=",
        block_ids=["b1", "b2"],
        reading_order=1,
    )
    option_a = Region(
        region_id="opt_a_1",
        region_type=RegionType.OPTION,
        bbox=BBox(70, 140, 420, 190),
        text="A. {1,2}",
        block_ids=["b3"],
        reading_order=2,
        metadata={"label": "A"},
    )
    option_b = Region(
        region_id="opt_b_1",
        region_type=RegionType.OPTION,
        bbox=BBox(500, 140, 860, 190),
        text="B. {2,3}",
        block_ids=["b4"],
        reading_order=3,
        metadata={"label": "B"},
    )
    option_c = Region(
        region_id="opt_c_1",
        region_type=RegionType.OPTION,
        bbox=BBox(70, 210, 420, 260),
        text="C. {3,4}",
        block_ids=["b5"],
        reading_order=4,
        metadata={"label": "C"},
    )
    option_d = Region(
        region_id="opt_d_1",
        region_type=RegionType.OPTION,
        bbox=BBox(500, 210, 860, 260),
        text="D. {1,4}",
        block_ids=["b6"],
        reading_order=5,
        metadata={"label": "D"},
    )
    return ParsedQuestion(
        question_no="1",
        score=5.0,
        question_type=QuestionType.SINGLE_CHOICE,
        stem_segments=[stem_region],
        stem=stem_region.text,
        options=[
            ParsedOption(label="A", text="{1,2}", regions=[option_a]),
            ParsedOption(label="B", text="{2,3}", regions=[option_b]),
            ParsedOption(label="C", text="{3,4}", regions=[option_c]),
            ParsedOption(label="D", text="{1,4}", regions=[option_d]),
        ],
        layout_regions=[stem_region, option_a, option_b, option_c, option_d],
    )


def build_fill_blank_example() -> ParsedQuestion:
    stem_region = Region(
        region_id="stem_1",
        region_type=RegionType.STEM_TEXT,
        bbox=BBox(40, 40, 980, 120),
        text="函数 f(x)=x^2-2x+1 的最小值为 ____ 。",
        block_ids=["b1", "b2"],
        reading_order=1,
    )
    blank_region = BlankRegion(
        region_id="blank_1",
        region_type=RegionType.BLANK,
        bbox=BBox(560, 58, 760, 98),
        text="____",
        block_ids=["b2"],
        reading_order=2,
        blank_index=1,
        placeholder="____",
        expected_answer_type="number",
    )
    return ParsedQuestion(
        question_no="12",
        score=4.0,
        question_type=QuestionType.FILL_BLANK,
        stem_segments=[stem_region],
        stem=stem_region.text,
        blanks=[blank_region],
        layout_regions=[stem_region, blank_region],
    )


def build_solution_example() -> ParsedQuestion:
    stem_region = Region(
        region_id="stem_1",
        region_type=RegionType.STEM_TEXT,
        bbox=BBox(40, 40, 980, 120),
        text="已知等差数列 {a_n} 满足 a_1=2，a_3=6。",
        block_ids=["b1"],
        reading_order=1,
    )
    sub_region_1 = Region(
        region_id="sub_1",
        region_type=RegionType.SUBQUESTION,
        bbox=BBox(70, 140, 980, 210),
        text="（1）求数列 {a_n} 的通项公式；",
        block_ids=["b2"],
        reading_order=2,
    )
    sub_region_2 = Region(
        region_id="sub_2",
        region_type=RegionType.SUBQUESTION,
        bbox=BBox(70, 225, 980, 295),
        text="（2）求前 n 项和 S_n。",
        block_ids=["b3"],
        reading_order=3,
    )
    return ParsedQuestion(
        question_no="18",
        score=12.0,
        question_type=QuestionType.SOLUTION,
        stem_segments=[stem_region],
        stem=stem_region.text,
        subquestions=[
            SubQuestion(
                marker="（1）",
                stem="求数列 {a_n} 的通项公式；",
                stem_segments=[sub_region_1],
                layout_regions=[sub_region_1],
                score=6.0,
            ),
            SubQuestion(
                marker="（2）",
                stem="求前 n 项和 S_n。",
                stem_segments=[sub_region_2],
                layout_regions=[sub_region_2],
                score=6.0,
            ),
        ],
        layout_regions=[stem_region, sub_region_1, sub_region_2],
    )


def build_figure_question_example() -> ParsedQuestion:
    stem_region = Region(
        region_id="stem_1",
        region_type=RegionType.STEM_TEXT,
        bbox=BBox(40, 40, 980, 120),
        text="如图，在 Rt△ABC 中，∠C=90°，AC=3，BC=4，则 AB=",
        block_ids=["b1"],
        reading_order=1,
    )
    figure_region = FigureRegion(
        region_id="fig_1",
        region_type=RegionType.FIGURE,
        bbox=BBox(280, 130, 720, 460),
        text="几何示意图",
        block_ids=["f1"],
        reading_order=2,
        figure_type="geometry_diagram",
        caption="直角三角形 ABC 图",
    )
    option_a = Region(
        region_id="opt_a_1",
        region_type=RegionType.OPTION,
        bbox=BBox(70, 500, 320, 550),
        text="A. 4",
        block_ids=["b2"],
        reading_order=3,
        metadata={"label": "A"},
    )
    option_b = Region(
        region_id="opt_b_1",
        region_type=RegionType.OPTION,
        bbox=BBox(360, 500, 610, 550),
        text="B. 5",
        block_ids=["b3"],
        reading_order=4,
        metadata={"label": "B"},
    )
    option_c = Region(
        region_id="opt_c_1",
        region_type=RegionType.OPTION,
        bbox=BBox(650, 500, 900, 550),
        text="C. 6",
        block_ids=["b4"],
        reading_order=5,
        metadata={"label": "C"},
    )
    return ParsedQuestion(
        question_no="7",
        score=5.0,
        question_type=QuestionType.SINGLE_CHOICE,
        has_figure=True,
        figures=[figure_region],
        stem_segments=[stem_region, figure_region],
        stem=stem_region.text,
        options=[
            ParsedOption(label="A", text="4", regions=[option_a]),
            ParsedOption(label="B", text="5", regions=[option_b]),
            ParsedOption(label="C", text="6", regions=[option_c]),
        ],
        layout_regions=[stem_region, figure_region, option_a, option_b, option_c],
    )


def build_all_examples() -> dict[str, ParsedQuestion]:
    return {
        "single_choice": build_single_choice_example(),
        "fill_blank": build_fill_blank_example(),
        "solution": build_solution_example(),
        "with_figure": build_figure_question_example(),
    }


PARSED_QUESTION_JSON_SCHEMA_STYLE: dict[str, Any] = {
    "title": "ParsedQuestion",
    "type": "object",
    "required": [
        "question_no",
        "score",
        "question_type",
        "has_figure",
        "figures",
        "stem_segments",
        "stem",
        "options",
        "blanks",
        "subquestions",
        "layout_regions",
    ],
    "properties": {
        "question_no": {"type": "string", "description": "题号"},
        "score": {"type": ["number", "null"], "description": "分值"},
        "question_type": {
            "type": "string",
            "enum": ["single_choice", "fill_blank", "solution"],
            "description": "题型",
        },
        "has_figure": {"type": "boolean", "description": "是否带图"},
        "figures": {
            "type": "array",
            "items": {"$ref": "#/definitions/FigureRegion"},
            "description": "题目中的图形区域",
        },
        "stem_segments": {
            "type": "array",
            "items": {"$ref": "#/definitions/Region"},
            "description": "题干分段，支持文本/公式/图形混排",
        },
        "stem": {"type": "string", "description": "拍平后的题干文本"},
        "options": {
            "type": "array",
            "items": {"$ref": "#/definitions/ParsedOption"},
            "description": "单选题选项",
        },
        "blanks": {
            "type": "array",
            "items": {"$ref": "#/definitions/BlankRegion"},
            "description": "填空位置",
        },
        "subquestions": {
            "type": "array",
            "items": {"$ref": "#/definitions/SubQuestion"},
            "description": "解答题小问",
        },
        "layout_regions": {
            "type": "array",
            "items": {"$ref": "#/definitions/Region"},
            "description": "版面级区域集合",
        },
    },
    "definitions": {
        "BBox": {
            "type": "object",
            "required": ["x1", "y1", "x2", "y2"],
            "properties": {
                "x1": {"type": "integer"},
                "y1": {"type": "integer"},
                "x2": {"type": "integer"},
                "y2": {"type": "integer"},
            },
        },
        "OCRBlock": {
            "type": "object",
            "required": ["block_id", "text", "bbox", "confidence", "block_type"],
        },
        "Region": {
            "type": "object",
            "required": ["region_id", "region_type", "bbox"],
        },
        "FigureRegion": {
            "allOf": [{"$ref": "#/definitions/Region"}],
            "description": "图形区域",
        },
        "BlankRegion": {
            "allOf": [{"$ref": "#/definitions/Region"}],
            "description": "填空区域",
        },
        "ParsedOption": {
            "type": "object",
            "required": ["label", "text", "regions"],
        },
        "SubQuestion": {
            "type": "object",
            "required": ["marker", "stem", "stem_segments", "layout_regions"],
        },
    },
}
