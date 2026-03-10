from __future__ import annotations

from .patterns import (
    count_blanks,
    find_option_label,
    find_question_no,
    find_score,
    find_subquestion_marker,
    has_blank,
    strip_option_label,
    strip_question_no,
    strip_score,
    strip_subquestion_marker,
)
from .schemas import DetectedBlock, FigureRegion, QuestionType, SegmentationResult
from .utils import sort_blocks


class LayoutSegmenter:
    def segment(
        self,
        blocks: list[DetectedBlock],
        figures: list[FigureRegion],
        question_type: QuestionType,
    ) -> SegmentationResult:
        ordered_blocks = sort_blocks(blocks)
        cleaned_blocks: list[DetectedBlock] = []
        question_no = ""
        score = None

        for index, block in enumerate(ordered_blocks):
            if block.kind not in {"text", "formula"}:
                cleaned_blocks.append(block)
                continue

            text = block.text
            if index == 0:
                if not question_no:
                    question_no = find_question_no(text)
                    text = strip_question_no(text)
                if score is None:
                    score = find_score(text)
                    text = strip_score(text)
            else:
                if score is None:
                    maybe_score = find_score(text)
                    if maybe_score is not None and block.bbox.y1 <= ordered_blocks[0].bbox.y2 + 10:
                        score = maybe_score
                        text = strip_score(text)

            cleaned_blocks.append(block.clone(text=text))

        if question_type is QuestionType.SINGLE_CHOICE:
            return self._segment_single_choice(cleaned_blocks, figures, question_no, score)
        if question_type is QuestionType.FILL_BLANK:
            return self._segment_fill_blank(cleaned_blocks, figures, question_no, score)
        return self._segment_solution(cleaned_blocks, figures, question_no, score)

    def _segment_single_choice(
        self,
        blocks: list[DetectedBlock],
        figures: list[FigureRegion],
        question_no: str,
        score: float | None,
    ) -> SegmentationResult:
        option_groups: list[tuple[str, list[DetectedBlock]]] = []
        stem_blocks: list[DetectedBlock] = []
        current_option_label = ""
        current_option_blocks: list[DetectedBlock] = []
        options_started = False

        for block in blocks:
            if block.kind not in {"text", "formula"}:
                if options_started and current_option_blocks:
                    current_option_blocks.append(block)
                else:
                    stem_blocks.append(block)
                continue

            label = find_option_label(block.text)
            if label:
                options_started = True
                if current_option_label:
                    option_groups.append((current_option_label, current_option_blocks))
                current_option_label = label
                current_option_blocks = [block.clone(text=strip_option_label(block.text))]
                continue

            if options_started and current_option_blocks:
                current_option_blocks.append(block)
            else:
                stem_blocks.append(block)

        if current_option_label:
            option_groups.append((current_option_label, current_option_blocks))

        return SegmentationResult(
            question_type=QuestionType.SINGLE_CHOICE,
            question_no=question_no,
            score=score,
            cleaned_blocks=blocks,
            stem_blocks=stem_blocks,
            option_groups=option_groups,
            figures=figures,
        )

    def _segment_fill_blank(
        self,
        blocks: list[DetectedBlock],
        figures: list[FigureRegion],
        question_no: str,
        score: float | None,
    ) -> SegmentationResult:
        blank_blocks = [block for block in blocks if block.kind in {"text", "formula"} and has_blank(block.text)]
        return SegmentationResult(
            question_type=QuestionType.FILL_BLANK,
            question_no=question_no,
            score=score,
            cleaned_blocks=blocks,
            stem_blocks=blocks,
            blank_blocks=blank_blocks,
            figures=figures,
        )

    def _segment_solution(
        self,
        blocks: list[DetectedBlock],
        figures: list[FigureRegion],
        question_no: str,
        score: float | None,
    ) -> SegmentationResult:
        stem_blocks: list[DetectedBlock] = []
        subquestion_groups: list[tuple[str, list[DetectedBlock]]] = []
        current_marker = ""
        current_blocks: list[DetectedBlock] = []

        for block in blocks:
            if block.kind not in {"text", "formula"}:
                if current_marker:
                    current_blocks.append(block)
                else:
                    stem_blocks.append(block)
                continue

            marker = find_subquestion_marker(block.text)
            if marker and marker not in {"I", "V", "X"}:
                if current_marker:
                    subquestion_groups.append((current_marker, current_blocks))
                current_marker = marker
                current_blocks = [block.clone(text=strip_subquestion_marker(block.text))]
                continue

            if current_marker:
                current_blocks.append(block)
            else:
                stem_blocks.append(block)

        if current_marker:
            subquestion_groups.append((current_marker, current_blocks))

        return SegmentationResult(
            question_type=QuestionType.SOLUTION,
            question_no=question_no,
            score=score,
            cleaned_blocks=blocks,
            stem_blocks=stem_blocks,
            subquestion_groups=subquestion_groups,
            figures=figures,
        )

    def estimate_blank_count(self, result: SegmentationResult) -> int:
        return sum(count_blanks(block.text) for block in result.blank_blocks)
