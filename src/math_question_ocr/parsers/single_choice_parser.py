from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..data_models import (
    BBox,
    FigureRegion,
    LineCluster,
    OCRBlock,
    ParsedOption,
    ParsedQuestion,
    QuestionType,
    Region,
    RegionType,
)

QUESTION_NO_RE = re.compile(r"^\s*(?P<no>\d+)\s*[.．、)]?\s*")
SCORE_RE = re.compile(r"[（(]?\s*(?P<score>\d+(?:\.\d+)?)\s*分\s*[)）]?")
OPTION_ANCHOR_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s　]))(?P<label>[ABCDabcd480OoG])\s*(?P<punc>[.．、:：\)）])"
)

LABEL_ALIASES = {
    "A": "A",
    "a": "A",
    "4": "A",
    "B": "B",
    "b": "B",
    "8": "B",
    "C": "C",
    "c": "C",
    "G": "C",
    "D": "D",
    "d": "D",
    "O": "D",
    "o": "D",
    "0": "D",
}

OPTION_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}


@dataclass(slots=True)
class _LineEntry:
    line_id: str
    text: str
    bbox: BBox
    block_ids: list[str] = field(default_factory=list)
    blocks: list[OCRBlock] = field(default_factory=list)
    reading_order: int = 0


@dataclass(slots=True)
class _OptionSlice:
    label: str
    text: str
    bbox: BBox
    block_ids: list[str]
    line_id: str
    reading_order: int


class SingleChoiceQuestionParser:
    def parse(
        self,
        *,
        ocr_blocks: list[OCRBlock] | None = None,
        line_clusters: list[LineCluster] | None = None,
        regions: list[Region] | None = None,
    ) -> ParsedQuestion:
        figures = self._collect_figures(regions or [])
        lines = self._collect_lines(
            ocr_blocks=ocr_blocks or [],
            line_clusters=line_clusters or [],
            regions=regions or [],
        )
        if not lines:
            raise ValueError("single choice parser 需要至少一种版面输入信息。")

        question_no = ""
        score = None
        stem_regions: list[Region] = []
        option_map: dict[str, list[Region]] = {}
        layout_regions: list[Region] = []
        option_started = False
        last_option_label = ""

        for index, line in enumerate(lines):
            raw_text = line.text.strip()
            working_text = raw_text
            if index == 0:
                question_match = QUESTION_NO_RE.match(working_text)
                question_no, working_text = self._extract_question_no(working_text)
                if question_no and question_match:
                    layout_regions.append(
                        Region(
                            region_id=f"question_no_{line.line_id}",
                            region_type=RegionType.QUESTION_NO,
                            bbox=self._estimate_span_bbox(line.bbox, raw_text, question_match.start("no"), question_match.end("no")),
                            text=question_no,
                            block_ids=line.block_ids,
                            reading_order=line.reading_order,
                        )
                    )
                score_match = SCORE_RE.search(working_text)
                score, working_text = self._extract_score(working_text, current_score=score)
                if score_match:
                    layout_regions.append(
                        Region(
                            region_id=f"score_{line.line_id}",
                            region_type=RegionType.SCORE,
                            bbox=self._estimate_span_bbox(line.bbox, raw_text, score_match.start("score"), score_match.end("score")),
                            text=score_match.group("score"),
                            block_ids=line.block_ids,
                            reading_order=line.reading_order,
                        )
                    )
            else:
                score_match = SCORE_RE.search(working_text)
                score, working_text = self._extract_score(working_text, current_score=score)
                if score_match:
                    layout_regions.append(
                        Region(
                            region_id=f"score_{line.line_id}",
                            region_type=RegionType.SCORE,
                            bbox=self._estimate_span_bbox(line.bbox, raw_text, score_match.start("score"), score_match.end("score")),
                            text=score_match.group("score"),
                            block_ids=line.block_ids,
                            reading_order=line.reading_order,
                        )
                    )

            if not working_text:
                continue

            sliced_options = self._split_option_line(line, working_text)
            if sliced_options:
                option_started = True
                for option_slice in sliced_options:
                    region = Region(
                        region_id=f"option_{option_slice.label}_{option_slice.line_id}_{option_slice.reading_order}",
                        region_type=RegionType.OPTION,
                        bbox=option_slice.bbox,
                        text=option_slice.text,
                        block_ids=option_slice.block_ids,
                        reading_order=option_slice.reading_order,
                        metadata={"label": option_slice.label},
                    )
                    option_map.setdefault(option_slice.label, []).append(region)
                    layout_regions.append(region)
                    last_option_label = option_slice.label
                continue

            region_type = RegionType.STEM_TEXT
            target_list = stem_regions
            if option_started and last_option_label:
                region_type = RegionType.OPTION
                target_list = option_map.setdefault(last_option_label, [])

            region = Region(
                region_id=f"{region_type.value}_{line.line_id}",
                region_type=region_type,
                bbox=line.bbox,
                text=working_text,
                block_ids=line.block_ids,
                reading_order=line.reading_order,
                metadata={"label": last_option_label} if region_type is RegionType.OPTION else {},
            )
            target_list.append(region)
            layout_regions.append(region)

        parsed_options = self._build_parsed_options(option_map)
        if not parsed_options:
            raise ValueError("未找到足够稳定的 A/B/C/D 选项锚点。")

        layout_regions.extend(figures)
        layout_regions.sort(key=lambda region: (region.reading_order, region.bbox.y1, region.bbox.x1, region.region_id))

        return ParsedQuestion(
            question_no=question_no,
            score=score,
            question_type=QuestionType.SINGLE_CHOICE,
            has_figure=bool(figures),
            figures=figures,
            stem_segments=stem_regions,
            stem="\n".join(region.text for region in stem_regions if region.text).strip(),
            options=parsed_options,
            blanks=[],
            subquestions=[],
            layout_regions=layout_regions,
        )

    def _collect_figures(self, regions: list[Region]) -> list[FigureRegion]:
        figures: list[FigureRegion] = []
        for region in regions:
            if isinstance(region, FigureRegion):
                figures.append(region)
            elif region.region_type is RegionType.FIGURE:
                figures.append(
                    FigureRegion(
                        region_id=region.region_id,
                        region_type=RegionType.FIGURE,
                        bbox=region.bbox,
                        text=region.text,
                        block_ids=list(region.block_ids),
                        reading_order=region.reading_order,
                        metadata=dict(region.metadata),
                    )
                )
        figures.sort(key=lambda item: (item.reading_order, item.bbox.y1, item.bbox.x1))
        return figures

    def _collect_lines(
        self,
        *,
        ocr_blocks: list[OCRBlock],
        line_clusters: list[LineCluster],
        regions: list[Region],
    ) -> list[_LineEntry]:
        if line_clusters:
            lines = [
                _LineEntry(
                    line_id=line.line_id,
                    text=line.text.strip(),
                    bbox=line.bbox,
                    block_ids=[block.block_id for block in line.blocks],
                    blocks=list(line.blocks),
                    reading_order=line.reading_order,
                )
                for line in line_clusters
                if line.text.strip()
            ]
            return self._sort_lines(lines)

        text_regions = [region for region in regions if region.region_type is not RegionType.FIGURE and region.text.strip()]
        if text_regions:
            lines = [
                _LineEntry(
                    line_id=region.region_id,
                    text=region.text.strip(),
                    bbox=region.bbox,
                    block_ids=list(region.block_ids),
                    blocks=[],
                    reading_order=region.reading_order,
                )
                for region in text_regions
            ]
            return self._sort_lines(lines)

        if ocr_blocks:
            return self._cluster_blocks_to_lines(ocr_blocks)
        return []

    def _cluster_blocks_to_lines(self, ocr_blocks: list[OCRBlock]) -> list[_LineEntry]:
        grouped: dict[str, list[OCRBlock]] = {}
        if all(block.line_id for block in ocr_blocks):
            for block in ocr_blocks:
                grouped.setdefault(block.line_id, []).append(block)
        else:
            sorted_blocks = sorted(ocr_blocks, key=lambda block: (block.bbox.y1, block.bbox.x1, block.block_id))
            threshold = max(10, int(sum(block.bbox.height or 1 for block in sorted_blocks) / len(sorted_blocks) * 0.6))
            line_index = 1
            current_key = f"line_{line_index}"
            current_center = None
            for block in sorted_blocks:
                block_center = (block.bbox.y1 + block.bbox.y2) / 2
                if current_center is None or abs(block_center - current_center) <= threshold:
                    grouped.setdefault(current_key, []).append(block)
                    current_center = block_center if current_center is None else (current_center + block_center) / 2
                else:
                    line_index += 1
                    current_key = f"line_{line_index}"
                    grouped[current_key] = [block]
                    current_center = block_center

        lines: list[_LineEntry] = []
        for line_id, blocks in grouped.items():
            sorted_line_blocks = sorted(blocks, key=lambda block: (block.bbox.x1, block.block_id))
            bbox = self._merge_bbox(block.bbox for block in sorted_line_blocks)
            lines.append(
                _LineEntry(
                    line_id=line_id,
                    text=" ".join(block.text for block in sorted_line_blocks if block.text).strip(),
                    bbox=bbox,
                    block_ids=[block.block_id for block in sorted_line_blocks],
                    blocks=sorted_line_blocks,
                    reading_order=0,
                )
            )
        return self._sort_lines(lines)

    def _sort_lines(self, lines: list[_LineEntry]) -> list[_LineEntry]:
        sorted_lines = sorted(
            lines,
            key=lambda line: (
                line.reading_order if line.reading_order else 10**9,
                line.bbox.y1,
                line.bbox.x1,
                line.line_id,
            ),
        )
        for index, line in enumerate(sorted_lines, start=1):
            if line.reading_order == 0:
                line.reading_order = index
        return sorted_lines

    def _extract_question_no(self, text: str) -> tuple[str, str]:
        match = QUESTION_NO_RE.match(text)
        if not match:
            return "", text
        question_no = match.group("no")
        return question_no, text[match.end() :].strip()

    def _extract_score(self, text: str, *, current_score: float | None) -> tuple[float | None, str]:
        if current_score is not None:
            return current_score, text
        match = SCORE_RE.search(text)
        if not match:
            return None, text
        score = float(match.group("score"))
        cleaned = SCORE_RE.sub("", text, count=1).strip()
        return score, cleaned

    def _split_option_line(self, line: _LineEntry, text: str) -> list[_OptionSlice]:
        matches = list(OPTION_ANCHOR_RE.finditer(text))
        if not matches:
            return []

        slices: list[_OptionSlice] = []
        for index, match in enumerate(matches):
            label = self._normalize_label(match.group("label"))
            if not label:
                continue
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            option_text = text[start:end].strip()
            if not option_text:
                continue
            bbox = self._estimate_span_bbox(line.bbox, text, match.start(), end)
            slices.append(
                _OptionSlice(
                    label=label,
                    text=option_text,
                    bbox=bbox,
                    block_ids=list(line.block_ids),
                    line_id=line.line_id,
                    reading_order=line.reading_order,
                )
            )
        return self._dedupe_and_sort_slices(slices)

    def _dedupe_and_sort_slices(self, slices: list[_OptionSlice]) -> list[_OptionSlice]:
        seen: set[str] = set()
        deduped: list[_OptionSlice] = []
        for item in slices:
            key = f"{item.label}:{item.text}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return sorted(deduped, key=lambda item: (OPTION_ORDER.get(item.label, 99), item.bbox.x1))

    def _build_parsed_options(self, option_map: dict[str, list[Region]]) -> list[ParsedOption]:
        parsed_options: list[ParsedOption] = []
        for label in ["A", "B", "C", "D"]:
            regions = option_map.get(label, [])
            if not regions:
                continue
            regions.sort(key=lambda region: (region.reading_order, region.bbox.y1, region.bbox.x1))
            parsed_options.append(
                ParsedOption(
                    label=label,
                    text="\n".join(region.text for region in regions if region.text).strip(),
                    regions=regions,
                )
            )
        return parsed_options

    def _normalize_label(self, raw_label: str) -> str:
        return LABEL_ALIASES.get(raw_label, "")

    def _merge_bbox(self, boxes) -> BBox:
        items = list(boxes)
        return BBox(
            x1=min(box.x1 for box in items),
            y1=min(box.y1 for box in items),
            x2=max(box.x2 for box in items),
            y2=max(box.y2 for box in items),
        )

    def _estimate_span_bbox(self, line_bbox: BBox, full_text: str, start: int, end: int) -> BBox:
        total = max(1, len(full_text))
        x1 = line_bbox.x1 + int(line_bbox.width * (start / total))
        x2 = line_bbox.x1 + int(line_bbox.width * (end / total))
        return BBox(max(line_bbox.x1, x1), line_bbox.y1, max(x1 + 1, x2), line_bbox.y2)
