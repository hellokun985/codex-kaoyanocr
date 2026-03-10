from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..data_models import (
    BBox,
    FigureRegion,
    LineCluster,
    OCRBlock,
    OCRBlockType,
    ParsedQuestion,
    QuestionType,
    Region,
    RegionType,
    SubQuestion,
)

QUESTION_NO_RE = re.compile(r"^\s*(?P<no>\d+)\s*[.．、)]?\s*")
SCORE_RE = re.compile(r"[（(]?\s*(?P<score>\d+(?:\.\d+)?)\s*分\s*[)）]?")
SUBQUESTION_RE = re.compile(
    r"^\s*(?P<marker>(?:[（(]\d+[)）])|(?:[IVXivx]{1,5}[.．、:：]?)|(?:\d+[.．、]))\s*(?P<content>.*)$"
)


@dataclass(slots=True)
class _LineEntry:
    line_id: str
    text: str
    bbox: BBox
    block_ids: list[str] = field(default_factory=list)
    blocks: list[OCRBlock] = field(default_factory=list)
    reading_order: int = 0


class SolutionQuestionParser:
    def parse(
        self,
        *,
        ocr_blocks: list[OCRBlock] | None = None,
        line_clusters: list[LineCluster] | None = None,
        regions: list[Region] | None = None,
        lines: list[LineCluster] | None = None,
        figures: list[FigureRegion] | None = None,
    ) -> ParsedQuestion:
        figures = figures or self._collect_figures(regions or [])
        collected_lines = self._collect_lines(
            ocr_blocks=ocr_blocks or [],
            line_clusters=line_clusters or lines or [],
            regions=regions or [],
        )
        if not collected_lines:
            raise ValueError("solution parser 需要至少一种版面输入信息。")

        question_no = ""
        score = None
        stem_regions: list[Region] = []
        layout_regions: list[Region] = []
        subquestions: list[SubQuestion] = []
        current_marker = ""
        current_regions: list[Region] = []

        for index, line in enumerate(collected_lines):
            raw_text = line.text.strip()
            text = raw_text

            if index == 0:
                question_match = QUESTION_NO_RE.match(text)
                question_no, text = self._extract_question_no(text)
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
                score_match = SCORE_RE.search(text)
                score, text = self._extract_score(text, current_score=score)
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
                score_match = SCORE_RE.search(text)
                score, text = self._extract_score(text, current_score=score)
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

            if not text:
                continue

            marker_match = SUBQUESTION_RE.match(text)
            if marker_match:
                if current_marker:
                    subquestions.append(self._to_subquestion(current_marker, current_regions))
                current_marker = marker_match.group("marker").strip()
                content = marker_match.group("content").strip()
                current_regions = []
                if content:
                    region = self._build_content_region(
                        line=line,
                        text=content,
                        region_id=f"sub_{line.line_id}",
                        force_subquestion=True,
                        marker=current_marker,
                    )
                    current_regions.append(region)
                    layout_regions.append(region)
                else:
                    # Preserve empty marker line so downstream can locate the subquestion anchor.
                    region = Region(
                        region_id=f"sub_marker_{line.line_id}",
                        region_type=RegionType.SUBQUESTION,
                        bbox=line.bbox,
                        text="",
                        block_ids=line.block_ids,
                        reading_order=line.reading_order,
                        metadata={"marker": current_marker, "is_marker_only": True},
                    )
                    current_regions.append(region)
                    layout_regions.append(region)
                continue

            region = self._build_content_region(
                line=line,
                text=text,
                region_id=f"{'sub' if current_marker else 'stem'}_{line.line_id}",
                force_subquestion=bool(current_marker),
                marker=current_marker,
            )
            if current_marker:
                current_regions.append(region)
            else:
                stem_regions.append(region)
            layout_regions.append(region)

        if current_marker:
            subquestions.append(self._to_subquestion(current_marker, current_regions))

        layout_regions.extend(figures)
        layout_regions.sort(key=lambda region: (region.reading_order, region.bbox.y1, region.bbox.x1, region.region_id))

        return ParsedQuestion(
            question_no=question_no,
            score=score,
            question_type=QuestionType.SOLUTION,
            has_figure=bool(figures),
            figures=figures,
            stem_segments=stem_regions,
            stem="\n".join(region.text for region in stem_regions if region.text).strip(),
            options=[],
            blanks=[],
            subquestions=subquestions,
            layout_regions=layout_regions,
        )

    def _build_content_region(
        self,
        *,
        line: _LineEntry,
        text: str,
        region_id: str,
        force_subquestion: bool,
        marker: str = "",
    ) -> Region:
        region_type = RegionType.SUBQUESTION if force_subquestion else self._infer_region_type(line)
        metadata = {"marker": marker} if marker else {}
        if any(block.block_type is OCRBlockType.FORMULA for block in line.blocks):
            metadata["has_formula_block"] = True
        if self._looks_formula_like(text):
            metadata["looks_formula_like"] = True
        return Region(
            region_id=region_id,
            region_type=region_type,
            bbox=line.bbox,
            text=text,
            block_ids=line.block_ids,
            reading_order=line.reading_order,
            metadata=metadata,
        )

    def _infer_region_type(self, line: _LineEntry) -> RegionType:
        if any(block.block_type is OCRBlockType.FORMULA for block in line.blocks):
            return RegionType.STEM_FORMULA
        if self._looks_formula_like(line.text):
            return RegionType.STEM_FORMULA
        return RegionType.STEM_TEXT

    def _looks_formula_like(self, text: str) -> bool:
        formula_chars = set("=+-*/^_<>[]{}()\\|")
        hit_count = sum(1 for char in text if char in formula_chars)
        return hit_count >= 4 or "矩阵" in text or "\\begin" in text

    def _to_subquestion(self, marker: str, regions: list[Region]) -> SubQuestion:
        normalized_regions = [region for region in regions if region.text or not region.metadata.get("is_marker_only")]
        effective_regions = normalized_regions or regions
        return SubQuestion(
            marker=marker,
            stem="\n".join(region.text for region in effective_regions if region.text).strip(),
            stem_segments=effective_regions,
            layout_regions=effective_regions,
        )

    def _extract_question_no(self, text: str) -> tuple[str, str]:
        match = QUESTION_NO_RE.match(text)
        if not match:
            return "", text
        return match.group("no"), text[match.end() :].strip()

    def _extract_score(self, text: str, *, current_score: float | None) -> tuple[float | None, str]:
        if current_score is not None:
            return current_score, text
        match = SCORE_RE.search(text)
        if not match:
            return None, text
        return float(match.group("score")), SCORE_RE.sub("", text, count=1).strip()

    def _estimate_span_bbox(self, line_bbox: BBox, full_text: str, start: int, end: int) -> BBox:
        total = max(1, len(full_text))
        x1 = line_bbox.x1 + int(line_bbox.width * (start / total))
        x2 = line_bbox.x1 + int(line_bbox.width * (end / total))
        return BBox(max(line_bbox.x1, x1), line_bbox.y1, max(x1 + 1, x2), line_bbox.y2)

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

        if not ocr_blocks:
            return []
        return self._cluster_blocks_to_lines(ocr_blocks)

    def _cluster_blocks_to_lines(self, ocr_blocks: list[OCRBlock]) -> list[_LineEntry]:
        grouped: dict[str, list[OCRBlock]] = {}
        if all(block.line_id for block in ocr_blocks):
            for block in ocr_blocks:
                grouped.setdefault(block.line_id, []).append(block)
        else:
            sorted_blocks = sorted(ocr_blocks, key=lambda block: (block.bbox.y1, block.bbox.x1, block.block_id))
            avg_height = sum(max(1, block.bbox.height) for block in sorted_blocks) / len(sorted_blocks)
            threshold = max(10, int(avg_height * 0.6))
            current_center = None
            current_line_id = "line_1"
            for block in sorted_blocks:
                center_y = (block.bbox.y1 + block.bbox.y2) / 2
                if current_center is None or abs(center_y - current_center) <= threshold:
                    grouped.setdefault(current_line_id, []).append(block)
                    current_center = center_y if current_center is None else (current_center + center_y) / 2
                else:
                    current_line_id = f"line_{len(grouped) + 1}"
                    grouped[current_line_id] = [block]
                    current_center = center_y

        lines: list[_LineEntry] = []
        for line_id, blocks in grouped.items():
            sorted_blocks = sorted(blocks, key=lambda block: (block.bbox.x1, block.block_id))
            lines.append(
                _LineEntry(
                    line_id=line_id,
                    text=" ".join(block.text for block in sorted_blocks if block.text).strip(),
                    bbox=BBox(
                        x1=min(block.bbox.x1 for block in sorted_blocks),
                        y1=min(block.bbox.y1 for block in sorted_blocks),
                        x2=max(block.bbox.x2 for block in sorted_blocks),
                        y2=max(block.bbox.y2 for block in sorted_blocks),
                    ),
                    block_ids=[block.block_id for block in sorted_blocks],
                    blocks=sorted_blocks,
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
