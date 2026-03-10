from __future__ import annotations

import re

from ..data_models import BBox, BlankRegion, FigureRegion, LineCluster, OCRBlock, ParsedQuestion, QuestionType, Region, RegionType

QUESTION_NO_RE = re.compile(r"^\s*(?P<no>\d+)\s*[.．、)]?\s*")
SCORE_RE = re.compile(r"[（(]?\s*(?P<score>\d+(?:\.\d+)?)\s*分\s*[)）]?")
BLANK_RE = re.compile(r"(_{2,}|﹍{2,}|—{2,}|-{2,}|（\s*）|\(\s*\)|\[+\s*\]+|□+|◻+)")


class FillBlankQuestionParser:
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
            raise ValueError("fill blank parser 需要至少一种版面输入信息。")

        question_no = ""
        score = None
        stem_regions: list[Region] = []
        blanks: list[BlankRegion] = []
        layout_regions: list[Region] = []
        blank_index = 1

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
                            block_ids=[block.block_id for block in line.blocks],
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
                            block_ids=[block.block_id for block in line.blocks],
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
                            block_ids=[block.block_id for block in line.blocks],
                            reading_order=line.reading_order,
                        )
                    )

            if not text:
                continue

            stem_region = Region(
                region_id=f"stem_{line.line_id}",
                region_type=RegionType.STEM_TEXT,
                bbox=line.bbox,
                text=text,
                block_ids=[block.block_id for block in line.blocks],
                reading_order=line.reading_order,
            )
            stem_regions.append(stem_region)
            layout_regions.append(stem_region)

            for match in BLANK_RE.finditer(text):
                blanks.append(
                    BlankRegion(
                        region_id=f"blank_{blank_index}",
                        region_type=RegionType.BLANK,
                        bbox=self._estimate_span_bbox(line.bbox, text, match.start(), match.end()),
                        text=match.group(0),
                        block_ids=[block.block_id for block in line.blocks],
                        reading_order=line.reading_order,
                        blank_index=blank_index,
                        placeholder=match.group(0),
                        blank_length=self._estimate_blank_length(match.group(0)),
                        char_start=match.start(),
                        char_end=match.end(),
                        metadata={"line_id": line.line_id},
                    )
                )
                blank_index += 1

        layout_regions.extend(blanks)
        layout_regions.extend(figures)
        layout_regions.sort(key=lambda region: (region.reading_order, region.bbox.y1, region.bbox.x1))

        return ParsedQuestion(
            question_no=question_no,
            score=score,
            question_type=QuestionType.FILL_BLANK,
            has_figure=bool(figures),
            figures=figures,
            stem_segments=stem_regions,
            stem="\n".join(region.text for region in stem_regions if region.text).strip(),
            options=[],
            blanks=blanks,
            subquestions=[],
            layout_regions=layout_regions,
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

    def _estimate_blank_length(self, placeholder: str) -> int:
        stripped = placeholder.strip()
        if not stripped:
            return 0
        if stripped in {"()", "（）"}:
            return 1
        return len(stripped)

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
    ) -> list[LineCluster]:
        if line_clusters:
            return sorted(
                [line for line in line_clusters if line.text.strip()],
                key=lambda line: (line.reading_order if line.reading_order else 10**9, line.bbox.y1, line.bbox.x1),
            )

        text_regions = [region for region in regions if region.region_type is not RegionType.FIGURE and region.text.strip()]
        if text_regions:
            return [
                LineCluster(
                    line_id=region.region_id,
                    bbox=region.bbox,
                    blocks=[],
                    text=region.text.strip(),
                    reading_order=region.reading_order,
                )
                for region in sorted(text_regions, key=lambda item: (item.reading_order, item.bbox.y1, item.bbox.x1))
            ]

        if not ocr_blocks:
            return []

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

        clusters: list[LineCluster] = []
        for reading_order, (line_id, blocks) in enumerate(
            sorted(grouped.items(), key=lambda item: (min(block.bbox.y1 for block in item[1]), min(block.bbox.x1 for block in item[1]))),
            start=1,
        ):
            sorted_blocks = sorted(blocks, key=lambda block: (block.bbox.x1, block.block_id))
            clusters.append(
                LineCluster(
                    line_id=line_id,
                    bbox=BBox(
                        x1=min(block.bbox.x1 for block in sorted_blocks),
                        y1=min(block.bbox.y1 for block in sorted_blocks),
                        x2=max(block.bbox.x2 for block in sorted_blocks),
                        y2=max(block.bbox.y2 for block in sorted_blocks),
                    ),
                    blocks=sorted_blocks,
                    text=" ".join(block.text for block in sorted_blocks if block.text).strip(),
                    reading_order=reading_order,
                )
            )
        return clusters
