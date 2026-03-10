from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from .schemas import DetectedBlock, QuestionDocument, SegmentationResult

COLOR_MAP = {
    "text": "#1f77b4",
    "formula": "#d62728",
    "figure": "#2ca02c",
}


class DebugVisualizer:
    def dump(
        self,
        output_dir: str | Path,
        blocks: list[DetectedBlock],
        segmentation: SegmentationResult,
        document: QuestionDocument,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        (output_path / "debug_layout.json").write_text(
            json.dumps(
                {
                    "blocks": [block.to_dict() for block in blocks],
                    "segmentation": segmentation.to_dict(),
                    "document": document.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        (output_path / "debug_layout.svg").write_text(
            self._build_svg(blocks, segmentation),
            encoding="utf-8",
        )

    def _build_svg(self, blocks: list[DetectedBlock], segmentation: SegmentationResult) -> str:
        max_x = 1200
        max_y = 1600
        if blocks:
            max_x = max(block.bbox.x2 for block in blocks) + 40
            max_y = max(block.bbox.y2 for block in blocks) + 40

        rects: list[str] = []
        for block in blocks:
            color = COLOR_MAP.get(block.kind, "#7f7f7f")
            text = escape(f"{block.kind}: {block.text[:30]}")
            rects.append(
                f'<rect x="{block.bbox.x1}" y="{block.bbox.y1}" width="{block.bbox.width}" '
                f'height="{block.bbox.height}" fill="none" stroke="{color}" stroke-width="2" />'
            )
            rects.append(
                f'<text x="{block.bbox.x1}" y="{max(12, block.bbox.y1 - 4)}" '
                f'font-size="12" fill="{color}">{text}</text>'
            )

        for label, option_blocks in segmentation.option_groups:
            if not option_blocks:
                continue
            bbox = option_blocks[0].bbox
            rects.append(
                f'<text x="{bbox.x1}" y="{bbox.y2 + 14}" font-size="12" fill="#9467bd">option:{label}</text>'
            )

        for marker, sub_blocks in segmentation.subquestion_groups:
            if not sub_blocks:
                continue
            bbox = sub_blocks[0].bbox
            rects.append(
                f'<text x="{bbox.x1}" y="{bbox.y2 + 14}" font-size="12" fill="#ff7f0e">sub:{escape(marker)}</text>'
            )

        for figure in segmentation.figures:
            rects.append(
                f'<rect x="{figure.bbox.x1}" y="{figure.bbox.y1}" width="{figure.bbox.width}" '
                f'height="{figure.bbox.height}" fill="none" stroke="#2ca02c" stroke-dasharray="6,4" '
                f'stroke-width="3" />'
            )
            rects.append(
                f'<text x="{figure.bbox.x1}" y="{max(12, figure.bbox.y1 - 6)}" '
                f'font-size="12" fill="#2ca02c">{escape(figure.figure_id)}</text>'
            )

        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{max_x}" height="{max_y}" '
            f'viewBox="0 0 {max_x} {max_y}">'
            f'<rect x="0" y="0" width="{max_x}" height="{max_y}" fill="#ffffff" />'
            + "".join(rects)
            + "</svg>"
        )
