from __future__ import annotations

from typing import Iterable

from .schemas import DetectedBlock, StemSegment


def sort_blocks(blocks: Iterable[DetectedBlock]) -> list[DetectedBlock]:
    return sorted(blocks, key=lambda block: (block.bbox.y1, block.bbox.x1, block.block_id))


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def merge_block_texts(blocks: Iterable[DetectedBlock]) -> str:
    texts = [normalize_text(block.text) for block in blocks if normalize_text(block.text)]
    return "\n".join(texts)


def blocks_to_segments(blocks: Iterable[DetectedBlock], *, start_order: int = 0) -> list[StemSegment]:
    segments: list[StemSegment] = []
    for index, block in enumerate(blocks, start=start_order):
        content = normalize_text(block.text)
        if not content and block.kind != "figure":
            continue
        segments.append(
            StemSegment(
                segment_id=block.block_id,
                kind=block.kind,
                content=content,
                bbox=block.bbox,
                order=index,
            )
        )
    return segments
