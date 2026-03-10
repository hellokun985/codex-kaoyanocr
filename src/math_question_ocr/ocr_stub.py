from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from .data_models import BBox, FigureRegion, OCRBlock, OCRBlockType, RegionType


class OCREngine(ABC):
    @abstractmethod
    def detect(self, image_path: str) -> list[OCRBlock]:
        raise NotImplementedError


class FormulaRecognizer(ABC):
    @abstractmethod
    def recognize(self, image_path: str, ocr_blocks: list[OCRBlock]) -> list[OCRBlock]:
        raise NotImplementedError


class FigureDetector(ABC):
    @abstractmethod
    def detect(self, image_path: str, ocr_blocks: list[OCRBlock]) -> list[FigureRegion]:
        raise NotImplementedError


class SidecarJsonOCREngine(OCREngine):
    def __init__(self, *, ocr_json_path: str | None = None) -> None:
        self.ocr_json_path = ocr_json_path

    def detect(self, image_path: str) -> list[OCRBlock]:
        path = self._resolve_json_path(image_path)
        if path is None or not path.exists():
            return []

        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("blocks", payload if isinstance(payload, list) else [])
        blocks: list[OCRBlock] = []
        for index, item in enumerate(items, start=1):
            bbox_raw = item["bbox"]
            block_type = OCRBlockType(item.get("block_type", "text"))
            blocks.append(
                OCRBlock(
                    block_id=item.get("block_id", f"b{index}"),
                    text=item.get("text", ""),
                    bbox=BBox(
                        x1=int(bbox_raw["x1"]),
                        y1=int(bbox_raw["y1"]),
                        x2=int(bbox_raw["x2"]),
                        y2=int(bbox_raw["y2"]),
                    ),
                    confidence=float(item.get("confidence", 1.0)),
                    block_type=block_type,
                    line_id=item.get("line_id", ""),
                    source=item.get("source", "sidecar_json"),
                    metadata=item.get("metadata", {}),
                )
            )
        return blocks

    def _resolve_json_path(self, image_path: str) -> Path | None:
        if self.ocr_json_path:
            return Path(self.ocr_json_path)
        image = Path(image_path)
        candidates = [
            image.with_suffix(image.suffix + ".ocr.json"),
            image.with_suffix(".ocr.json"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]


class PlaceholderFormulaRecognizer(FormulaRecognizer):
    def recognize(self, image_path: str, ocr_blocks: list[OCRBlock]) -> list[OCRBlock]:
        return []


class PlaceholderFigureDetector(FigureDetector):
    def __init__(self, *, figure_json_path: str | None = None) -> None:
        self.figure_json_path = figure_json_path

    def detect(self, image_path: str, ocr_blocks: list[OCRBlock]) -> list[FigureRegion]:
        path = self._resolve_json_path(image_path)
        if path is None or not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("figures", payload if isinstance(payload, list) else [])
        figures: list[FigureRegion] = []
        for index, item in enumerate(items, start=1):
            bbox_raw = item["bbox"]
            figures.append(
                FigureRegion(
                    region_id=item.get("region_id", f"fig_{index}"),
                    region_type=RegionType.FIGURE,
                    bbox=BBox(
                        x1=int(bbox_raw["x1"]),
                        y1=int(bbox_raw["y1"]),
                        x2=int(bbox_raw["x2"]),
                        y2=int(bbox_raw["y2"]),
                    ),
                    text=item.get("text", ""),
                    block_ids=item.get("block_ids", []),
                    reading_order=int(item.get("reading_order", index)),
                    metadata=item.get("metadata", {}),
                    figure_type=item.get("figure_type", "unknown"),
                    caption=item.get("caption", ""),
                )
            )
        return figures

    def _resolve_json_path(self, image_path: str) -> Path | None:
        if self.figure_json_path:
            return Path(self.figure_json_path)
        image = Path(image_path)
        candidates = [
            image.with_suffix(image.suffix + ".figures.json"),
            image.with_suffix(".figures.json"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None
