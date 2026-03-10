from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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


class PaddleOCREngine(OCREngine):
    def __init__(
        self,
        *,
        lang: str = "ch",
        use_angle_cls: bool = True,
        det: bool = True,
        rec: bool = True,
        **kwargs: Any,
    ) -> None:
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.det = det
        self.rec = rec
        self.extra_kwargs = kwargs
        self._ocr: Any | None = None

    def detect(self, image_path: str) -> list[OCRBlock]:
        ocr = self._get_ocr_instance()
        result = self._run_ocr(ocr, image_path)
        entries = self._normalize_result(result)

        blocks: list[OCRBlock] = []
        for index, entry in enumerate(entries, start=1):
            polygon, text, confidence = self._split_entry(entry)
            if not polygon:
                continue

            bbox = self._polygon_to_bbox(polygon)
            blocks.append(
                OCRBlock(
                    block_id=f"paddle_{index}",
                    text=text,
                    bbox=bbox,
                    confidence=confidence,
                    block_type=OCRBlockType.TEXT,
                    line_id="",
                    source="paddleocr",
                    metadata={"polygon": polygon},
                )
            )
        return blocks

    def _run_ocr(self, ocr: Any, image_path: str) -> Any:
        try:
            return ocr.ocr(image_path, cls=self.use_angle_cls)
        except TypeError as exc:
            if "unexpected keyword argument 'cls'" not in str(exc):
                raise
        return ocr.ocr(image_path)

    def _get_ocr_instance(self) -> Any:
        if self._ocr is not None:
            return self._ocr

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "未安装 PaddleOCR 依赖，无法使用 paddle OCR backend。"
                "请先安装 paddleocr 和 paddlepaddle，再使用 --ocr-backend paddle。"
            ) from exc

        init_error_messages: list[str] = []
        for kwargs in self._build_init_candidates(PaddleOCR):
            try:
                self._ocr = PaddleOCR(**kwargs)
                return self._ocr
            except (TypeError, ValueError) as exc:
                init_error_messages.append(f"{kwargs!r} -> {exc}")

        details = " | ".join(init_error_messages) if init_error_messages else "无可用初始化参数组合"
        raise RuntimeError(
            "当前已安装的 PaddleOCR 版本与适配器初始化参数不兼容。"
            f"尝试过的参数组合: {details}"
        )

    def _build_init_candidates(self, paddle_ocr_cls: Any) -> list[dict[str, Any]]:
        signature = self._get_init_signature(paddle_ocr_cls)

        candidates = [
            self._filter_supported_kwargs(
                signature,
                {
                    "lang": self.lang,
                    "use_angle_cls": self.use_angle_cls,
                    "det": self.det,
                    "rec": self.rec,
                    **self.extra_kwargs,
                },
            ),
            self._filter_supported_kwargs(
                signature,
                {
                    "lang": self.lang,
                    "use_angle_cls": self.use_angle_cls,
                },
            ),
            self._filter_supported_kwargs(
                signature,
                {
                    "lang": self.lang,
                    "det": self.det,
                    "rec": self.rec,
                },
            ),
            self._filter_supported_kwargs(
                signature,
                {
                    "lang": self.lang,
                },
            ),
            {},
        ]

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, Any], ...]] = set()
        for candidate in candidates:
            key = tuple(sorted(candidate.items()))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _get_init_signature(self, paddle_ocr_cls: Any) -> inspect.Signature | None:
        try:
            return inspect.signature(paddle_ocr_cls.__init__)
        except (TypeError, ValueError):
            return None

    def _filter_supported_kwargs(
        self,
        signature: inspect.Signature | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if signature is None:
            return dict(kwargs)

        supported = {
            name
            for name, param in signature.parameters.items()
            if name != "self" and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return {key: value for key, value in kwargs.items() if key in supported}

    def _normalize_result(self, result: Any) -> list[Any]:
        if not result:
            return []

        normalized: list[Any] = []

        if isinstance(result, list) and result and isinstance(result[0], dict):
            for page in result:
                normalized.extend(self._extract_entries_from_page_dict(page))
            return normalized

        if isinstance(result, list):
            if result and isinstance(result[0], list):
                first = result[0]
                if first and self._looks_like_legacy_entry(first[0]):
                    return first
                if self._looks_like_legacy_entry(first):
                    return result
            if result and self._looks_like_legacy_entry(result[0]):
                return result

        return normalized

    def _extract_entries_from_page_dict(self, page: dict[str, Any]) -> list[dict[str, Any]]:
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        polygons = page.get("rec_polys") or page.get("dt_polys") or []

        size = min(len(texts), len(polygons))
        if scores:
            size = min(size, len(scores))

        entries: list[dict[str, Any]] = []
        for index in range(size):
            entries.append(
                {
                    "polygon": self._to_polygon(polygons[index]),
                    "text": str(texts[index] or ""),
                    "confidence": float(scores[index]) if index < len(scores) else 1.0,
                }
            )
        return entries

    def _looks_like_legacy_entry(self, value: Any) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return False
        polygon = value[0]
        return self._is_polygon_like(polygon)

    def _split_entry(self, entry: Any) -> tuple[list[list[float]], str, float]:
        if isinstance(entry, dict):
            polygon = self._to_polygon(entry.get("polygon"))
            text = str(entry.get("text", "") or "")
            try:
                confidence = float(entry.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            return polygon, text, confidence

        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            return [], "", 1.0

        polygon = self._to_polygon(entry[0])
        rec = entry[1]

        text = ""
        confidence = 1.0
        if isinstance(rec, (list, tuple)) and len(rec) >= 1:
            text = str(rec[0] or "")
            if len(rec) >= 2:
                try:
                    confidence = float(rec[1])
                except (TypeError, ValueError):
                    confidence = 1.0
        elif rec is not None:
            text = str(rec)

        return polygon, text, confidence

    def _is_polygon_like(self, value: Any) -> bool:
        try:
            points = value.tolist() if hasattr(value, "tolist") else value
        except Exception:
            points = value
        return isinstance(points, (list, tuple)) and len(points) >= 4

    def _to_polygon(self, value: Any) -> list[list[float]]:
        try:
            points = value.tolist() if hasattr(value, "tolist") else value
        except Exception:
            points = value

        if isinstance(points, (list, tuple)) and len(points) == 4 and all(
            not isinstance(item, (list, tuple)) for item in points
        ):
            try:
                x1, y1, x2, y2 = [float(item) for item in points]
            except (TypeError, ValueError):
                return []
            return [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ]

        normalized_polygon: list[list[float]] = []
        if isinstance(points, (list, tuple)):
            for point in points:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    try:
                        normalized_polygon.append([float(point[0]), float(point[1])])
                    except (TypeError, ValueError):
                        continue
        return normalized_polygon

    def _polygon_to_bbox(self, polygon: list[list[float]]) -> BBox:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        return BBox(
            x1=int(min(xs)),
            y1=int(min(ys)),
            x2=int(max(xs)),
            y2=int(max(ys)),
        )


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
