from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .schemas import BBox, DetectedBlock, FigureRegion


class OCREngine(ABC):
    @abstractmethod
    def detect(self, image_path: str) -> list[DetectedBlock]:
        raise NotImplementedError


class FormulaRecognizer(ABC):
    @abstractmethod
    def recognize(self, image_path: str, text_blocks: list[DetectedBlock]) -> list[DetectedBlock]:
        raise NotImplementedError


class FigureDetector(ABC):
    @abstractmethod
    def detect(self, image_path: str, text_blocks: list[DetectedBlock]) -> list[FigureRegion]:
        raise NotImplementedError


class PaddleOCRAdapter(OCREngine):
    def __init__(self, *, lang: str = "ch", use_angle_cls: bool = True) -> None:
        self.lang = lang
        self.use_angle_cls = use_angle_cls

    def detect(self, image_path: str) -> list[DetectedBlock]:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PaddleOCR 未安装，无法执行真实 OCR。") from exc

        ocr = PaddleOCR(lang=self.lang, use_angle_cls=self.use_angle_cls)
        result = ocr.ocr(image_path, cls=self.use_angle_cls)
        blocks: list[DetectedBlock] = []
        for index, line in enumerate(result[0] if result else []):
            points, (text, score) = line
            xs = [int(point[0]) for point in points]
            ys = [int(point[1]) for point in points]
            blocks.append(
                DetectedBlock(
                    block_id=f"text_{index + 1}",
                    kind="text",
                    text=text,
                    bbox=BBox(min(xs), min(ys), max(xs), max(ys)),
                    score=float(score),
                    source="paddleocr",
                )
            )
        return blocks


class NullFormulaRecognizer(FormulaRecognizer):
    def recognize(self, image_path: str, text_blocks: list[DetectedBlock]) -> list[DetectedBlock]:
        return []


class PlaceholderFormulaRecognizer(FormulaRecognizer):
    def __init__(self, backend: str = "pix2tex") -> None:
        self.backend = backend

    def recognize(self, image_path: str, text_blocks: list[DetectedBlock]) -> list[DetectedBlock]:
        raise NotImplementedError(
            f"公式识别适配器 {self.backend} 仅预留接口，当前仓库第一阶段未接入真实模型。"
        )


class OpenCVFigureDetector(FigureDetector):
    def __init__(self, *, min_area_ratio: float = 0.01, expand_text_px: int = 4) -> None:
        self.min_area_ratio = min_area_ratio
        self.expand_text_px = expand_text_px

    def detect(self, image_path: str, text_blocks: list[DetectedBlock]) -> list[FigureRegion]:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            return []

        image = cv2.imread(str(Path(image_path)))
        if image is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

        text_mask = np.zeros_like(binary)
        for block in text_blocks:
            x1 = max(0, block.bbox.x1 - self.expand_text_px)
            y1 = max(0, block.bbox.y1 - self.expand_text_px)
            x2 = min(binary.shape[1], block.bbox.x2 + self.expand_text_px)
            y2 = min(binary.shape[0], block.bbox.y2 + self.expand_text_px)
            text_mask[y1:y2, x1:x2] = 255

        binary[text_mask > 0] = 0
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = image.shape[0] * image.shape[1] * self.min_area_ratio

        figures: list[FigureRegion] = []
        for index, contour in enumerate(contours, start=1):
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < min_area or w < 30 or h < 30:
                continue
            figures.append(
                FigureRegion(
                    figure_id=f"figure_{index}",
                    bbox=BBox(x, y, x + w, y + h),
                    figure_type="diagram",
                    source="opencv",
                )
            )
        figures.sort(key=lambda item: (item.bbox.y1, item.bbox.x1))
        return figures
