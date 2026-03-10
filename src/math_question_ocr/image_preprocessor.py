from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PreprocessResult:
    original_path: str
    processed_path: str
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] | None = None


class ImagePreprocessor:
    def preprocess(self, image_path: str, *, output_dir: str | None = None) -> PreprocessResult:
        image = Path(image_path)
        if not image.exists():
            raise FileNotFoundError(f"image not found: {image_path}")

        metadata: dict[str, Any] = {"preprocess_mode": "noop"}
        processed_path = str(image)
        width: int | None = None
        height: int | None = None

        try:
            import cv2  # type: ignore

            frame = cv2.imread(str(image))
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                height, width = binary.shape[:2]
                if output_dir:
                    out_dir = Path(output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    target = out_dir / f"{image.stem}.preprocessed.png"
                    cv2.imwrite(str(target), binary)
                    processed_path = str(target)
                    metadata["preprocess_mode"] = "opencv_gray_otsu"
                else:
                    metadata["preprocess_mode"] = "opencv_gray_otsu_unsaved"
        except ImportError:
            try:
                from PIL import Image  # type: ignore

                with Image.open(image) as pil_image:
                    width, height = pil_image.size
                    metadata["preprocess_mode"] = "pil_probe_only"
            except ImportError:
                metadata["preprocess_mode"] = "file_exists_only"

        return PreprocessResult(
            original_path=str(image),
            processed_path=processed_path,
            width=width,
            height=height,
            metadata=metadata,
        )
