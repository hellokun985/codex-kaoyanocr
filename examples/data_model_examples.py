from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from math_question_ocr.data_models import PARSED_QUESTION_JSON_SCHEMA_STYLE, build_all_examples


def main() -> None:
    payload = {
        "examples": {name: question.to_dict() for name, question in build_all_examples().items()},
        "json_schema_style": PARSED_QUESTION_JSON_SCHEMA_STYLE,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
