from __future__ import annotations

from collections import Counter

from .patterns import find_option_label, has_blank
from .schemas import DetectedBlock, QuestionType


class QuestionTypeClassifier:
    def classify(self, blocks: list[DetectedBlock]) -> QuestionType:
        option_counter = Counter()
        blank_hits = 0
        for block in blocks:
            if block.kind not in {"text", "formula"}:
                continue
            label = find_option_label(block.text)
            if label:
                option_counter[label] += 1
            if has_blank(block.text):
                blank_hits += 1

        if len(option_counter) >= 2:
            return QuestionType.SINGLE_CHOICE
        if blank_hits > 0:
            return QuestionType.FILL_BLANK
        return QuestionType.SOLUTION
