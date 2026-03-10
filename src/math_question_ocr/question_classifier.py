from __future__ import annotations

import re
from dataclasses import dataclass, field

from .data_models import JsonSerializable, LineCluster, OCRBlock, QuestionType

OPTION_LABEL_RE = re.compile(r"(?:(?<=^)|(?<=[\s　]))(?P<label>[ABCDabcd480OoG])\s*[.．、:：\)）]")
BLANK_RE = re.compile(r"(_{2,}|﹍{2,}|_{1,}\s*|（\s*）|\(\s*\)|\[+\s*\]+|□+|◻+)")
SUBQUESTION_RE = re.compile(r"^\s*(?:[（(]\d+[)）]|\d+[.．、]|[IVXivx]{1,5}[.．、]?)")
FULL_SCORE_RE = re.compile(r"本题满分\s*\d+(?:\.\d+)?\s*分")
SOLUTION_KEYWORD_RE = re.compile(r"(证明|求证|求解|解答|计算|求下列|已知|设|若|求)")
MATRIX_HINT_RE = re.compile(r"(矩阵|行列式|\[[^\]]{3,}\]|\\begin\{matrix\})")

LABEL_ALIASES = {
    "A": "A",
    "a": "A",
    "4": "A",
    "B": "B",
    "b": "B",
    "8": "B",
    "C": "C",
    "c": "C",
    "G": "C",
    "D": "D",
    "d": "D",
    "O": "D",
    "o": "D",
    "0": "D",
}


@dataclass(slots=True)
class ClassificationResult(JsonSerializable):
    question_type: QuestionType
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)


class RuleBasedQuestionClassifier:
    def classify(
        self,
        *,
        ocr_blocks: list[OCRBlock] | None = None,
        line_clusters: list[LineCluster] | None = None,
    ) -> ClassificationResult:
        lines = line_clusters or []
        texts = [line.text.strip() for line in lines if line.text.strip()]
        if not texts:
            texts = [block.text.strip() for block in (ocr_blocks or []) if block.text.strip()]

        features = self._extract_features(texts)
        scores, reason_map = self._score(features)
        winner = max(scores, key=scores.get)
        reasons = reason_map[winner] or ["默认回退到该题型的规则得分最高。"]
        confidence = round(max(0.0, min(0.99, scores[winner])), 2)

        return ClassificationResult(
            question_type=QuestionType(winner),
            reasons=reasons,
            confidence=confidence,
            score_breakdown={key: round(value, 3) for key, value in scores.items()},
        )

    def _extract_features(self, texts: list[str]) -> dict[str, object]:
        option_labels: list[str] = []
        option_hits = 0
        blank_hits = 0
        subquestion_hits = 0
        full_score_hit = False
        solution_keyword_hits = 0
        matrix_hits = 0
        long_line_hits = 0

        for text in texts:
            option_matches = OPTION_LABEL_RE.findall(text)
            option_hits += len(option_matches)
            option_labels.extend(self._normalize_labels(option_matches))
            blank_hits += len(BLANK_RE.findall(text))
            if SUBQUESTION_RE.match(text):
                subquestion_hits += 1
            if FULL_SCORE_RE.search(text):
                full_score_hit = True
            solution_keyword_hits += len(SOLUTION_KEYWORD_RE.findall(text))
            matrix_hits += len(MATRIX_HINT_RE.findall(text))
            if len(self._normalize_text(text)) >= 18:
                long_line_hits += 1

        distinct_option_labels = sorted(set(option_labels))
        return {
            "line_count": len(texts),
            "option_hits": option_hits,
            "distinct_option_labels": distinct_option_labels,
            "blank_hits": blank_hits,
            "subquestion_hits": subquestion_hits,
            "full_score_hit": full_score_hit,
            "solution_keyword_hits": solution_keyword_hits,
            "matrix_hits": matrix_hits,
            "long_line_hits": long_line_hits,
            "multi_line_long_body": long_line_hits >= 2,
        }

    def _score(self, features: dict[str, object]) -> tuple[dict[str, float], dict[str, list[str]]]:
        scores = {
            QuestionType.SINGLE_CHOICE.value: 0.05,
            QuestionType.FILL_BLANK.value: 0.05,
            QuestionType.SOLUTION.value: 0.15,
        }
        reasons = {
            QuestionType.SINGLE_CHOICE.value: [],
            QuestionType.FILL_BLANK.value: [],
            QuestionType.SOLUTION.value: [],
        }

        option_labels = features["distinct_option_labels"]
        option_hits = int(features["option_hits"])
        blank_hits = int(features["blank_hits"])
        subquestion_hits = int(features["subquestion_hits"])
        full_score_hit = bool(features["full_score_hit"])
        solution_keyword_hits = int(features["solution_keyword_hits"])
        matrix_hits = int(features["matrix_hits"])
        multi_line_long_body = bool(features["multi_line_long_body"])
        line_count = int(features["line_count"])

        if len(option_labels) >= 2:
            scores[QuestionType.SINGLE_CHOICE.value] += 0.90
            reasons[QuestionType.SINGLE_CHOICE.value].append(
                f"检测到选项标签 {','.join(option_labels)}，符合选择题结构。"
            )
        if len(option_labels) >= 4:
            scores[QuestionType.SINGLE_CHOICE.value] += 0.12
            reasons[QuestionType.SINGLE_CHOICE.value].append("A/B/C/D 选项标签较完整。")
        if option_hits >= 4:
            scores[QuestionType.SINGLE_CHOICE.value] += 0.05
        if blank_hits > 0:
            scores[QuestionType.SINGLE_CHOICE.value] -= 0.20
        if subquestion_hits > 0:
            scores[QuestionType.SINGLE_CHOICE.value] -= 0.25
        if multi_line_long_body and len(option_labels) < 2:
            scores[QuestionType.SINGLE_CHOICE.value] -= 0.10

        if blank_hits > 0:
            scores[QuestionType.FILL_BLANK.value] += 0.92
            reasons[QuestionType.FILL_BLANK.value].append(f"检测到 {blank_hits} 处空白线/答案槽位。")
        if blank_hits >= 2:
            scores[QuestionType.FILL_BLANK.value] += 0.08
        if len(option_labels) >= 2:
            scores[QuestionType.FILL_BLANK.value] -= 0.35
        if subquestion_hits > 0:
            scores[QuestionType.FILL_BLANK.value] -= 0.18
        if multi_line_long_body and blank_hits == 0:
            scores[QuestionType.FILL_BLANK.value] -= 0.12

        if subquestion_hits > 0:
            scores[QuestionType.SOLUTION.value] += 0.42
            reasons[QuestionType.SOLUTION.value].append(f"检测到 {subquestion_hits} 个小问标记。")
        if full_score_hit:
            scores[QuestionType.SOLUTION.value] += 0.12
            reasons[QuestionType.SOLUTION.value].append("出现“本题满分 xx 分”表述。")
        if solution_keyword_hits > 0:
            bonus = min(0.24, 0.06 * solution_keyword_hits)
            scores[QuestionType.SOLUTION.value] += bonus
            reasons[QuestionType.SOLUTION.value].append("存在证明/求解类表述。")
        if matrix_hits > 0:
            scores[QuestionType.SOLUTION.value] += 0.10
            reasons[QuestionType.SOLUTION.value].append("检测到矩阵或大块公式特征。")
        if multi_line_long_body:
            scores[QuestionType.SOLUTION.value] += 0.14
            reasons[QuestionType.SOLUTION.value].append("存在多行长正文，接近解答题版式。")
        if line_count >= 4:
            scores[QuestionType.SOLUTION.value] += 0.05
        if len(option_labels) >= 2:
            scores[QuestionType.SOLUTION.value] -= 0.38
        if blank_hits > 0 and len(option_labels) == 0:
            scores[QuestionType.SOLUTION.value] -= 0.12

        for key in scores:
            scores[key] = max(0.0, min(0.99, scores[key]))

        return scores, reasons

    def _normalize_labels(self, labels: list[str]) -> list[str]:
        normalized: list[str] = []
        for label in labels:
            if isinstance(label, tuple):
                raw = label[0]
            else:
                raw = label
            mapped = LABEL_ALIASES.get(raw, "")
            if mapped:
                normalized.append(mapped)
        return normalized

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.split())
