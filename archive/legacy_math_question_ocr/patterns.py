from __future__ import annotations

import re

QUESTION_NO_RE = re.compile(r"^\s*(?P<no>\d+)\s*[.．、)]?\s*")
SCORE_RE = re.compile(r"[（(]?\s*(?P<score>\d+(?:\.\d+)?)\s*分\s*[)）]?")
OPTION_RE = re.compile(r"^\s*(?P<label>[A-D])\s*[.．、:：]\s*(?P<content>.*)$")
BLANK_RE = re.compile(r"(_{2,}|﹍{2,}|_{1,}\s*|（\s*）|\(\s*\)|\[+\s*\]+)")
SUBQUESTION_RE = re.compile(
    r"^\s*(?P<marker>(?:[（(]\d+[)）])|(?:\d+\.)|(?:[IVXivx]{1,5}[.．、]?))\s*(?P<content>.*)$"
)


def find_question_no(text: str) -> str:
    match = QUESTION_NO_RE.match(text)
    return match.group("no") if match else ""


def strip_question_no(text: str) -> str:
    return QUESTION_NO_RE.sub("", text, count=1).strip()


def find_score(text: str) -> float | None:
    match = SCORE_RE.search(text)
    return float(match.group("score")) if match else None


def strip_score(text: str) -> str:
    return SCORE_RE.sub("", text).strip()


def find_option_label(text: str) -> str:
    match = OPTION_RE.match(text)
    return match.group("label") if match else ""


def strip_option_label(text: str) -> str:
    match = OPTION_RE.match(text)
    if not match:
        return text.strip()
    return match.group("content").strip()


def has_blank(text: str) -> bool:
    return bool(BLANK_RE.search(text))


def count_blanks(text: str) -> int:
    return len(BLANK_RE.findall(text))


def find_subquestion_marker(text: str) -> str:
    match = SUBQUESTION_RE.match(text)
    return match.group("marker") if match else ""


def strip_subquestion_marker(text: str) -> str:
    match = SUBQUESTION_RE.match(text)
    if not match:
        return text.strip()
    return match.group("content").strip()
