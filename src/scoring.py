"""Turn raw model completions into binary outcomes for the capability sets.

Refusal and political labels come from judge models (see judges.py); these two
scorers are deterministic string extraction, so they live here and are self-tested
offline. Both are deliberately lenient about surrounding formatting (the models are
not forced into a rigid output format) but strict about what counts as an answer.
"""
from __future__ import annotations

import re

# A multiple-choice answer letter, as it tends to appear in free-form completions:
# "Answer: B", "The answer is (C).", "**D**", "b)", or a bare leading "C".
_MCQ_TAGGED = re.compile(
    r"(?:answer|option|choice)\b[^A-Da-d]{0,10}([A-Da-d])\b", re.IGNORECASE)
_MCQ_PAREN = re.compile(r"[(\[]\s*([A-Da-d])\s*[)\].:]")


def extract_mcq_answer(completion: str) -> str | None:
    """Best-effort single-letter answer (A-D) from an MMLU completion, else None."""
    if not completion:
        return None
    # Prefer an explicitly tagged answer, scanning from the end (models restate it last).
    tagged = _MCQ_TAGGED.findall(completion)
    if tagged:
        return tagged[-1].upper()
    paren = _MCQ_PAREN.findall(completion)
    if paren:
        return paren[-1].upper()
    # A bare leading letter, allowing markdown emphasis: "**D**", "C.", "b) ...".
    # Require the next character to be non-alphanumeric so prose ("Apples...") is not a hit.
    clean = completion.strip().lstrip("*").strip()
    if clean and clean[0].upper() in "ABCD" and (len(clean) == 1 or not clean[1].isalnum()):
        return clean[0].upper()
    return None


def grade_mmlu(completion: str, gold: str) -> bool:
    return extract_mcq_answer(completion) == gold.upper()


# A number, possibly negative, with thousands separators, decimals, trailing %, or $.
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")


def _normalize_number(tok: str) -> str:
    tok = tok.replace(",", "").replace("$", "").replace("%", "")
    if tok.endswith(".0"):
        tok = tok[:-2]
    if "." in tok:
        tok = tok.rstrip("0").rstrip(".")
    return tok


def extract_final_number(completion: str) -> str | None:
    """The model's final numeric answer from a GSM8K completion, normalized."""
    if not completion:
        return None
    # Prefer the number after a GSM8K-style '####' marker if the model used one.
    if "####" in completion:
        tail = completion.rsplit("####", 1)[1]
        nums = _NUMBER.findall(tail)
        if nums:
            return _normalize_number(nums[0])
    nums = _NUMBER.findall(completion)
    return _normalize_number(nums[-1]) if nums else None


def grade_gsm8k(completion: str, gold: str) -> bool:
    got = extract_final_number(completion)
    return got is not None and got == _normalize_number(gold)


def _demo() -> None:
    # MMLU: tagged, parenthesized, bolded, leading, restated-last, and no-answer.
    assert extract_mcq_answer("Answer: B") == "B"
    assert extract_mcq_answer("The answer is (C).") == "C"
    assert extract_mcq_answer("**D**") == "D"
    assert extract_mcq_answer("b) because ...") == "B"
    assert extract_mcq_answer("I think A, but actually the answer is D") == "D"
    assert extract_mcq_answer("none of these look right") is None
    assert grade_mmlu("The answer is A.", "A") and not grade_mmlu("Answer: A", "B")

    # GSM8K: commas, $, decimals, ####, trailing text, pick the final one.
    assert extract_final_number("So the total is 1,024 apples.") == "1024"
    assert extract_final_number("It costs $42.") == "42"
    assert extract_final_number("...#### 18") == "18"
    assert extract_final_number("First 10, then 20, so 30 total.") == "30"
    assert extract_final_number("72.0") == "72"
    assert extract_final_number("no numbers here") is None
    assert grade_gsm8k("The answer is 7.", "7")
    assert grade_gsm8k("= 1,200", "1200")
    assert not grade_gsm8k("I get 41.", "42")
    # '####' marker wins over a later stray number.
    assert extract_final_number("work... #### 15 (that's 15 items)") == "15"
    print("scoring self-check passed")


if __name__ == "__main__":
    _demo()
