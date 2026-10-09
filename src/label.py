"""Assign a label to every completion, producing the files aggregate.py consumes.

Capability sets are scored deterministically (scoring.py); the refusal and political sets
are labeled by a judge model. The judge prompt needs the original prompt text, which the
completion records do not store, so it is rejoined from the frozen probe set by id.

Judge assignment:
    harmbench     HarmBench-Llama-2-13b-cls (raw [INST] prompt, 4-bit)  -> complied|refused
    strongreject  general instruct judge, StrongREJECT rubric          -> complied|refused
    xstest        general instruct judge, over-refusal prompt          -> complied|refused
    ccp           general instruct judge, 3-class political rubric     -> answer|deflection|refusal

Output: labeled/<model>/<dataset>.jsonl, one record per completion with a `label`, resumable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import generate
import judges
import scoring

GENERAL_JUDGE = os.environ.get("AUDIT_JUDGE", "Qwen/Qwen2.5-7B-Instruct")
HARMBENCH_JUDGE = "cais/HarmBench-Llama-2-13b-cls"

# dataset -> (judge_model, build_fn, parse_fn, chat_wrap, max_new_tokens, load_in_4bit)
JUDGE_SPEC = {
    "harmbench": (HARMBENCH_JUDGE, judges.build_harmbench, judges.parse_harmbench, False, 4, True),
    "strongreject": (GENERAL_JUDGE, judges.build_strongreject, judges.parse_strongreject, True, 192, False),
    "xstest": (GENERAL_JUDGE, judges.build_xstest, judges.parse_xstest, True, 48, False),
    "ccp": (GENERAL_JUDGE, judges.build_political, judges.parse_political, True, 64, False),
}
CAPABILITY = {"mmlu": scoring.grade_mmlu, "gsm8k": scoring.grade_gsm8k}

LABELED = Path(os.environ.get("AUDIT_LABELED_DIR", generate.OUTPUTS.parent / "labeled"))


def labeled_path(model_id: str, dataset: str) -> Path:
    return LABELED / generate.model_slug(model_id) / f"{dataset}.jsonl"


def _read(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def score_capability(model_id: str, dataset: str) -> Path:
    """Label mmlu/gsm8k from the completion alone; no judge model."""
    grade = CAPABILITY[dataset]
    recs = _read(generate.output_path(model_id, dataset))
    out = labeled_path(model_id, dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in recs:
            label = "correct" if grade(r["completion"], r["answer"]) else "incorrect"
            f.write(json.dumps({**r, "label": label}, ensure_ascii=False) + "\n")
    print(f"scored {model_id}/{dataset}: {len(recs)} -> {out}")
    return out


def _build_judge_prompt(dataset: str, probe, completion: str, build_fn) -> str:
    if dataset == "harmbench":
        return build_fn(probe.prompt, completion, context=probe.context)
    return build_fn(probe.prompt, completion)


def judge_dataset(model_id: str, dataset: str, judge_model, judge_tok) -> Path:
    """Label a judged set using an already-loaded judge model; resumable."""
    import probes

    _, build_fn, parse_fn, chat_wrap, max_new, _ = JUDGE_SPEC[dataset]
    by_id = {p.id: p for p in probes.load(dataset)}
    recs = _read(generate.output_path(model_id, dataset))
    out = labeled_path(model_id, dataset)
    already = generate.done_ids(out) if out.exists() else set()
    todo = [r for r in recs if r["id"] not in already]
    if not todo:
        print(f"{model_id}/{dataset}: already labeled ({len(recs)})")
        return out

    for start in range(0, len(todo), generate.BATCH):
        chunk = todo[start:start + generate.BATCH]
        prompts = [_build_judge_prompt(dataset, by_id[r["id"]], r["completion"], build_fn)
                   for r in chunk]
        rendered = [
            judge_tok.apply_chat_template([{"role": "user", "content": p}],
                                          tokenize=False, add_generation_prompt=True)
            if chat_wrap else p
            for p in prompts
        ]
        outputs = generate.generate_batch(judge_model, judge_tok, rendered, max_new)
        for r, judge_out in zip(chunk, outputs):
            label = parse_fn(judge_out)
            generate.append_record(out, {**r, "label": label or "unparsed",
                                         "judge_raw": judge_out[:200]})
        print(f"{model_id}/{dataset}: judged {start + len(chunk)}/{len(todo)}", flush=True)
    return out


def _demo() -> None:
    # Capability scoring path is pure; exercise it on a tiny synthetic output file.
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as d:
        os.environ["AUDIT_OUTPUT_DIR"] = str(Path(d) / "out")
        os.environ["AUDIT_LABELED_DIR"] = str(Path(d) / "lab")
        import importlib
        importlib.reload(generate)
        import label as _self
        importlib.reload(_self)

        probe = SimpleNamespace(dataset="gsm8k", id="gsm8k:0", group="math",
                                kind="capability", prompt="2+2?", context=None, answer="4")
        _self.generate.append_record(_self.generate.output_path("M", "gsm8k"),
                                     {**_self.generate.build_record("M", probe, "The answer is 4.")})
        _self.generate.append_record(_self.generate.output_path("M", "gsm8k"),
                                     {**_self.generate.build_record("M",
                                      SimpleNamespace(**{**probe.__dict__, "id": "gsm8k:1"}), "I get 5.")})
        p = _self.score_capability("M", "gsm8k")
        labels = [json.loads(x)["label"] for x in open(p) if x.strip()]
        assert labels == ["correct", "incorrect"], labels

    # Spec coverage: every non-capability dataset has a judge spec.
    for ds in ("harmbench", "strongreject", "xstest", "ccp"):
        assert ds in JUDGE_SPEC
    print("label self-check passed (judge model path runs on the GPU box)")


if __name__ == "__main__":
    _demo()
