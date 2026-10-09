"""Box orchestrator: generate -> label -> aggregate, resumable end to end.

Phases run in order and each skips completed work, so a destroyed-and-replaced instance
re-runs this and continues. Models load one at a time; judges load once and label every
target. Generation and completions stay in the output root (off the repo); only the
aggregate JSON in results/ is meant to be committed.

    python src/run_audit.py all        # everything
    python src/run_audit.py generate   # just completions
    python src/run_audit.py label      # just labeling (after generation)
    python src/run_audit.py aggregate  # just the result tables
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import aggregate
import generate
import label

FULL = ["harmbench", "strongreject", "xstest", "ccp", "mmlu", "gsm8k"]
RQ2_SETS = ["harmbench", "strongreject", "xstest", "ccp"]   # base vs instruct

TARGETS = {
    "Qwen/Qwen3-8B": FULL,
    "zai-org/GLM-4-9B-0414": FULL,
    "meta-llama/Llama-3.1-8B-Instruct": FULL,
    "Qwen/Qwen3-8B-Base": RQ2_SETS,          # URIAL-prompted pretraining reference
}
JUDGED = ["strongreject", "xstest", "ccp"]    # general judge
CAPABILITY = ["mmlu", "gsm8k"]                 # scored, no model
RESULTS = aggregate.RESULTS


def phase_generate() -> None:
    for model_id, datasets in TARGETS.items():
        generate.run_model(model_id, datasets)


def phase_label() -> None:
    # 1. capability: deterministic, no model
    for model_id, datasets in TARGETS.items():
        for ds in CAPABILITY:
            if ds in datasets:
                label.score_capability(model_id, ds)

    # 2. general instruct judge: one load for StrongREJECT / XSTest / CCP
    if any(ds in d for d in TARGETS.values() for ds in JUDGED):
        jm, jt = generate.load_model(label.GENERAL_JUDGE)
        try:
            for ds in JUDGED:
                for model_id, datasets in TARGETS.items():
                    if ds in datasets:
                        label.judge_dataset(model_id, ds, jm, jt)
        finally:
            generate.free_gpu(jm)

    # 3. HarmBench classifier (4-bit): one load for the harmbench set
    hm, ht = generate.load_model(label.HARMBENCH_JUDGE, load_in_4bit=True)
    try:
        for model_id, datasets in TARGETS.items():
            if "harmbench" in datasets:
                label.judge_dataset(model_id, "harmbench", hm, ht)
    finally:
        generate.free_gpu(hm)


def phase_aggregate() -> None:
    summary = aggregate.summarize(label.LABELED)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # RQ2: base vs instruct on the same prompts, where both have been labeled.
    base = "Qwen/Qwen3-8B-Base"
    instruct = "Qwen/Qwen3-8B"
    rq2 = {}
    for ds in RQ2_SETS:
        bp, ip = label.labeled_path(base, ds), label.labeled_path(instruct, ds)
        if bp.exists() and ip.exists():
            rq2[ds] = aggregate.paired_difference(
                aggregate.load_labeled(bp), aggregate.load_labeled(ip), ds)
    (RESULTS / "rq2_base_vs_instruct.json").write_text(json.dumps(rq2, indent=1))
    print(json.dumps({"rates": summary, "rq2": rq2}, indent=1))


def main() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if phase in ("all", "generate"):
        phase_generate()
    if phase in ("all", "label"):
        phase_label()
    if phase in ("all", "aggregate"):
        phase_aggregate()
    print(f"[run_audit] phase={phase} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
