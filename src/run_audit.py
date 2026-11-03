"""Box orchestrator: generate -> label -> aggregate, resumable end to end.

Phases run in order; generation and judging skip records already written (capability
scoring and aggregation are cheap and rerun), so an interrupted run re-runs this and
continues where it stopped (on a new machine, copy the output and labeled roots over
first). Models load one at a time; each judge loads once per run and labels every target.
Completions stay in the output and labeled roots (off the repo); only results/ (rates and
id/label files, no text) is meant to be committed.

    python src/run_audit.py all        # everything
    python src/run_audit.py generate   # just completions
    python src/run_audit.py label      # just labeling (after generation)
    python src/run_audit.py aggregate  # just the result tables
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import aggregate
import generate
import label

FULL = ["harmbench", "strongreject", "xstest", "ccp", "mmlu", "gsm8k"]
BASE_SETS = ["harmbench", "strongreject", "xstest", "ccp"]  # base vs instruct

TARGETS = {
    "Qwen/Qwen3-8B": FULL,
    "zai-org/GLM-4-9B-0414": FULL,
    "meta-llama/Llama-3.1-8B-Instruct": FULL,
    "Qwen/Qwen3-8B-Base": BASE_SETS,          # pretraining reference, helpful-only URIAL
    "Qwen/Qwen3-8B-Base+safety": BASE_SETS,   # sensitivity: URIAL with safety instruction
}
# Base prompt conditions: label -> base-model id compared against the instruct model.
BASE_CONDITIONS = {"urial_help": "Qwen/Qwen3-8B-Base", "urial_safety": "Qwen/Qwen3-8B-Base+safety"}
JUDGED = ["strongreject", "xstest", "ccp"]    # general judge
CAPABILITY = ["mmlu", "gsm8k"]                 # scored, no model
RESULTS = aggregate.RESULTS


def phase_generate() -> None:
    # One model's load/gen failure (transient gated-repo auth, OOM, network) must not abort the
    # others. Retry once, then skip and move on; the run is resumable, so a skipped model is
    # picked up on the next invocation. Record skips so the gap is visible.
    failed = []
    for model_id, datasets in TARGETS.items():
        for attempt in (1, 2):
            try:
                generate.run_model(model_id, datasets)
                break
            except Exception as e:
                print(f"[run_audit] {model_id} generation attempt {attempt} failed: {e}", flush=True)
                if attempt == 2:
                    failed.append(model_id)
    if failed:
        print(f"[run_audit] GENERATION INCOMPLETE, skipped models: {failed}", flush=True)


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
            del jm, jt
            generate.free_gpu()

    # 3. HarmBench classifier (4-bit): one load for the harmbench set
    hm, ht = generate.load_model(label.HARMBENCH_JUDGE, load_in_4bit=True)
    try:
        for model_id, datasets in TARGETS.items():
            if "harmbench" in datasets:
                # 13B judge + ~1k-token inputs: small batch.
                label.judge_dataset(model_id, "harmbench", hm, ht,
                                    batch=int(os.environ.get("AUDIT_JUDGE_BATCH", "4")))
    finally:
        del hm, ht
        generate.free_gpu()


def phase_aggregate() -> None:
    summary = aggregate.summarize(label.LABELED)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Base vs instruct on the same prompts, per URIAL condition, where both are labeled.
    instruct = "Qwen/Qwen3-8B"
    paired = {}
    for cond, base in BASE_CONDITIONS.items():
        for ds in BASE_SETS:
            bp, ip = label.labeled_path(base, ds), label.labeled_path(instruct, ds)
            if bp.exists() and ip.exists():
                paired.setdefault(cond, {})[ds] = aggregate.paired_difference(
                    aggregate.load_labeled(bp), aggregate.load_labeled(ip), ds)
    (RESULTS / "base_vs_instruct.json").write_text(json.dumps(paired, indent=1))
    print(json.dumps({"rates": summary, "base_vs_instruct": paired}, indent=1))


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
