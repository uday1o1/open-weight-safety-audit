"""Probe-set loaders for the audit.

Six evaluation sets, each fetched from a pinned primary source and normalized to a
common `Probe` record:

    harmful refusal   HarmBench (standard+contextual, copyright excluded)  300
    harmful refusal   StrongREJECT                                         313
    over-refusal      XSTest (safe prompts only)                           250
    political         promptfoo CCP-sensitive-prompts                    1,360
    capability        MMLU (seeded subset of the 'all' test split)       1,000
    capability        GSM8K (seeded subset of the 'main' test split)       500

Reproducibility (ML lens): every source is pinned to a commit (REV). Manifests under
manifests/ record, per item, a sha256 of the prompt text and the gold-answer hash --
never the text itself -- so the exact evaluated set is frozen and verifiable without
redistributing harmful or politically sensitive prompts. Raw text is re-fetched on
demand into data/raw/ (gitignored) and checked against the manifest hashes at load.
The two capability subsets are drawn with a fixed seed and their chosen ids frozen.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFESTS = ROOT / "manifests"

SEED = 253              # course number; fixes the MMLU/GSM8K subsets
MMLU_N = 1000
GSM8K_N = 500

# Pinned source revisions, captured 2026-10-09.
REV = {
    "harmbench": "c0423b952435fcc8467108d8f25962dbae5b7de2",
    "strongreject": "3432b2d696b428f242bd507df96d80f686571d5e",
    "xstest": "475f10bf0a3d6a9dfb174b6de1a38afbfdff98a5",
    "ccp": "f6fc9a004916913a84883591a96a6709feb02992",
    "mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
}

SOURCE = {
    "harmbench": f"https://raw.githubusercontent.com/centerforaisafety/HarmBench/{REV['harmbench']}/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    "strongreject": f"https://raw.githubusercontent.com/alexandrasouly/strongreject/{REV['strongreject']}/strongreject_dataset/strongreject_dataset.csv",
    "xstest": f"https://raw.githubusercontent.com/paul-rottger/xstest/{REV['xstest']}/xstest_prompts.csv",
    "ccp": f"https://huggingface.co/datasets/promptfoo/CCP-sensitive-prompts/resolve/{REV['ccp']}/ccp-sensitive-prompts.csv",
    "mmlu": f"https://huggingface.co/datasets/cais/mmlu/resolve/{REV['mmlu']}/all/test-00000-of-00001.parquet",
    "gsm8k": f"https://huggingface.co/datasets/openai/gsm8k/resolve/{REV['gsm8k']}/main/test-00000-of-00001.parquet",
}

EXPECTED_COUNT = {
    "harmbench": 300, "strongreject": 313, "xstest": 250,
    "ccp": 1360, "mmlu": MMLU_N, "gsm8k": GSM8K_N,
}


@dataclass(frozen=True)
class Probe:
    dataset: str
    id: str
    group: str            # category, for slicing
    kind: str             # harmful | safe | political | capability
    prompt: str
    context: str | None = None
    answer: str | None = None   # gold answer for capability sets


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fetch(name: str) -> bytes:
    """Download a pinned source into data/raw/, cached by name+revision."""
    RAW.mkdir(parents=True, exist_ok=True)
    ext = "parquet" if SOURCE[name].endswith(".parquet") else "csv"
    cache = RAW / f"{name}-{REV[name][:12]}.{ext}"
    if not cache.exists():
        req = urllib.request.Request(SOURCE[name], headers={"User-Agent": "audit/1.0"})
        token = (Path.home() / ".cache/huggingface/token")
        if "huggingface.co" in SOURCE[name] and token.exists():
            req.add_header("Authorization", f"Bearer {token.read_text().strip()}")
        with urllib.request.urlopen(req) as r:
            data = r.read()
        if data[:40].lstrip().startswith(b"Access to") or data[:15] == b"<!DOCTYPE html>":
            raise RuntimeError(f"{name}: source returned an access/error page, not data")
        cache.write_bytes(data)
    return cache.read_bytes()


def _rows_csv(name: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(_fetch(name).decode("utf-8", "replace"))))


def _rows_parquet(name: str) -> list[dict]:
    t = pq.read_table(io.BytesIO(_fetch(name)))
    return t.to_pylist()


# ---------- per-source normalization to the full (pre-subset) probe list ----------
def _harmbench() -> list[Probe]:
    out = []
    for r in _rows_csv("harmbench"):
        if r["FunctionalCategory"] not in ("standard", "contextual"):
            continue  # copyright excluded
        ctx = r.get("ContextString") or None
        out.append(Probe("harmbench", r["BehaviorID"], r["SemanticCategory"],
                          "harmful", r["Behavior"], context=ctx))
    return out


def _strongreject() -> list[Probe]:
    return [Probe("strongreject", f"sr:{i}", r["category"], "harmful", r["forbidden_prompt"])
            for i, r in enumerate(_rows_csv("strongreject"))]


def _xstest() -> list[Probe]:
    return [Probe("xstest", r["id"], r["type"], "safe", r["prompt"])
            for r in _rows_csv("xstest") if r["label"] == "safe"]


def _ccp() -> list[Probe]:
    return [Probe("ccp", f"ccp:{i}", r["subject"], "political", r["prompt"])
            for i, r in enumerate(_rows_csv("ccp"))]


def _mmlu_full() -> list[Probe]:
    out = []
    for i, r in enumerate(_rows_parquet("mmlu")):
        choices = r["choices"]
        body = r["question"] + "\n" + "\n".join(
            f"{'ABCD'[j]}. {c}" for j, c in enumerate(choices))
        out.append(Probe("mmlu", f"{r['subject']}:{i}", r["subject"], "capability",
                         body, answer="ABCD"[r["answer"]]))
    return out


def _gsm8k_full() -> list[Probe]:
    out = []
    for i, r in enumerate(_rows_parquet("gsm8k")):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append(Probe("gsm8k", f"gsm8k:{i}", "math", "capability",
                         r["question"], answer=gold))
    return out


FULL = {
    "harmbench": _harmbench, "strongreject": _strongreject, "xstest": _xstest,
    "ccp": _ccp, "mmlu": _mmlu_full, "gsm8k": _gsm8k_full,
}
SUBSET_N = {"mmlu": MMLU_N, "gsm8k": GSM8K_N}


def _subset(items: list[Probe], name: str) -> list[Probe]:
    """Deterministic seeded subset for the capability sets; identity otherwise."""
    k = SUBSET_N.get(name)
    if k is None:
        return items
    idx = sorted(random.Random(f"{SEED}:{name}").sample(range(len(items)), k))
    return [items[i] for i in idx]


# ---------- manifest build / verify ----------
def build_manifest(name: str) -> dict:
    items = _subset(FULL[name](), name)
    manifest = {
        "dataset": name,
        "source": SOURCE[name],
        "revision": REV[name],
        "kind": items[0].kind,
        "count": len(items),
        "seed": SEED if name in SUBSET_N else None,
        "items": [
            {"id": p.id, "group": p.group, "ctx": p.context is not None,
             "prompt_sha256": _sha(p.prompt),
             "answer_sha256": _sha(p.answer) if p.answer is not None else None}
            for p in items
        ],
    }
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / f"{name}.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load(name: str) -> list[Probe]:
    """Return the frozen probe set, verified against the committed manifest hashes."""
    mpath = MANIFESTS / f"{name}.json"
    if not mpath.exists():
        raise FileNotFoundError(f"no manifest for {name}; run build_manifest first")
    manifest = json.loads(mpath.read_text())
    by_id = {p.id: p for p in _subset(FULL[name](), name)}
    out = []
    for entry in manifest["items"]:
        p = by_id.get(entry["id"])
        if p is None:
            raise ValueError(f"{name}: manifest id {entry['id']} missing from source")
        if _sha(p.prompt) != entry["prompt_sha256"]:
            raise ValueError(f"{name}: prompt hash mismatch for {entry['id']}")
        out.append(p)
    return out


def _demo() -> None:
    for name, expected in EXPECTED_COUNT.items():
        items = _subset(FULL[name](), name)
        assert len(items) == expected, f"{name}: got {len(items)}, want {expected}"
        assert len({p.id for p in items}) == expected, f"{name}: duplicate ids"
        # seeded subsets must be stable across calls
        if name in SUBSET_N:
            again = _subset(FULL[name](), name)
            assert [p.id for p in items] == [p.id for p in again], f"{name}: unstable subset"
        print(f"{name:12} {len(items):>5}  e.g. group={items[0].group!r}")
    print("probes self-check passed")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        for name in EXPECTED_COUNT:
            m = build_manifest(name)
            print(f"built manifest {name}: {m['count']} items")
    else:
        _demo()
