"""Generation harness: run a model over a probe set and record completions.

Runs on the GPU box (lazy-imports transformers so this module loads without torch).
The same harness produces model completions and, with a judge prompt, judge outputs --
judging is just generation with a judge model. Decoding is greedy for reproducibility;
Qwen3 is run with thinking mode off per the audit design.

Output: one JSONL per (model, dataset) under outputs/<model_slug>/<dataset>.jsonl, one
record per probe. The run is resumable -- already-recorded ids are skipped -- so a
destroyed-and-replaced instance continues where it stopped. Completions may contain
harmful text, so outputs/ is gitignored; only aggregate rates leave the box.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

# Max new tokens per set: enough for a judge to rule, longer where reasoning is needed.
MAX_NEW_TOKENS = {
    "harmbench": 512, "strongreject": 512, "xstest": 512,
    "ccp": 512, "mmlu": 8, "gsm8k": 512,
}


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def output_path(model_id: str, dataset: str) -> Path:
    return OUTPUTS / model_slug(model_id) / f"{dataset}.jsonl"


def done_ids(path: str | Path) -> set[str]:
    """Ids already recorded in a JSONL output, for resuming."""
    p = Path(path)
    if not p.exists():
        return set()
    out = set()
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(json.loads(line)["id"])
    return out


def append_record(path: str | Path, rec: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_record(model_id: str, probe, completion: str) -> dict:
    return {
        "model": model_id, "dataset": probe.dataset, "id": probe.id,
        "group": probe.group, "kind": probe.kind, "completion": completion,
        "answer": probe.answer,
    }


# ---------- box-side: model loading and generation (lazy transformers) ----------
def load_model(model_id: str, dtype: str = "bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), device_map="auto")
    model.eval()
    return model, tok


def _apply_template(tok, prompt: str, model_id: str) -> str:
    """Wrap a user prompt in the model's chat template; Qwen3 thinking mode off."""
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_id.lower():
        kwargs["enable_thinking"] = False
    return tok.apply_chat_template(messages, **kwargs)


def generate_one(model, tok, text: str, max_new_tokens: int, model_id: str) -> str:
    import torch

    rendered = _apply_template(tok, text, model_id)
    inputs = tok(rendered, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def run(model_id: str, dataset: str) -> Path:
    """Generate completions for one model over one probe set; resumable."""
    import probes

    items = probes.load(dataset)
    path = output_path(model_id, dataset)
    already = done_ids(path)
    todo = [p for p in items if p.id not in already]
    if not todo:
        print(f"{model_id} / {dataset}: all {len(items)} done")
        return path
    model, tok = load_model(model_id)
    maxn = MAX_NEW_TOKENS[dataset]
    for i, probe in enumerate(todo, 1):
        text = (f"{probe.context}\n\n{probe.prompt}" if probe.context else probe.prompt)
        completion = generate_one(model, tok, text, maxn, model_id)
        append_record(path, build_record(model_id, probe, completion))
        if i % 25 == 0:
            print(f"{model_id} / {dataset}: {i}/{len(todo)} (+{len(already)} resumed)")
    print(f"{model_id} / {dataset}: wrote {len(todo)} records to {path}")
    return path


def _demo() -> None:
    import tempfile
    from types import SimpleNamespace

    assert model_slug("Qwen/Qwen3-8B") == "Qwen__Qwen3-8B"
    assert output_path("Qwen/Qwen3-8B", "mmlu").name == "mmlu.jsonl"

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.jsonl"
        assert done_ids(p) == set()
        probe = SimpleNamespace(dataset="ccp", id="ccp:0", group="Taiwan",
                                kind="political", prompt="q", context=None, answer=None)
        append_record(p, build_record("M", probe, "a response"))
        assert done_ids(p) == {"ccp:0"}
        rec = json.loads(p.read_text().strip())
        assert rec["model"] == "M" and rec["completion"] == "a response"
        # Resume skips the recorded id.
        probe2 = SimpleNamespace(**{**probe.__dict__, "id": "ccp:1"})
        todo = [x for x in (probe, probe2) if x.id not in done_ids(p)]
        assert [x.id for x in todo] == ["ccp:1"]

    print("generate self-check passed (model path not exercised; runs on the GPU box)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] != "selfcheck":
        run(sys.argv[1], sys.argv[2])
    else:
        _demo()
