"""Generation harness: run a model over a probe set and record completions.

Runs on the GPU box (lazy-imports torch/transformers so this module loads without them).
Batched greedy decoding keeps a single 24 GB GPU sufficient. Instruction-tuned models
use their chat template (Qwen3 with thinking mode off); the Qwen3 base model is prompted
with the URIAL in-context alignment template for the RQ2 pretraining comparison. The same
harness serves judge models (judging is generation with a judge prompt), and can load a
large judge in 4-bit.

Output: one JSONL per (model, dataset) under the output root (AUDIT_OUTPUT_DIR, default
outside the repo on the box), one record per probe, resumable by skipping recorded ids.
Completions may contain harmful text, so the output root is never committed; only aggregate
rates leave the box.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = Path(os.environ.get("AUDIT_OUTPUT_DIR", ROOT / "outputs"))
CACHE = Path(os.environ.get("AUDIT_CACHE_DIR", ROOT / "data" / "raw"))

MAX_NEW_TOKENS = {
    "harmbench": 512, "strongreject": 512, "xstest": 512,
    "ccp": 512, "mmlu": 8, "gsm8k": 512,
}
BATCH = int(os.environ.get("AUDIT_BATCH", "16"))

# URIAL in-context alignment prompts for the base model (RQ2), Re-Align/URIAL commit b3d6f523.
# "help" (inst_1k_v4.help.txt) has no safety instruction and no refusal example, so any
# refusal it produces comes from the base model itself: the primary RQ2 condition.
# "safety" (inst_1k_v4.txt) tells the model it may "reject to answer some controversial
# topics" and shows a refusal example: a sensitivity condition, selected with a "+safety"
# suffix on the model id (e.g. "Qwen/Qwen3-8B-Base+safety").
URIAL_ROOT = "https://raw.githubusercontent.com/Re-Align/URIAL/b3d6f523644fcd6592f90c4344b3615d4c4db30c/urial_prompts/"
URIAL_FILES = {"help": "inst_1k_v4.help.txt", "safety": "inst_1k_v4.txt"}


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def output_path(model_id: str, dataset: str) -> Path:
    return OUTPUTS / model_slug(model_id) / f"{dataset}.jsonl"


def done_ids(path: str | Path) -> set[str]:
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


def hf_id(model_id: str) -> str:
    """Hugging Face repo id, without a condition suffix like "+safety"."""
    return model_id.split("+")[0]


def urial_variant(model_id: str) -> str:
    return "safety" if model_id.endswith("+safety") else "help"


def is_base_model(model_id: str) -> bool:
    return hf_id(model_id).lower().rstrip("/").endswith("-base")


def _urial_template(variant: str = "help") -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    name = URIAL_FILES[variant]
    cache = CACHE / f"urial_{name}"
    if not cache.exists():
        import time
        req = urllib.request.Request(URIAL_ROOT + name, headers={"User-Agent": "audit/1.0"})
        for i in range(6):                           # retry/backoff: transient SSL/network blip
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    cache.write_bytes(r.read())
                break
            except Exception as e:
                if i == 5:
                    raise
                print(f"[generate] URIAL fetch failed ({e}); retry in {2 ** i}s", flush=True)
                time.sleep(2 ** i)
    return cache.read_text()


def render_base(query: str, variant: str = "help") -> str:
    """URIAL prompt: the system preamble and in-context examples, then this query."""
    return _urial_template(variant) + f"\n# Query:\n```\n{query}\n```\n\n# Answer:\n```\n"


# ---------- box-side: model loading and batched generation ----------
def load_model(model_id: str, dtype: str = "bfloat16", load_in_4bit: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = hf_id(model_id)                      # "+safety" etc. selects a prompt, not weights
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                       # decoder-only batched generation
    kwargs = {"torch_dtype": getattr(torch, dtype), "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")
        kwargs.pop("torch_dtype")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tok


def render_prompt(tok, text: str, model_id: str) -> str:
    """Chat template for instruct models (Qwen3 thinking off); URIAL for the base model."""
    if is_base_model(model_id):
        return render_base(text, urial_variant(model_id))
    messages = [{"role": "user", "content": text}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_id.lower():
        kwargs["enable_thinking"] = False
    return tok.apply_chat_template(messages, **kwargs)


def generate_batch(model, tok, rendered: list[str], max_new_tokens: int) -> list[str]:
    import torch

    inputs = tok(rendered, return_tensors="pt", padding=True, truncation=True,
                 max_length=4096).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    gen = out[:, inputs.input_ids.shape[1]:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    # Base-model completions run until the closing fence; cut there.
    return [t.split("```")[0].split("# Query:")[0].strip() for t in texts]


# MMLU: 0-shot, answer read from the restricted next-token distribution over A-D (the
# method behind Meta's reported "MMLU 0-shot"). Free generation lets chat models start
# reasoning and never emit a letter within a short budget. The format instruction is
# appended at render time so the hash-frozen probe text is unchanged.
MMLU_SUFFIX = "\n\nAnswer with only the letter (A, B, C, or D) of the correct option."


def choose_letter(model, tok, rendered: list[str]) -> list[str]:
    import torch

    # Each letter may tokenize with or without a leading space; take the best variant.
    ids = {L: sorted({tok.encode(v, add_special_tokens=False)[0] for v in (L, " " + L)})
           for L in "ABCD"}
    inputs = tok(rendered, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]          # left-padded: last = next token
    scores = torch.stack([logits[:, ids[L]].max(dim=-1).values for L in "ABCD"], dim=-1)
    return ["ABCD"[i] for i in scores.argmax(dim=-1).tolist()]


def free_gpu() -> None:
    """Release cached GPU memory. Callers must `del` their own model references first --
    a function cannot drop the caller's names, so a model still referenced stays resident."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_loaded(model, tok, model_id: str, dataset: str) -> Path:
    """Generate one probe set with an already-loaded model; batched and resumable."""
    import probes

    items = probes.load(dataset)
    path = output_path(model_id, dataset)
    already = done_ids(path)
    todo = [p for p in items if p.id not in already]
    if not todo:
        print(f"{model_id} / {dataset}: all {len(items)} done", flush=True)
        return path
    maxn = MAX_NEW_TOKENS[dataset]
    done = 0
    for start in range(0, len(todo), BATCH):
        chunk = todo[start:start + BATCH]
        suffix = MMLU_SUFFIX if dataset == "mmlu" else ""
        rendered = [render_prompt(
            tok, (f"{p.context}\n\n{p.prompt}" if p.context else p.prompt) + suffix, model_id)
            for p in chunk]
        completions = (choose_letter(model, tok, rendered) if dataset == "mmlu"
                       else generate_batch(model, tok, rendered, maxn))
        for p, c in zip(chunk, completions):
            append_record(path, build_record(model_id, p, c))
        done += len(chunk)
        print(f"{model_id} / {dataset}: {done}/{len(todo)} (+{len(already)} resumed)", flush=True)
    print(f"{model_id} / {dataset}: wrote {done} records to {path}", flush=True)
    return path


def _pending(model_id: str, dataset: str) -> bool:
    import probes
    done = done_ids(output_path(model_id, dataset))
    return any(p.id not in done for p in probes.load(dataset))


def run_model(model_id: str, datasets: list[str], load_in_4bit: bool = False) -> None:
    """Load a model once, generate every listed probe set, then free the GPU."""
    if not any(_pending(model_id, ds) for ds in datasets):
        print(f"{model_id}: all {len(datasets)} sets already done", flush=True)
        return
    model, tok = load_model(model_id, load_in_4bit=load_in_4bit)
    try:
        for ds in datasets:
            run_loaded(model, tok, model_id, ds)
    finally:
        del model
        free_gpu()


def run(model_id: str, dataset: str, load_in_4bit: bool = False) -> Path:
    """Single (model, dataset) run: load, generate, free. For ad-hoc use."""
    model, tok = load_model(model_id, load_in_4bit=load_in_4bit)
    try:
        return run_loaded(model, tok, model_id, dataset)
    finally:
        del model
        free_gpu()


def _demo() -> None:
    import tempfile
    from types import SimpleNamespace

    assert model_slug("Qwen/Qwen3-8B") == "Qwen__Qwen3-8B"
    assert output_path("Qwen/Qwen3-8B", "mmlu").name == "mmlu.jsonl"
    assert is_base_model("Qwen/Qwen3-8B-Base") and not is_base_model("Qwen/Qwen3-8B")
    assert is_base_model("Qwen/Qwen3-8B-Base+safety")
    assert hf_id("Qwen/Qwen3-8B-Base+safety") == "Qwen/Qwen3-8B-Base"
    assert urial_variant("Qwen/Qwen3-8B-Base+safety") == "safety"
    assert urial_variant("Qwen/Qwen3-8B-Base") == "help"
    assert model_slug("Qwen/Qwen3-8B-Base+safety") == "Qwen__Qwen3-8B-Base+safety"

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.jsonl"
        assert done_ids(p) == set()
        probe = SimpleNamespace(dataset="ccp", id="ccp:0", group="Taiwan",
                                kind="political", prompt="q", context=None, answer=None)
        append_record(p, build_record("M", probe, "a response"))
        assert done_ids(p) == {"ccp:0"}
        rec = json.loads(p.read_text().strip())
        assert rec["model"] == "M" and rec["completion"] == "a response"
        probe2 = SimpleNamespace(**{**probe.__dict__, "id": "ccp:1"})
        todo = [x for x in (probe, probe2) if x.id not in done_ids(p)]
        assert [x.id for x in todo] == ["ccp:1"]

    print("generate self-check passed (model path not exercised; runs on the GPU box)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        run(sys.argv[1], sys.argv[2], load_in_4bit="--4bit" in sys.argv)
    else:
        _demo()
