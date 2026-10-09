# Open-Weight Safety Audit

An evidence-based threat audit of how much built-in safety behavior survives when open-weight language models are downloaded and run locally.

When a model runs on the operator's own hardware, the provider's server-side safeguards are gone and the safety behavior in the weights becomes the main line of defense. That behavior can be removed: refusal is largely mediated by a single direction in the model's activations, and public tools automate its ablation. This project measures what survives that removal, what it costs in capability, and which controls reduce the resulting risks.

## Research questions

1. How much do the shipped models refuse harmful requests, over-refuse safe ones, and refuse or deflect politically sensitive prompts, and how does this differ across developers?
2. Is politically sensitive refusal introduced mainly by post-training alignment rather than by pretraining?
3. How reliably can supply-chain tampering (checksum mismatch, altered chat template, unsafe serialization, an exposed local server) be detected, and at what false-positive cost?
4. Which technical and organizational controls reduce the resulting risks?

Removal of safety behavior from open weights (abliteration) is the top residual risk in
the register; it is assessed from published work rather than reproduced here.

## Models

| Model | Role |
|---|---|
| Qwen3-8B | System under audit |
| Qwen3-8B-Base | Pretraining reference (in-context alignment) |
| GLM-4-9B-0414 | Second developer |
| Llama-3.1-8B-Instruct | Cross-origin comparison |

## Method

- **Harmful-request refusal:** HarmBench (standard + contextual), StrongREJECT, judged by the HarmBench Llama-2 classifier and the StrongREJECT rubric.
- **Over-refusal:** XSTest safe prompts.
- **Politically sensitive refusal:** promptfoo CCP-sensitive prompt set, judged by a 3-class rubric (answer / deflection / refusal) validated against hand labels (Cohen's κ ≥ 0.6).
- **Capability:** MMLU and GSM8K subsets, scored by answer extraction.
- **Pretraining vs post-training (RQ2):** Qwen3-8B-Base under in-context alignment compared to Qwen3-8B on the same prompts.
- **Tamper detection (RQ3):** checksum, pickle code-execution scan, chat-template diff, and an off-loopback inference-server probe, measured on clean and tampered samples with a false-positive rate.
- **Statistics:** Wilson intervals, Newcombe paired differences, McNemar tests, Bonferroni correction (family of 20, α = 0.0025).
- **Threat mapping:** MITRE ATLAS.

## Models

| Model | Role |
|---|---|
| Qwen3-8B | System under audit |
| Qwen3-8B-Base | Pretraining reference (in-context alignment) |
| GLM-4-9B-0414 | Second developer |
| Llama-3.1-8B-Instruct | Cross-origin comparison (gated on Hugging Face) |

## Repository layout

```
src/
  stats.py        Wilson / Newcombe / McNemar / Cohen's kappa
  probes.py       loaders for the six evaluation sets (pinned sources)
  scoring.py      MMLU / GSM8K answer extraction
  judges.py       judge prompts + output parsers (refusal, political)
  generate.py     resumable completion harness (runs on the GPU box)
  aggregate.py    labeled completions -> reported rates and comparisons
  supplychain.py  checksum / pickle scan / template diff / exposure probe
manifests/        frozen probe sets: prompt hashes only, no prompt text
results/          aggregate rates (committed output)
```

`outputs/`, `data/`, and model weights are gitignored: completions may contain harmful
text and stay on the evaluation machine; only aggregate rates in `results/` are published.

## Running

Analysis host (CPU). Every pure module ships a self-check:

```bash
pip install -r requirements.txt
for m in stats probes scoring supplychain judges generate aggregate; do
  PYTHONPATH=src python src/$m.py || exit 1
done
python src/probes.py build        # (re)build the frozen manifests
```

GPU box (one rented 24–48 GB GPU). Needs `torch` and `transformers`, plus a Hugging Face
token in `~/.cache/huggingface/token`:

```bash
PYTHONPATH=src python src/generate.py Qwen/Qwen3-8B harmbench   # resumable per model/set
```

Then copy `outputs/` back and run `python src/aggregate.py run` to write `results/`.

## Reproducibility

Every data source is pinned to a commit; `manifests/*.json` freeze each evaluated set by
prompt hash (re-fetch and the hashes must match) and fix the two capability subsets with a
seed. Decoding is greedy. Judge prompts are vendored verbatim with their source commit.

## Scope and safety

All experiments run offline on self-hosted, openly licensed weights. No model has its
safety behavior modified in this repository. Harmful outputs stay on the evaluation
machine; only aggregate rates are published here.

## Status

In progress — analysis library complete and tested; model runs pending the GPU box.
