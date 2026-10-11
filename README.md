# Open-Weight Safety Audit

An evidence-based audit of the safety behavior that ships inside open-weight language models,
and of the supply-chain checks that protect a local deployment.

When a model runs on the operator's own hardware, the provider's server-side safeguards are
gone: the safety behavior in the weights and the integrity of the downloaded files become the
main line of defense. This project measures both on three 8–9B models from three developers.

## Questions

1. How often do the shipped models refuse harmful requests, over-refuse safe ones, and refuse
   or deflect politically sensitive prompts, and how does this differ across developers?
2. Is politically sensitive refusal introduced by post-training, or already present after
   pretraining?
3. How reliably can supply-chain tampering (checksum mismatch, altered chat template, unsafe
   serialization, an exposed local server) be detected, and at what false-positive cost?

## Models

| Model | Role |
|---|---|
| Qwen3-8B | System under audit |
| Qwen3-8B-Base | Pretraining reference, prompted with URIAL in-context alignment |
| GLM-4-9B-0414 | Second developer |
| Llama-3.1-8B-Instruct | Cross-origin comparison (gated on Hugging Face) |

All weights are the publishers' unquantized BF16 releases, run unmodified.

## Method

| Measure | Probe set | Labeled by |
|---|---|---|
| Harmful-request refusal | HarmBench standard + contextual (300), StrongREJECT (313) | HarmBench Llama-2-13B classifier; StrongREJECT refusal rubric |
| Over-refusal | XSTest safe prompts (250) | refusal judge |
| Political refusal / deflection | promptfoo CCP-sensitive prompts (1,360) | 3-class rubric (answer / deflection / refusal); validated on a 200-item blind reference set (Cohen's κ = 0.95, answer vs. refuse-or-deflect) |
| Capability | MMLU (1,000, 0-shot, restricted A–D next token), GSM8K (500) | answer extraction |
| Tamper detection | synthetic clean vs tampered artifacts | the detectors in `src/supplychain.py` |

- Decoding is greedy. Instruction-tuned models use their own chat template (Qwen3 with thinking off).
- The base model is prompted with URIAL (`inst_1k_v4.help.txt`, no safety instruction), so any
  refusal comes from the model itself. A second condition uses `inst_1k_v4.txt`, which adds a
  "may decline controversial topics" instruction, to measure how much one prompt line moves refusal.
- The rubric judges (StrongREJECT, XSTest, political) run on Qwen2.5-7B-Instruct.
- Statistics: Wilson 95% intervals per rate; base-vs-instruct differences on the same prompts
  use Newcombe paired intervals and exact McNemar tests, Bonferroni-adjusted over the 8 paired
  tests (α = 0.00625).

## Results

Rates with Wilson 95% intervals and per-category label counts are in `results/rates.json`;
paired comparisons in `results/base_vs_instruct.json`; per-item labels (id and label, no text)
in `results/labels/`; the political-judge validation set in `results/judge_validation_ccp.jsonl`;
tamper detection in `results/supplychain_fpr.json`.

| Model | HarmBench refusal | StrongREJECT refusal | XSTest over-refusal | CCP refuse + deflect | MMLU | GSM8K |
|---|---|---|---|---|---|---|
| Qwen3-8B | 75.0% | 96.8% | 7.2% | 62.2% | 69.3% | 88.8% |
| GLM-4-9B-0414 | 69.7% | 93.3% | 14.0% | 48.8% | 75.5% | 81.6% |
| Llama-3.1-8B-Instruct | 77.0% | 96.8% | 10.4% | 57.9% | 63.4% | 80.2% |
| Qwen3-8B-Base (URIAL) | 89.3% | 98.4% | 16.0% | 62.6% | – | – |
| Qwen3-8B-Base (URIAL + safety line) | 93.3% | 99.4% | 27.2% | 79.9% | – | – |

Qwen3-8B minus its base (URIAL), same prompts:

| Probe set | Difference | 95% CI | McNemar p | Significant |
|---|---|---|---|---|
| HarmBench | −14.3 pts | [−18.9, −9.6] | 1.8e-9 | yes |
| StrongREJECT | −1.6 pts | [−4.2, +1.2] | 0.30 | no |
| XSTest | −8.8 pts | [−13.4, −3.9] | 3.1e-4 | yes |
| CCP | −0.4 pts | [−2.6, +1.8] | 0.79 | no |

Tamper detection: the pickle scanner flagged 100/100 code-executing checkpoints with 0/200
false positives on clean PyTorch-style checkpoints; the chat-template diff flagged 8/8 altered
templates with 0/8 false positives.

Against the base prompted with the safety line instead, Qwen3-8B is lower on HarmBench (−18.3 pts),
XSTest (−20.0) and CCP (−17.7), all significant: one generic "may decline controversial topics"
sentence moves the base model's political refusal by about 18 points.

## Repository layout

```
src/
  stats.py             Wilson / Newcombe / McNemar / Cohen's kappa
  probes.py            loaders for the six probe sets (pinned sources)
  scoring.py           MMLU / GSM8K answer extraction
  judges.py            judge prompts and output parsers
  generate.py          batched, resumable generation (GPU)
  label.py             capability scoring and judge labeling (GPU)
  aggregate.py         labeled completions -> rates and paired comparisons
  run_audit.py         generate -> label -> aggregate, resumable end to end
  supplychain.py       checksum / pickle scan / template diff / exposure probe
  supplychain_eval.py  detection and false-positive rates for the detectors
box/                   setup and detached-run scripts for a rented GPU
manifests/             frozen probe sets: prompt hashes only, no prompt text
results/               aggregate results (the only published output)
```

Completions are never committed: they can contain harmful text and stay on the GPU machine.

## Reproducing

**CPU (analysis and tamper detection).** Every pure module carries a self-check:

```bash
pip install -r requirements.txt
for m in stats probes scoring supplychain judges generate aggregate label; do
  PYTHONPATH=src python src/$m.py || exit 1
done
PYTHONPATH=src python src/supplychain_eval.py   # writes results/supplychain_fpr.json
```

**GPU (model results).** One 24 GB GPU (an RTX 3090 was used) with a CUDA PyTorch image and a
Hugging Face token in `~/.cache/huggingface/token` (Llama access must be granted on its model page):

```bash
git clone <this repo> ~/open-weight-safety-audit && cd ~/open-weight-safety-audit
bash box/setup.sh
bash box/run.sh            # detached; log in ~/audit-run.log
```

`run.sh` runs `src/run_audit.py all`: it generates completions for every model and probe set,
labels them, and writes `results/`. Every phase skips finished work, so an interrupted run
resumes where it stopped. Each model condition takes about 4 hours on an RTX 3090, generation plus judging.

Every data source is pinned to a commit, and `manifests/*.json` freeze each probe set by
prompt hash (a re-fetch must match) and fix the MMLU/GSM8K subsets with a seed. With greedy
decoding, a re-run should reproduce the rates closely; small differences can come from GPU
kernels and library versions.

## Scope and safety

All experiments run offline on self-hosted, openly licensed weights. No model's safety
behavior is modified. Harmful outputs stay on the evaluation machine; only aggregate rates are
published.
