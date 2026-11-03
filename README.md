# Open-Weight Safety Audit

An evidence-based audit of the safety behavior that ships inside open-weight language models,
and of the supply-chain checks that protect a local deployment.

When a model runs on the operator's own hardware, the provider's server-side safeguards are
gone: the safety behavior in the weights and the integrity of the downloaded files become the
main line of defense. This project measures both on three 8B- and 9B-class models from three developers.

## Questions

1. How often do the shipped models refuse harmful requests, over-refuse safe ones, and refuse
   or deflect politically sensitive prompts, and how does this differ across developers?
2. Is politically sensitive refusal introduced by post-training, or already present after
   pretraining?
3. How reliably can supply-chain tampering (checksum mismatch, altered chat template, unsafe
   serialization, an exposed local server) be detected, and at what false-positive cost?
4. Can those detectors be turned into a control that stops a tampered release before it is
   loaded or served?
5. How much of the remaining harmful compliance does an open safety classifier in front of and
   behind the model remove, and what does it cost on safe requests?

## Models

| Model | Role |
|---|---|
| Qwen3-8B | System under audit |
| Qwen3-8B-Base | Pretraining reference, prompted with URIAL in-context alignment |
| GLM-4-9B-0414 | Second developer |
| Llama-3.1-8B-Instruct | Cross-origin comparison (gated on Hugging Face) |

The audited models run from the publishers' unquantized BF16 releases, unmodified. Every model,
judge and classifier loads at a pinned Hugging Face revision (`MODEL_REV` in `src/generate.py`).

## Method

| Measure | Probe set | Labeled by |
|---|---|---|
| Harmful-request refusal | HarmBench standard + contextual (300), StrongREJECT (313) | HarmBench Llama-2-13B classifier (4-bit NF4); StrongREJECT refusal rubric |
| Over-refusal | XSTest safe prompts (250) | over-refusal rubric (`src/judges.py`) |
| Political refusal / deflection | promptfoo CCP-sensitive prompts (1,360; topics the Chinese Communist Party treats as sensitive) | 3-class rubric (answer / deflection / refusal); checked against 200 reference labels assigned by a separate general-purpose LLM (not the judge model) that did not see the judge's labels: Cohen's κ = 0.95 for answer vs. refuse-or-deflect (the reported rate) and 0.49 for the three-way split, which none of the reported rates uses |
| Capability | MMLU (1,000, 0-shot, restricted A–D next token), GSM8K (500) | answer extraction |
| Safety-classifier control | the HarmBench, StrongREJECT and XSTest prompts and each model's answers | Llama Guard 3 8B as input filter (prompt) and output filter (prompt + answer); harmful compliance from the audit labels |
| Tamper detection | clean vs tampered artifacts; the real chat templates of the audited models (pinned) plus four held-out publishers' templates | the detectors in `src/supplychain.py`; end to end through the admission gate in `src/gate.py` |

- Decoding is greedy. Instruction-tuned models use their own chat template (Qwen3 with thinking off).
- The base model is prompted with [URIAL](https://github.com/Re-Align/URIAL) in-context
  alignment (`inst_1k_v4.help.txt`: a helpful-assistant preamble and answered examples, with no
  safety instruction and no refusal example), so no refusal is instructed or demonstrated by the
  prompt. A second condition uses URIAL's `inst_1k_v4.txt`, whose preamble includes safety
  instructions (including "you can reject to answer some controversial topics") and whose
  examples include one refusal, to measure how much a safety-oriented prompt moves refusal.
- The rubric judges (StrongREJECT, XSTest, political) run on Qwen2.5-7B-Instruct. The HarmBench
  classifier counts an answer as a refusal whenever it does not judge it an instance of the
  harmful behavior, so a vague or off-target answer also counts as refused: the HarmBench rate
  measures non-compliance, not only explicit refusal. Judge outputs that cannot be parsed
  (8 StrongREJECT labels: 1 Qwen3, 3 GLM, 4 Llama; none in other sets) count as not refused.
- Statistics: Wilson 95% intervals per rate; base-vs-instruct differences on the same prompts
  use Newcombe paired intervals and exact McNemar tests, Bonferroni-adjusted over the 8 paired
  tests (4 probe sets × 2 base prompts; α = 0.00625). Intervals are unadjusted 95%.
- Scope: single-turn English prompts sent as written (no jailbreak rewriting); one greedy run per
  model and condition; the political judge is the one checked against reference labels; URIAL on
  the released base checkpoint stands in for pretraining-only behavior.

## Results

Rates with Wilson 95% intervals and per-category label counts are in `results/rates.json`;
paired comparisons in `results/base_vs_instruct.json`; per-item labels (id and label, no text)
in `results/labels/`; the political-judge validation set in `results/judge_validation_ccp.jsonl`;
the safety-classifier control in `results/guard.json` (per-item verdicts in `results/guard/`);
tamper detection in `results/supplychain_fpr.json`.

| Model | HarmBench refusal | StrongREJECT refusal | XSTest over-refusal | CCP refuse + deflect | MMLU | GSM8K |
|---|---|---|---|---|---|---|
| Qwen3-8B | 75.0% | 96.8% | 7.2% | 62.2% | 69.3% | 88.8% |
| GLM-4-9B-0414 | 69.7% | 93.3% | 14.0% | 48.8% | 75.5% | 81.6% |
| Llama-3.1-8B-Instruct | 77.0% | 96.8% | 10.4% | 57.9% | 63.4% | 80.2% |
| Qwen3-8B-Base (URIAL) | 89.3% | 98.4% | 16.0% | 62.6% | – | – |
| Qwen3-8B-Base (URIAL safety prompt) | 93.3% | 99.4% | 27.2% | 79.9% | – | – |

Qwen3-8B minus its base (URIAL), same prompts:

| Probe set | Difference | 95% CI | McNemar p | Significant |
|---|---|---|---|---|
| HarmBench | −14.3 pts | [−19.1, −9.8] | 1.8e-9 | yes |
| StrongREJECT | −1.6 pts | [−4.4, +1.0] | 0.30 | no |
| XSTest | −8.8 pts | [−13.7, −4.2] | 3.1e-4 | yes |
| CCP | −0.4 pts | [−2.6, +1.8] | 0.79 | no |

Cross-model differences in the first table are descriptive; each rate's Wilson interval is in
`results/rates.json`. Against the base prompted with the URIAL safety prompt instead, Qwen3-8B is
lower on HarmBench (−18.3 pts), XSTest (−20.0 pts) and CCP (−17.7 pts), all significant. The
safety-oriented prompt alone raises the base model's CCP refuse-or-deflect rate from 62.6% to
79.9% (+17.4 pts).

Llama Guard 3 as an input and output filter around each model (`results/guard.json`). Harmful
compliance is the share of HarmBench prompts whose answer the HarmBench classifier judges an
instance of the harmful behavior (the complement of refusal):

| Model | HarmBench harmful compliance | + output filter | + input and output filter | Harmful answers caught by output filter | XSTest helpful | + both filters |
|---|---|---|---|---|---|---|
| Qwen3-8B | 25.0% | 3.7% | 1.3% | 64/75 | 92.8% | 91.6% |
| GLM-4-9B-0414 | 30.3% | 5.0% | 2.3% | 76/91 | 86.0% | 85.2% |
| Llama-3.1-8B-Instruct | 23.0% | 4.3% | 2.0% | 56/69 | 89.6% | 89.2% |

The input filter flags 96.3% of HarmBench and 97.1% of StrongREJECT prompts and 7 of 250 safe
XSTest prompts (2.8%); the output filter blocks 1 of 671 helpful XSTest answers across the three
models. With both filters, harmful compliance on HarmBench falls by 91–95% relative to the unfiltered
model (Llama 91.3%, GLM 92.3%, Qwen3 94.7%), at a cost of 0.4–1.2 points of XSTest helpfulness.
The 17 harmful HarmBench answers that still get through are all cybercrime (8), illegal-activity
(5) or misinformation (4) behaviors. On StrongREJECT few answers are judged compliant
(1.9–5.8%); the output filter passes 28 of those 33 answers, and with the input filter only 1–3 per
model remain. The labels used for the filter measurement match the audit labels on all 2,589
items (`labels_match_audit` in `results/guard.json`).

Tamper detection and the admission gate (`results/supplychain_fpr.json`):

| Detector | Detected | False positives |
|---|---|---|
| Pickle code-execution scan | 100/100 gadgets (4 gadgets × 5 placements × 5 protocols) | 0/200 clean state-dict pickles |
| Checksum vs pinned SHA-256 | 200/200 one-bit flips, truncations, appends | 0/200 |
| Template render diff, audited models' templates | 15/15 injections | 0/12 harmless edits |
| Template render diff, held-out templates (4 publishers) | 16/17 injections | 0/16 harmless edits |
| Template text diff (baseline) | 15/15 | 12/12 harmless edits |
| Exposed inference server probe | 20/20 bound to all interfaces | 0/20 bound to loopback |

A plain text diff flags every harmless edit, so the template check renders each template on
probe conversations covering its main branches (system prompt, multi-turn, tools, no-thinking) and
compares the resulting prompts. The one held-out miss sits in a document-retrieval branch no
probe reaches; inside the gate the same edit is still caught by the checksum.

The gate (`src/gate.py`) pins a trusted release and admits a copy only if every file hash,
pickle scan and template render matches; otherwise it blocks it and, with `--quarantine`, moves it to a quarantine directory. Over 20
synthetic pinned releases (random weights, the audited models' real chat templates, and a clean
pickle checkpoint as in Llama) it admitted 20/20 clean copies and blocked 20/20 for each attack: modified
weights, a smuggled pickle, an injected template, and a payload inside a pinned pickle file.
Every audited release publishes a SHA-256 for 100% of its weight files; Llama-3.1-8B-Instruct
also ships a pickle checkpoint (`original/consolidated.00.pth`), which the gate scans on every
check.

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
  guard.py             Llama Guard 3 input/output filter: harmful compliance removed vs safe requests blocked (GPU)
  supplychain.py       checksum / pickle scan / template render diff / exposure probe
  supplychain_eval.py  detection and false-positive rates for the detectors and the gate
  gate.py              admission gate: pin a release, block and quarantine tampered copies
  demo.py              live walkthrough: pin, attack, detect, block
box/                   setup and detached-run scripts for a rented GPU
manifests/             frozen probe sets: prompt hashes only, no prompt text
results/               rates, per-item labels (no text), detector results (the only published output)
```

Completions are never committed: they can contain harmful text, and the harmful-request
completions stay on the GPU machine.

## Reproducing

**CPU (analysis and tamper detection).** Every module except the orchestrator and the
walkthrough (`demo.py`) carries a CPU self-check (the GPU modules check their pure parts):

```bash
pip install -r requirements.txt
for m in stats probes scoring supplychain judges generate aggregate label guard gate; do
  PYTHONPATH=src python src/$m.py || break
done
PYTHONPATH=src python src/supplychain_eval.py   # writes results/supplychain_fpr.json
PYTHONPATH=src python src/demo.py               # live attack -> detect -> block walkthrough
```

`supplychain_eval.py` and `demo.py` download the pinned chat templates on first use. The Llama
chat template is gated, so `supplychain_eval.py` needs a Hugging Face token with Llama
access in `~/.cache/huggingface/token`.

Guarding a real deployment:

```bash
PYTHONPATH=src python src/gate.py pin   ~/models/Qwen3-8B qwen3.manifest.json   # once, on a trusted download
PYTHONPATH=src python src/gate.py check ~/models/Qwen3-8B qwen3.manifest.json \
    --quarantine ~/quarantine && vllm serve ~/models/Qwen3-8B
```

**GPU (model results).** One CUDA GPU (the reported results ran on an A100 40GB) with a CUDA
PyTorch image and a Hugging Face token in `~/.cache/huggingface/token` (Llama access must be granted on its model page):

```bash
git clone https://github.com/uday1o1/open-weight-safety-audit.git ~/open-weight-safety-audit
cd ~/open-weight-safety-audit
bash box/setup.sh
AUDIT_BATCH=16 AUDIT_JUDGE_BATCH=8 bash box/run.sh   # detached; log in ~/audit-run.log
```

`run.sh` runs `src/run_audit.py all`: it generates completions for each model on its probe sets
(the base-model conditions skip MMLU and GSM8K), labels them, and writes `results/`. Generation and judging skip records already written, so an
interrupted run resumes where it stopped. The reported results used these batch sizes on an
A100 40GB. Without them, `AUDIT_BATCH` defaults to 8 (`run.sh`) and the judge batch to 4 for the
HarmBench classifier and 8 for Llama Guard; on a smaller card lower both, and greedy outputs can
then differ slightly.

`AUDIT_BATCH=16 AUDIT_JUDGE_BATCH=8 bash box/run.sh guard` runs the safety-classifier control
(`src/guard.py run`): it generates the three instruction-tuned models' answers on the harmful
and safe sets (reusing any already in the output directory), labels them with the audit judges, screens prompts and answers with Llama Guard 3
8B (access must be granted on its model page), and writes `results/guard.json`. Greedy
decoding at the audit's batch sizes is meant to reproduce the audit's answers;
`labels_match_audit` in `results/guard.json` counts the items whose label matches the audit
label.

Every data source and model is pinned to a revision, and `manifests/*.json` freeze each probe
set by prompt hash (a re-fetch must match; HarmBench contexts are fixed by the commit pin) and
fix the MMLU/GSM8K subsets with a seed. With greedy decoding, a re-run should reproduce the
rates closely; small differences can come from GPU kernels, batch size and library versions.

## Scope and safety

Every model the code runs (audited models, judges and classifier) is self-hosted from openly
available weights (Apache-2.0, MIT and the Llama 3.1 Community License); the code calls no hosted
model API. No model's safety behavior is modified. Harmful outputs stay on the evaluation
machine; only rates and per-item labels (ids and labels, no text) are published.
