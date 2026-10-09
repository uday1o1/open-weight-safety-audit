# Open-Weight Safety Audit

An evidence-based threat audit of how much built-in safety behavior survives when open-weight language models are downloaded and run locally.

When a model runs on the operator's own hardware, the provider's server-side safeguards are gone and the safety behavior in the weights becomes the main line of defense. That behavior can be removed: refusal is largely mediated by a single direction in the model's activations, and public tools automate its ablation. This project measures what survives that removal, what it costs in capability, and which controls reduce the resulting risks.

## Research questions

1. How much harmful-request refusal and politically sensitive refusal survives refusal-direction ablation, and at what cost to general capability?
2. Is politically sensitive refusal introduced mainly by post-training alignment rather than by pretraining?
3. Do the effects replicate across developers?
4. Which controls reduce the resulting risks, and how reliably, measured by false-positive rate, can tampered models be detected?

## Models

| Model | Role |
|---|---|
| Qwen3-8B | System under audit |
| Qwen3-8B-Base | Pretraining reference (in-context alignment) |
| GLM-4-9B-0414 | Second developer |
| Llama-3.1-8B-Instruct | Cross-origin comparison |

## Method

- **Intervention:** difference-in-means refusal-direction ablation (Arditi et al., 2024); Heretic as a secondary method.
- **Safety:** HarmBench, StrongREJECT.
- **Over-refusal:** XSTest.
- **Politically sensitive refusal:** promptfoo CCP-sensitive prompt set.
- **Capability:** MMLU, GSM8K.
- **Tamper detection:** behavioral canary check, false-positive rate on legitimate fine-tunes.
- **Statistics:** Wilson intervals, Newcombe paired differences, McNemar tests with Bonferroni correction.
- **Threat mapping:** MITRE ATLAS.

## Scope and safety

All experiments run offline on self-hosted, openly licensed weights. Modified models are never uploaded or shared, and are deleted after the study. Harmful outputs stay on the evaluation machine; only aggregate rates are published here.

## Status

In progress.
