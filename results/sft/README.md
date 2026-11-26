# SFT-only results

Qwen3-30B-A3B-Instruct after SFT only (no RL), on FukuyamaBench Set A (78 cases).
Inference: 8 samples/case, temperature 0.7, thinking off, max_tokens 14000.

| Model | Exact pass@1 | pass@3 | pass@5 | pass@8 |
|---|---|---|---|---|
| Qwen3-30B-A3B-Instruct (base) | 6.9% | 8.6% | 9.0% | 9.0% |
| **SFT HQ** (paper SFT base) | 2.7% | — | — | 5.1% |
| SFT HQ v2 | 2.4% | 5.4% | 7.7% | 10.3% |
| SFT HQ → GSPO v6 ep3 (paper) | 8.3% | 12.1% | 13.9% | 15.4% |

SFT alone is below the base model on exact match. The gain comes from GSPO.
These numbers come from the evaluator as it was when the runs were made, which was
before the label-aware checkpoint matching in `eval/matching.py`.
To re-score with the current code, use `eval/eval_infer_pathway.py --result_file <outputs_A_s8.json>`.

## Recipes

| | SFT HQ | SFT HQ v2 |
|---|---|---|
| Data | `johnny-w/flower:sft_hq`, 3,650 = 2,353 FlowER (≤25 heavy atoms, 3–10 steps, valid SMILES, dedup) + 1,297 external | SFT HQ + 124 Gemini-3.1-Pro named-reaction mechanisms (16 families) = 3,774 |
| Training | full FT, full LM loss, LR 1e-5 cosine, warmup 100, bsz 16, 3 ep (687 steps), max_len 16384 (`train/run_sft_hq.sh`) | same, but `assistant_only_loss`, LR 1e-5 constant w/ warmup 50, 3 ep (708 steps) |

## Files

- `sft_hq/outputs_A_s8.json`: raw generations (each case has `predictions` and `ground_truth_pathway`)
- `sft_hq/train.log`: full training log (final train_loss 0.164)
- `sft_hq_v2/outputs_A_s8.json`: raw generations
- `sft_hq_v2/eval_report.txt`, `eval_report_detailed.json`: pass@k report

The training log for HQ v2 was not kept.
