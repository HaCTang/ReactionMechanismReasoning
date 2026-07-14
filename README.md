# ReactionMechanismReasoning

Code and benchmark assets for **Learning Mechanistic Reasoning for Chemical Reactions with Large Language Models** (COLM 2026 submission).

Trimmed for reproducibility: FukuyamaBench (JSON), benchmark construction, annotation → SFT → GSPO (**paper recipe only**), and evaluation.

## Layout

```
fukuyama_bench/     FukuyamaBench (319 cases) + construction scripts
annotation/         Gemini CoT annotation → TRL messages
train/              SFT (paper: SFT HQ)
rl/                 GSPO v6 only (paper SOTA)
eval/               Pathway inference + pass@k
data_prep/          Build Haocheng1/mech-infer-train-v1 (+ seeds)
```

Case IDs encode partitions: **A**=78, **B**=131, **C**=110.

## Datasets

| Purpose | Hugging Face | Notes |
|---------|--------------|-------|
| Paper SFT | [`johnny-w/flower`](https://huggingface.co/datasets/johnny-w/flower) `sft_hq` | 3,650 |
| Paper RL | `johnny-w/flower` `rl_v6` | 2,000 → verl parquet |
| Diverse pre-annotation | [`Haocheng1/mech-infer-train-v1`](https://huggingface.co/datasets/Haocheng1/mech-infer-train-v1) | 10k trajectories |

```python
from datasets import load_dataset
sft = load_dataset("johnny-w/flower", "sft_hq", split="train")
rl  = load_dataset("johnny-w/flower", "rl_v6")
```

## Paper recipe

```
Qwen3-30B-A3B-Instruct
  → SFT on sft_hq (full LM loss, 3 ep)   → qwen3-30b-pathway-hq
  → GSPO v6 on rl_v6 (3 ep, reward v5)   → qwen3-30b-gspo-v6-ep3
```

Reported: **FukuyamaBench Set A exact pass@1 ≈ 8.3%**.

```bash
pip install -r requirements.txt
pip install -r train/requirements_hq_train_repro.txt   # pinned SFT stack

bash train/run_sft_hq.sh

# RL: install verl, then
git submodule update --init verl
# optional on CPU-RAM-tight nodes:
#   cd verl && git apply ../patches/verl_fsdp_offload_oom_fix.patch && cd ..
# Write data/verl_v6/{train,validation}.parquet from flower:rl_v6 (or rl/prepare_verl_data.py)
python rl/export_rl_v6_parquet.py
bash rl/run_gspo.sh
```

Use **vLLM 0.14.x** with verl (newer versions can break weight sync).

## FukuyamaBench construction

Given reaction / mechanism figures from the workbook:

```bash
python fukuyama_bench/build/RxnIM.py
export GOOGLE_API_KEY=...
python fukuyama_bench/build/Gemini_mechanism_test.py
python fukuyama_bench/build/find_unbalanced_mechanisms.py
python fukuyama_bench/build/renew_mechanisms.py
python fukuyama_bench/build/renewfigure.py
python fukuyama_bench/build/Gemini_ckpt_generator.py
python fukuyama_bench/build/mapping_figures.py
```

Shipped cases (`fukuyama_bench/mechanisms/*/`) contain `mechanism.json`, `rxn.json`, `mapped_rxn.json`, `ckpt.txt` only (no PNGs).

## Evaluation

```bash
python eval/run_infer_pathway.py \
  -m vllm --vllm_url http://HOST:8000/v1 \
  --vllm_model checkpoints/qwen3-30b-gspo-v6-ep3-hf \
  -p A -w 3 --sample_size 8 \
  --mechanisms_dir fukuyama_bench/mechanisms

python eval/eval_infer_pathway.py \
  --mechanisms_dir fukuyama_bench/mechanisms \
  --result_file outputs/eval/pathway/<result>.json
```

Primary metric: unbiased exact-match **pass@1**.

## Optional: rebuild mech-infer-train-v1

Prefer downloading from HF. To rebuild from `data_prep/sources/`:

```bash
python data_prep/build_training_dataset.py --target 10000
python annotation/annotate_reaction.py --input ... --output ... \
  --prompts annotation/pathway_annotation_prompts.yaml
python annotation/convert_to_train.py --input ... --output ...
```

## Secrets

Do not commit keys. Use env vars or local `*.key` files (gitignored):

- `GOOGLE_API_KEY` / `gemini.key`
- `HF_TOKEN` / `hf_token.key`
- `OPENAI_API_KEY`, `SILICONFLOW_API_KEY` (optional baselines)

## Citation

Please cite the COLM 2026 paper *Learning Mechanistic Reasoning for Chemical Reactions with Large Language Models*.
