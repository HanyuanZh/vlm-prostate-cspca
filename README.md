# Multimodal Integration in VLMs for csPCa Detection

Inference code for the paper *Exploring Multimodal Integration in Vision-Language Models
for Clinical Significant Prostate Cancer Detection*.

- **Data:** https://drive.google.com/drive/folders/12EuUeEnfmpzD5WPda4lcMLWiCg7Y9W6H?usp=sharing
- **Supplementary material:** https://drive.google.com/drive/folders/1kaG-lbmMekBJ13rVCp4zTkwG3vpGOkfL?usp=sharing

Nine vision-language models are asked for a **verbalised probability** of clinically
significant prostate cancer (csPCa), for every combination of MRI sequences and
clinical measures:

| Dataset | Imaging | Clinical measures | Classes |
|---|---|---|---|
| PI-CAI, 1,500 bpMRI exams | T2W, high-b DWI, ADC | PSA, prostate volume | csPCa yes / no |

Local checkpoints are pinned to the Hugging Face commits the study ran
(`python -m vlmbench recipe` shows them); `--checkpoint` loads another model or a
local copy instead.

| Model | Checkpoint | Runs on |
|---|---|---|
| MedGemma-4B | `google/medgemma-4b-it` | local GPU |
| MedGemma-27B | `google/medgemma-27b-it` | local GPU |
| Lingshu-7B / -32B | `lingshu-medical-mllm/Lingshu-7B`, `-32B` | local GPU |
| DeepSeek-VL2-tiny / -small | `deepseek-ai/deepseek-vl2-tiny`, `-small` | local GPU (separate env) |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B` | local GPU |
| Gemini-3.5-Flash | `gemini-3.5-flash` | Google Gemini API |
| GPT-5.6-Sol | `gpt-5.6-sol` | OpenAI API |

## Install

```bash
pip install -r requirements.txt            # MedGemma, Lingshu, Qwen, Gemini, GPT
# DeepSeek-VL2 needs transformers 4.38, so give it its own environment:
pip install -r requirements-deepseek.txt
```

API keys are read from the environment only — `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
and `OPENAI_API_KEY`. Never put keys in files inside this repository.

## Data

The processed data used in the paper are at the **Data** link above. See
[data/README.md](data/README.md) for the original dataset and the directory layout the
code expects under `--data-root`.

## Prompts

The full prompt of every response format is in [prompts/](prompts/), with the models that use it.

## Usage

```bash
# the setting of every run in the paper (checkpoint, prompt, token budget, decoding)
python -m vlmbench recipe

# print the first prompts of a run without loading a model
python -m vlmbench run -e main -m qwen36 --data-root /path/to/data --dry-run

# local model, one GPU (re-running the same command resumes)
python -m vlmbench run -e main -m qwen36 --data-root /path/to/data

# local model, 4 GPUs as 4 shards, then merged
scripts/run_local_shards.sh main qwen36 /path/to/data 4

# API model through the Batch API (about half price)
python -m vlmbench batch build  -e main -m gpt --data-root /path/to/data
python -m vlmbench batch submit -e main -m gpt --yes      # incurs charges
python -m vlmbench batch status -e main -m gpt
python -m vlmbench batch fetch  -e main -m gpt
```

`run` also works for the API models (one synchronous call per request), which is
handy for a quick `--limit 20` check.

Results go to `results/<experiment>/<model>.csv`.

### Experiments

| `-e` | What | Requests per model |
|---|---|---|
| `main` | every MRI-sequence combination × clinical-measure setting | 42,000 (7 × 4 × 1,500) |
| `placebo` | image-only inputs with silently appended duplicate, blank or noise images; prompt unchanged | 22,500 (15 × 1,500) |

The placebo experiment was run for Qwen3.6, Lingshu-32B, Gemini and GPT.

Cases lacking PSA or prostate volume (896 of the 42,000 `main` requests) are written
with `error = missing_psa` / `missing_prostate_volume` and not sent.

### Output columns

| Column | Meaning |
|---|---|
| `prob_verbalized` | stated probability of csPCa |
| `label_pred` | `prob_verbalized >= 0.5` |
| `prob_verbalized_raw` / `prob_raw` | the model's reply, verbatim (`main` / `placebo`) |
| `label_logit`, `logit_prob_1`, `label_logit_raw` | local models, `main` only: one constrained `0`/`1` token and its softmax probability |
| `true_binary` | reference label |
| `error` | why a row has no probability (`missing_*`, `unparseable_reply`, API errors) |

## Default settings

`python -m vlmbench recipe` prints the setting of every run.

* **Local models**: greedy decoding in bfloat16, up to 16 new tokens. MedGemma-27B
  uses the wording that asks for the probability only, with up to 2,048 new tokens.
* **Gemini-3.5-Flash**: temperature 1.0, default thinking level (`medium`, named
  explicitly, in the placebo experiment), JSON reply.
* **GPT-5.6-Sol**: temperature 1.0, default reasoning effort, JSON reply,
  `image_detail` low.

## Tests

```bash
pytest          # no GPU, no API calls, no dataset needed
```

## Citation

To be added on publication.
