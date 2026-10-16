# Privacy Risk Classification using LLM-as-a-Judge and Distillation

Code for the paper **"Privacy Risk Classification using LLM as-a-judge"** (anonymous submission).

The pipeline labels text as **LOW / MEDIUM / HIGH privacy risk** following NIST SP 800-122, in four stages:

1. Three open LLMs act as independent judges and label about 12,000 texts drawn from three public corpora.
2. A majority vote (at least 2 of 3 judges agree) gives one label per text, and a class-balanced training set is built.
3. A ModernBERT-base student is fine-tuned on that set (distillation of the judge consensus).
4. The student, the judges, and baselines are evaluated against a 250-text human-annotated survey, with bootstrap confidence intervals and McNemar's test.

```
raw corpora ──► sourcing ──► 3 LLM judges ──► majority vote ──► balance ──► ModernBERT student
(Nemotron-PII,   (uniform     (GPT-OSS-120B,   (>=2 of 3 agree,  (2,000 per    (fine-tune on Modal)
 Enron, WNUT-17)  random       Qwen3-Next-80B,  others dropped)    class)             │
                  draw)        Llama-4-Scout)                                       ▼
                                                              human survey (250) ──► metrics, bootstrap CIs,
                                                                                     Presidio + always-LOW baselines
```

## Contents

- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Data](#data)
- [Reproducing the paper, step by step](#reproducing-the-paper-step-by-step)
- [Settings used in the paper](#settings-used-in-the-paper)
- [Reference results](#reference-results)
- [Mapping paper tables to code](#mapping-paper-tables-to-code)
- [Notes and limitations](#notes-and-limitations)

## Repository layout

```
config.py                 all paths, judge models, sampling and GPU settings
prompts.py                the single zero-shot judge prompt (identical for all judges and corpora)
parsing.py                parses "Label: LOW/MEDIUM/HIGH" from judge output
sourcing.py               stage 1: streaming, uniform random draw from each raw corpus
modal_app.py              Modal GPU classes (one per judge) and run_judge_job (detached orchestrator)
judges_runner.py          local-side loop that calls one judge at a time
checkpoint_io.py          resumable .jsonl checkpoints, used locally and remotely
voting.py                 stage 2: majority vote, Fleiss' kappa, dropped-row log
balance.py                stage 2b: class balancing and class weights
pipeline.py               CLI orchestrator for stages source / judge / vote / balance
train_modernbert.py       stage 3: ModernBERT fine-tuning on Modal
test_modernbert.py        stage 3b: predictions and metrics for any labeled CSV, on Modal
presidio_baseline_eval.py classical baseline (Presidio + spaCy) and always-LOW baseline
bootstrap_ci_survey.py    bootstrap CIs, paired gains over always-LOW, McNemar's test
config_small.py,          small-judge counterparts of config.py and modal_app.py
modal_app_small.py        (small-judge experiment of the paper)
baseline_results/         saved output of presidio_baseline_eval.py
bootstrap_results/        saved output of bootstrap_ci_survey.py
data/                     working folders (see Data)
```

The folders `scripts/` and `out/` hold auxiliary material that is not used by the commands in this README.

## Setup

Requirements: Python 3.11 or newer on your machine, a [Modal](https://modal.com) account (all GPU work runs on Modal), and a Hugging Face token (used for model downloads and to avoid rate limits).

```bash
# 1. Local environment (orchestration, voting, balancing)
pip install -r requirements.txt            # modal, pandas, numpy, statsmodels

# 2. Extra packages for the evaluation scripts
pip install scikit-learn scipy matplotlib
pip install spacy presidio-analyzer==2.2.364
python -m spacy download en_core_web_lg

# 3. Modal
modal token new                            # one-time: links this machine to your Modal account
modal secret create huggingface-secret HF_TOKEN=<your_hf_token>
```

GPU environments are built by Modal from the image definitions in `modal_app.py`, `train_modernbert.py` and `test_modernbert.py`. Package versions inside those images are not pinned, apart from `triton==3.4` and `kernels==0.16.0` for the GPT-OSS image.

Modal volumes `privacy-judge-data` (data, checkpoints, trained models) and `privacy-judge-model-cache` (downloaded weights) are created automatically on first use.

## Data

The repository does not redistribute the raw corpora. Download them and place them as shown.

| Corpus | Role | Where to get it | Place at |
|---|---|---|---|
| Nemotron-PII | training source | Hugging Face dataset `nvidia/Nemotron-PII` | `data/raw/nemotron.csv` |
| Enron Email | training source | Klimt and Yang, ECML 2004 | `data/raw/enron.csv` |
| WNUT-17 | training source | Derczynski et al., WNUT 2017 | `data/raw/wnut17.csv` |
| Human survey (250 texts) | evaluation only | Meisenbacher et al., arXiv:2508.12158 | `data/sourced/survey_with_label3.csv` |

- The text column of each raw CSV is auto-detected, or set it in `TEXT_COLUMN_HINTS` in `config.py`.
- The survey file needs the columns `text_id`, `text` and `label3` (plus the original rating columns, which are passed through). `label3` is the three-class mapping used in the paper: ratings 1-2 are LOW, 3 is MEDIUM, 4-5 are HIGH, after resolving each text to its most common rating (ties go to the rating closest to the mean).
- The survey is never used for training or model selection.

## Reproducing the paper, step by step

Run everything from the repository root.

### Step 1: source a random sample from each corpus

```bash
python pipeline.py --stage source
```

`sourcing.py` streams each raw CSV in chunks, keeps texts of 20-4,000 characters with at least 4 words, removes exact duplicates (by content hash), and draws a **uniform random** sample by reservoir sampling (seed 42). The draw size is `BASE_SAMPLE_SIZE x` a random factor in `[2.0, 2.5]` (`config.py`), which gives a pool larger than needed. Output: `data/sourced/{nemotron,enron,wnut17}_sourced.csv`.

### Step 2: label with the three judges

Push the sourced data to the Modal volume, then run each judge on each corpus. The detached orchestrator keeps running if your laptop closes, and it resumes from checkpoints.

```bash
modal volume put privacy-judge-data data/sourced /sourced

# one invocation per (judge, corpus); repeat for judge_a, judge_b, judge_c
# and corpora nemotron, enron, wnut17, survey
modal run --detach modal_app.py::run_judge_job --judge-key judge_a --corpus-name nemotron
modal run --detach modal_app.py::run_judge_job --judge-key judge_b --corpus-name nemotron
modal run --detach modal_app.py::run_judge_job --judge-key judge_c --corpus-name nemotron
# ... same for enron, wnut17 and survey

modal app logs privacy-risk-judge-panel                      # progress
modal volume get privacy-judge-data /judged data/judged      # download results
```

| Judge key | Model | Precision / quantization | GPU |
|---|---|---|---|
| `judge_a` | `openai/gpt-oss-120b` | MXFP4 | 1x H200 |
| `judge_b` | `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` | FP8 | 1x H200 |
| `judge_c` | `RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic` | FP8 (compressed-tensors) | 1x H200 |

- All judges use greedy decoding (`do_sample=False`) and the same zero-shot prompt (`prompts.py`, reproduced in the paper). The prompt has no few-shot examples.
- `judge_a` runs at low reasoning effort with `max_new_tokens=512`, because GPT-OSS emits reasoning before its answer. `judge_b` and `judge_c` use `max_new_tokens=16`.
- Inference batch sizes are 16, 32 and 16 for `judge_a`, `judge_b` and `judge_c` (`INFERENCE_BATCH_SIZE` in `config.py`). Lower them if you hit CUDA out-of-memory errors.
- Outputs that do not parse to LOW / MEDIUM / HIGH are recorded as `UNKNOWN` and do not count as a vote.
- Smoke test one judge first: `modal run modal_app.py --judge judge_a` (sends two hard-coded texts and prints the raw generations).
- **Row cap.** `run_judge_job` judges only the first `MAX_SAMPLES` rows of each sourced file (a constant inside `run_judge_job` in `modal_app.py`, currently `3000`, and `None` for the survey). The paper judged **5,000 / 4,000 / 3,000** rows of Nemotron-PII / Enron / WNUT-17 (12,000 in total), so set that constant to the value for the corpus you are running before launching each job.

### Step 3: majority vote

```bash
python pipeline.py --stage vote
```

`voting.py` keeps a row when at least 2 of the 3 judges agree on a valid label (`MIN_AGREEING_JUDGES = 2`), drops and logs the rest, and prints Fleiss' kappa (rows with an `UNKNOWN` label are excluded from kappa). Output: `data/voted/{corpus}_majority.csv` and `{corpus}_no_majority_dropped.csv`.

### Step 4: build the balanced training set

The paper uses **2,000 samples per class (6,000 in total)**, but `config.py` ships with `SAMPLES_PER_CLASS = 3_000`. Set it to 2000 first, then run:

```bash
python pipeline.py --stage balance
```

`balance.py` pools all corpora, samples each class down to `SAMPLES_PER_CLASS` (seed 42), and does not upsample: if a class is short it keeps every available row and prints a warning. Output: `data/final/final_training_corpus.csv` (columns `sample_id,text,label`) and `data/final/class_weights.csv`.

### Step 5: fine-tune the ModernBERT student

The script defaults (1 epoch, max length 512) differ from the paper, so pass the paper's settings explicitly:

```bash
modal run train_modernbert.py \
  --train-file data/final/final_training_corpus.csv \
  --text-col text --label-col label \
  --run-name privacy_v7_epoch3 \
  --model-id answerdotai/ModernBERT-base \
  --epochs 3 --max-length 2048 --batch-size 8 \
  --learning-rate 5e-5 --weight-decay 0.01 --warmup-ratio 0.1 \
  --val-fraction 0.1 --class-weights balanced --seed 42
```

- The local file is uploaded to the Modal volume automatically. Use `--overwrite yes` to reuse a run name.
- Training uses a stratified 90/10 train/validation split, class-weighted cross-entropy, a linear schedule, bf16 when supported, and one A10G (24 GB) by default (override with the `MODERNBERT_TRAIN_GPU` environment variable).
- The best epoch by validation macro-F1 is restored at the end (`load_best_model_at_end`), and that checkpoint is saved.
- Saved on the volume under `/models/<run-name>/`: model and tokenizer, `label_map.json`, `train_config.json` (all settings and a hash of the training file), and `metrics.json`.
- Optional download: `modal volume get privacy-judge-data /models/privacy_v7_epoch3 models/privacy_v7_epoch3`.

### Step 6: predict on the human-annotated survey

```bash
modal run test_modernbert.py \
  --run-name privacy_v7_epoch3 \
  --test-file data/sourced/survey_with_label3.csv \
  --text-col text --label-col label3
```

This writes `data/predictions/privacy_v7_epoch3/survey_with_label3_predictions.csv` (your columns plus `predicted_label`, `predicted_confidence`, `prob_<LABEL>`, `prediction_status`, `is_correct`), together with a `_summary.json` and `_summary.md` containing accuracy, macro and weighted F1, per-class metrics and the confusion matrix. Without `--max-length` it reuses the training length (2048).

### Step 7: baselines and statistical tests

```bash
# Presidio + spaCy rule baseline and always-LOW baseline, scored against the survey labels
python presidio_baseline_eval.py --csv data/predictions/privacy_v7_epoch3/survey_with_label3_predictions.csv

# bootstrap CIs (10,000 resamples, seed 42), paired gains over always-LOW, McNemar's test
python bootstrap_ci_survey.py --csv data/predictions/privacy_v7_epoch3/survey_with_label3_predictions.csv
```

- `presidio_baseline_eval.py` uses fixed rules written before evaluation: HIGH if a critical identifier is found (US SSN, credit card, bank account, IBAN, passport, driver's license, ITIN, crypto, UK NHS) at confidence 0.5 or higher; MEDIUM if a contact identifier is found (email, phone, IP, MAC) or a PERSON co-occurs with a LOCATION, ORGANIZATION or NRP at confidence 0.4 or higher; otherwise LOW. It reads only the text column. Output goes to `baseline_results/`.
- `bootstrap_ci_survey.py` reports accuracy, macro F1, weighted F1 and quadratic weighted kappa with 95% percentile CIs, per-class F1 with CIs, paired bootstrap gains over always-LOW for seven metrics (accuracy, macro F1, weighted F1, QWK and the three per-class F1 scores), and continuity-corrected McNemar's test. Output goes to `bootstrap_results/` (`baseline_comparison.csv` has 7 rows).

## Settings used in the paper

| Item | Value |
|---|---|
| Risk framework | NIST SP 800-122, Section 3.2 (six factors), classes LOW / MEDIUM / HIGH |
| Judges | GPT-OSS-120B, Qwen3-Next-80B-A3B-Instruct-FP8, Llama-4-Scout-17B-16E-Instruct-FP8 |
| Judge prompt | zero-shot, `prompts.py` (system prompt plus `TEXT: {text}`) |
| Rows judged | Nemotron-PII 5,000, Enron 4,000, WNUT-17 3,000 (12,000 in total) |
| Vote rule | at least 2 of 3 judges agree; others dropped |
| Training set | 6,000 samples, 2,000 per class (3,945 unanimous, 2,055 decided 2-of-3) |
| Student | ModernBERT-base (149M parameters) |
| Fine-tuning | AdamW, lr 5e-5, weight decay 0.01, linear schedule with 10% warmup, batch size 8, max length 2048, 3 epochs, bf16, class-weighted cross-entropy, seed 42, stratified 90/10 split |
| Student hardware | 1x NVIDIA A10G (24 GB) |
| Judge hardware | 1x NVIDIA H200 (141 GB) each, on Modal |
| Evaluation set | 250 human-rated texts (145 LOW / 61 MEDIUM / 44 HIGH) |
| Bootstrap | 10,000 resamples, seed 42, percentile 95% CIs |

Composition of the training set by source (Table I of the paper):

| Dataset | LOW | MEDIUM | HIGH | Total |
|---|---:|---:|---:|---:|
| Nemotron-PII | 747 | 1,224 | 1,918 | 3,889 |
| Enron Email | 632 | 719 | 68 | 1,419 |
| WNUT-17 | 621 | 57 | 14 | 692 |
| Total | 2,000 | 2,000 | 2,000 | 6,000 |

## Reference results

Expected values on the 250-text survey (saved copies are in `baseline_results/` and `bootstrap_results/`):

| System | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Always-LOW | 0.580 | 0.245 | 0.426 |
| Presidio + spaCy | 0.524 | 0.289 | 0.442 |
| Majority vote of the three judges | 0.740 | 0.626 | 0.724 |
| **Student ModernBERT** | **0.728** | **0.630** | **0.718** |

Student 95% bootstrap CIs: accuracy [0.672, 0.784], macro F1 [0.556, 0.698], weighted F1 [0.659, 0.776], QWK 0.642 [0.548, 0.728]. McNemar's test against always-LOW: b = 55, c = 18, chi-square = 17.75, p = 2.5e-05.

Student confusion matrix (rows = human label, columns = prediction; LOW, MEDIUM, HIGH):

```
true LOW      127   18    0
true MEDIUM    17   41    3
true HIGH       7   23   14
```

Re-running can give slightly different numbers: judge outputs depend on the model files and library versions that Modal resolves at build time, and GPU training is not bit-exact even with a fixed seed.

## Mapping paper tables to code

| Paper table / result | Produced by |
|---|---|
| Table I (training set by source) | `balance.py` output, `data/final/final_training_corpus.csv`, joined with the `sample_id` prefix (corpus name) |
| Tables II and V (student, baselines) | `test_modernbert.py` summary and `presidio_baseline_eval.py` |
| Table III (judges vs. human labels) | `data/judged/survey_judge_{a,b,c}.csv` (columns `judge_{a,b,c}_label` next to `label3`), scored with scikit-learn precision / recall / F1 (macro and support-weighted, class order LOW, MEDIUM, HIGH) |
| Table IV (student vs. teacher agreement) | student predictions CSV compared with the judges' majority label; the single survey text on which all three judges disagree is resolved with the ordinal median label (this differs from `voting.py`, which drops no-majority rows when building training data) |
| Table VI (bootstrap CIs and paired gains) | `bootstrap_ci_survey.py` (`summary_metrics.csv`, `per_class_f1.csv`, `baseline_comparison.csv`) |
| Small-judge experiment (Section IV-D) | `config_small.py` and `modal_app_small.py` |
| Tables VII-IX (error analysis, JEV) | student and judge outputs on the survey; JEV (`jev-1.13.0`) was queried zero-shot through its hosted API with the same prompt and is not reproduced by the commands above |

## Notes and limitations

- **Modal costs and quotas.** The three judges need H200-class GPUs. Run the smoke test and a small `MAX_SAMPLES` first.
- **Resumability.** Judging checkpoints after every chunk to `data/judged/checkpoints/{corpus}_{judge}.jsonl` on the volume. Re-running the same command skips rows that are already labeled.
- **Unpinned dependencies.** Besides the packages noted in Setup, library versions are resolved when the Modal images are built, so very different versions may behave differently.
- **Label noise.** Judge labels are not ground truth. The student inherits the judges' main bias: sensitive text without a direct identifier (for example mental-health, medical or legal content) is often labeled MEDIUM instead of HIGH. This is the dominant error in the paper (23 of 44 HIGH texts predicted as MEDIUM).
- **Data handling.** Do not commit raw Enron or other corpora with real personal information. `data/raw/` is intended for local use only.
- **Evaluation size.** The survey has only 250 texts, so use the confidence intervals above rather than point estimates when comparing systems.

## Anonymity and license

This repository is provided anonymously for double-blind review. Author, affiliation and license information will be added after the review period.
