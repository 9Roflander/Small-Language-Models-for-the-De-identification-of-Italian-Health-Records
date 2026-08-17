# Small Language Models for the De-identification of Italian Health Records

Fine-tuning **Llama-3.2-3B** with LoRA + 4-bit NF4 quantization for GDPR-compliant de-identification of Italian clinical notes. Targets four PII categories: **NOME** (names), **ETÀ** (age), **DATA** (dates), and **LUOGO/INDIRIZZO** (locations/addresses).

## Headline Result

| Model | Params | NOME | ETÀ | DATA | LUOGO | **Macro F1** |
|---|---|---|---|---|---|---|
| llama3.2:3b (zero-shot, Miranda et al.) | 3B | 0.53 | 0.41 | 0.29 | 0.41 | 0.41 |
| gemma3:4b (zero-shot, paper) | 4B | 0.73 | 0.30 | 0.27 | 0.75 | 0.51 |
| gemma3:12b (zero-shot, paper best) | 12B | **0.81** | 0.14 | 0.65 | **0.88** | 0.620 |
| **OURS — Gemma-3 1B fine-tuned (5-fold CV)** | **1B** | 0.592 | **0.745** | 0.582 | 0.523 | **0.611 ± 0.109** |
| **OURS — Llama-3.2-3B fine-tuned v4 (5-fold CV)** | **3B** | 0.714 | 0.554 | 0.536 | 0.790 | **🏆 0.6486 ± 0.073** |

Two headline findings:

1. **A 3B-parameter fine-tuned model running on a 12 GB consumer GPU beats zero-shot 12B+ models under rigorous 5-fold cross-validation.** Largest win: **ETÀ +0.144 vs paper's best** (fine-tuning resolves age-format ambiguities that confuse general-purpose LLMs).
2. **A 1B-parameter fine-tuned model essentially matches a 12B zero-shot model** (0.611 vs 0.620). On the ETÀ category specifically, Gemma-3 1B fine-tuned reaches **0.745** — the highest of any system tested, including our own 3B Llama.

> Gemma-3 4B was attempted but does not fit on a 12 GB GPU even with aggressive memory savings — its 262K-vocab cross-entropy is the OOM cliff. See `FINAL_REPORT.md` §4.5.

See [`FINAL_REPORT.md`](FINAL_REPORT.md) for the complete methodology, ablations, and analysis.

## BERT-NER Baseline

A fine-tuned Italian BERT token classifier (`dbmdz/bert-base-italian-xxl-cased`), evaluated under the identical 5-fold CV protocol and scored with the same placeholder-count metric, to test whether the "obvious" NER architecture beats the generative redaction approach above.

| Model | Params | Approach | Macro F1 |
|---|---|---|---|
| gemma3:12b (zero-shot, paper best) | 12B | zero-shot | 0.620 |
| **BERT-NER (ours)** | **0.11B** | fine-tuned token classifier | **0.569 ± 0.143** |
| Gemma-3 1B fine-tuned (ours) | 1B | LoRA fine-tune, generative | 0.611 ± 0.109 |
| Llama-3.2-3B v4 fine-tuned (ours) | 3B | LoRA fine-tune, generative | 🏆 0.649 ± 0.073 |

It does **not** beat either fine-tuned LLM, and lands below the paper's best zero-shot baseline too — despite being 27–100× smaller. Per-category, its strongest result is ETÀ (0.784, the best ETÀ score of any system tested here), while NOME/DATA/LUOGO show a consistent precision-near-1.0-but-recall-collapses pattern across folds, consistent with class-imbalance underfitting (entity tokens are a small minority against `O` tokens, plain cross-entropy, only 4 epochs, a freshly-initialized classification head). Full per-fold numbers: [`cv_bert_results.json`](cv_bert_results.json).

Unlike the LLM pipeline, this baseline trains on a laptop — no CUDA GPU required, runs on Apple Silicon (MPS) or CPU.

**Follow-up ablations** — a chunking fix for notes longer than BERT's 512-token window, and domain-adaptive continual pretraining (CPT) mirroring the LLM's Phase 1 — are in [`bert_cpt_and_chunking_findings/`](bert_cpt_and_chunking_findings/README.md). Headline: CPT gives +0.080 absolute macro F1 when isolated, but a naive chunking fix actually *hurt* by −0.061 (entities fragment across chunk boundaries), netting only +0.019 over the original baseline (0.569 → 0.588).

## Reference Work

This work builds on and benchmarks against:

> Michele Miranda, Sébastien Bratières, Stefano Patarnello, and Livia Lilli. 2025. **Mamma Mia! Where's My Name? De-Identifying Italian Clinical Notes with Large Language Models.** In *Proceedings of the Eleventh Italian Conference on Computational Linguistics (CLiC-it 2025)*, pages 735–746, Cagliari, Italy. CEUR Workshop Proceedings.

- Paper: [aclanthology.org/2025.clicit-1.78](https://aclanthology.org/2025.clicit-1.78/)
- Authors' page: [praiselab-picuslab](https://github.com/praiselab-picuslab)

We use the same 80-note gold standard (CLinkaRT/E3C derivative), the same four PII categories, and the same deterministic placeholder-count evaluation metric — enabling head-to-head comparison.

## Method

The training pipeline:

1. **Phase 1 — Continual Pre-Training (CPT)** on Italian clinical text (`NLP-FBK/dyspnea-clinical-notes` + `praiselab-picuslab/DART`) to adapt vocabulary.
2. **Phase 2 — Redaction SFT** on a curated mix of synthetic + gold data. Input = clinical note, output = same note with `[NOME]`, `[ETÀ]`, `[DATA]`, `[LUOGO/INDIRIZZO]` tags substituted inline.
3. **Evaluation** — 5-fold cross-validation using the deterministic placeholder-count metric.

Two methodological contributions:

1. **Style-anchored synthetic data generation** using Gemini 2.5 Flash. Few-shot prompting with real gold notes as anchors, schema-constrained outputs, entity-distribution balancing. ~$15 total API cost for ~1700 valid synthetic samples.
2. **The empirical finding that synthetic-data style transfer matters more than volume.** A control trained on 1000 stylistically-mismatched synthetic notes achieved macro F1 = 0.18 — worse than zero-shot. The same setup with style-anchored data jumped to 0.535 → 0.649.

## Repository Layout

```
.
├── FINAL_REPORT.md                 # Full report with methodology and results
├── README.md
├── requirements.txt
├── cv_v3_results.json              # v3 5-fold CV results (878 synth samples)
├── cv_v4_results.json              # v4 5-fold CV results (1676 synth samples) — final
├── cv_gemma1b_results.json         # Gemma-3-1B comparison run
├── cv_bert_results.json            # BERT-NER baseline, 5-fold CV
├── bert_cpt_and_chunking_findings/ # BERT ablations: chunking fix + domain-adaptive CPT
│   ├── README.md                   #   writeup + isolated per-variable deltas
│   ├── cv_bert_chunked_results.json    #   chunking fix alone (no CPT), 5-fold CV
│   └── cv_bert_cpt_results.json        #   CPT + chunking fix, 5-fold CV
├── data/
│   ├── gold_standard_80.json       # 80 manually annotated Italian clinical notes
│   ├── synthetic_v2_1000.json      # 624 Gemini-generated, style-anchored, with-names
│   ├── synthetic_v4_named_more.json   # 577 additional with-names (v4)
│   ├── synthetic_v2_noname_400.json   # 254 Gemini-generated, name-free
│   ├── synthetic_v4_noname_more.json  # 221 additional name-free (v4)
│   └── test_indices_seed42.json    # Persisted train/test split (seed=42)
└── src/
    ├── generate_synthetic_v2.py        # Style-anchored Gemini generator (single-thread)
    ├── generate_synthetic_parallel.py  # Parallel variant (3 API keys)
    ├── generate_noname.py              # Name-free synthetic generator
    ├── train_redact_v3.py              # Final redaction SFT (single split)
    ├── eval_redact_v3.py               # Held-out eval with paper metric
    ├── train_evaluate_cv_v3.py         # v3 5-fold CV pipeline
    ├── train_evaluate_cv_v4.py         # v4 5-fold CV pipeline (final)
    ├── train_evaluate_cv_gemma.py      # Gemma-3 (1B/4B) CV comparison
    ├── eval_paper_metric.py            # Standalone paper-metric evaluator
    ├── generalization_test.py          # OOD test (Kazakh/Uzbek entities, year 2034)
    ├── train_evaluate_cv_bert.py        # BERT-NER 5-fold CV baseline (token classification)
    ├── pretrain_cpt_bert.py             # BERT domain-adaptive CPT (dyspnea notes + DART drug inserts)
    ├── train_evaluate_cv_bert_cpt.py    # BERT-NER CV starting from the CPT checkpoint above
    └── train_evaluate_cv_bert_chunked_test.py  # Ablation: chunking fix alone, no CPT
```

## Setup

**Requirements:** NVIDIA GPU with ≥ 12 GB VRAM (tested on RTX 4070 Ti) for the LLM pipelines. The BERT-NER baseline (`src/train_evaluate_cv_bert.py` and friends) is much lighter — 110M params — and trains fine on CPU or Apple Silicon (MPS); no CUDA GPU required. It was developed and run on an Apple M4 MacBook (16 GB).

```bash
git clone https://github.com/9Roflander/Small-Language-Models-for-the-De-identification-of-Italian-Health-Records.git
cd Small-Language-Models-for-the-De-identification-of-Italian-Health-Records

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt

huggingface-cli login   # required for gated meta-llama/Llama-3.2-3B
```

**Windows note:** set `PYTHONUTF8=1` before running any script:
```powershell
$env:PYTHONUTF8 = "1"
```

## Reproducing the Results

```bash
# (Optional) Regenerate synthetic data — costs ~$15 in Gemini API credits.
# Provided JSONs in data/ are sufficient to skip this step.
export GOOGLE_API_KEY=your-gemini-key
python src/generate_synthetic_parallel.py --n 1000
python src/generate_noname.py --n 400

# Final v4 5-fold cross-validation (the headline result)
python src/train_evaluate_cv_v4.py

# Optional comparisons
python src/train_evaluate_cv_v3.py                    # v3 baseline (878 synth)
python src/train_evaluate_cv_gemma.py --model_size 1b # Gemma-3-1B comparison
python src/train_evaluate_cv_gemma.py --model_size 4b # Gemma-3-4B comparison

# OOD generalization test (Kazakh/Uzbek entities never seen in training)
python src/generalization_test.py
```

Phase 1 (CPT) artifacts are not included in this repo (large model weights). The Phase 2 scripts expect a merged base model at `./temp_merged_phase1/`. See `FINAL_REPORT.md` §2.3 for the Phase 1 procedure.

### BERT-NER baseline

Trains an Italian BERT token classifier on the same 5-fold split and gold+synthetic+CRF data mix as the LLM CV pipelines above, so results are directly comparable. No merged Phase-1 checkpoint required — it starts straight from the public `dbmdz/bert-base-italian-xxl-cased` checkpoint.

```bash
python src/train_evaluate_cv_bert.py    # 5-fold CV -> cv_bert_results.json (~3-4h on an M4 laptop)
```

**Optional: give BERT its own domain-adaptation step (mirrors the LLM's Phase 1 CPT).** Continues pretraining BERT with masked-language-modeling on the same two corpora named in `FINAL_REPORT.md` §2.3, before the NER fine-tune:

- [`NLP-FBK/dyspnea-clinical-notes`](https://huggingface.co/datasets/NLP-FBK/dyspnea-clinical-notes) (`it` split, 2,667 Italian clinical notes) — public, no auth needed.
- [`praiselab-picuslab/DART`](https://huggingface.co/datasets/praiselab-picuslab/DART) (16,029 Italian drug package inserts — indications, dosage, contraindications, interactions, adverse effects, one row per medicinal product) — **gated**. Request access on the dataset page with the HF account you intend to use, then authenticate locally:
  ```bash
  pip install -U "huggingface_hub[cli]"
  hf auth login   # paste a token from https://huggingface.co/settings/tokens; run this yourself, not via a script, so the token is never echoed anywhere
  ```
  `pretrain_cpt_bert.py` deterministically subsamples 6,000 of DART's 16,029 rows (`seed=42`) to keep training wall-clock time reasonable on a laptop; this is documented in the script rather than silently truncated, so it's easy to raise `DART_SAMPLE_SIZE` back toward the full dataset on faster hardware.

```bash
python src/pretrain_cpt_bert.py         # MLM continual pretraining -> ./bert_it_clinical_cpt/ (~80min on an M4 laptop)
python src/train_evaluate_cv_bert_cpt.py  # 5-fold CV from that checkpoint -> cv_bert_cpt_results.json (~3-4h)
```

## Entity Categories

| Type | Description | Example |
|---|---|---|
| `NOME` | Patient or doctor names | *Maria Rossi* |
| `ETÀ` | Age references | *45 anni*, *all'età di 70 anni* |
| `DATA` | Dates and timestamps | *marzo 2018*, *15/04/2020* |
| `LUOGO/INDIRIZZO` | Locations and addresses | *Ospedale San Raffaele, Milano* |

## Citation

```bibtex
@misc{sultangazy2026italiandeid,
  title  = {Fine-tuning a 3B LLM with Style-Anchored Synthetic Data Beats Zero-Shot
            12B Models on Italian Clinical De-Identification},
  author = {Sultangazy, Dair},
  year   = {2026},
  note   = {Builds on Miranda et al. (2025), CLiC-it 2025.}
}

@inproceedings{miranda2025mammamia,
  title     = {Mamma Mia! Where's My Name? De-Identifying Italian Clinical Notes
               with Large Language Models},
  author    = {Miranda, Michele and Bratières, Sébastien
               and Patarnello, Stefano and Lilli, Livia},
  booktitle = {Proceedings of the Eleventh Italian Conference on
               Computational Linguistics (CLiC-it 2025)},
  pages     = {735--746},
  year      = {2025},
  address   = {Cagliari, Italy},
  publisher = {CEUR Workshop Proceedings}
}
```

## License

Code is released under the MIT License. The synthetic clinical notes in `data/` are generated using Google's Gemini 2.5 Flash API and contain no real PII.
