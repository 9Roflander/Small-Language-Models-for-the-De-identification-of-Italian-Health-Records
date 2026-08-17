# BERT-NER Ablations: Chunking Fix vs. Domain-Adaptive CPT

Two follow-up experiments on the `dbmdz/bert-base-italian-xxl-cased` NER baseline (`cv_bert_results.json`, macro F1 = 0.569 ± 0.143), run on the same 5-fold split (seed=42) and the same gold+synthetic+CRF data mix, so all three are directly comparable.

## Results

| Run | Base checkpoint | Chunking fix | Macro F1 | Results file |
|---|---|---|---|---|
| Baseline | `dbmdz/bert-base-italian-xxl-cased` | no (truncate at 512) | **0.569 ± 0.143** | `../cv_bert_results.json` |
| Chunked | `dbmdz/bert-base-italian-xxl-cased` | yes | **0.508 ± 0.125** | `cv_bert_chunked_results.json` |
| CPT + Chunked | `./bert_it_clinical_cpt` (domain-adapted) | yes | **0.588 ± 0.146** | `cv_bert_cpt_results.json` |

Isolating each variable (both new runs share the chunking fix, so compare them to each other for the CPT effect, and each to baseline for the chunking effect):

- **Chunking fix alone: −0.061** (0.569 → 0.508). Hurt, despite fixing real truncation.
- **CPT on top of chunking: +0.080** (0.508 → 0.588). Clear win.
- **Net vs. original baseline: +0.019** (0.569 → 0.588). The CPT gain is real but mostly offset by the chunking regression.

## Finding 1: The chunking fix hurt, contrary to expectation

`src/train_evaluate_cv_bert.py`'s original tokenizer truncated every note at `MAX_LENGTH=512` wordpieces. 55.8% of synthetic notes and 40% of gold notes exceed that window, so their tails were silently dropped from both training and evaluation.

The fix (see the diff to `make_tokenize_fn` / `predict_spans` in `src/train_evaluate_cv_bert.py`) splits each note into non-overlapping 510-token chunks instead of truncating, so the full note contributes to both training and inference. `src/train_evaluate_cv_bert_chunked_test.py` reruns the CV with this fix and no CPT, to isolate its effect.

Result: macro F1 dropped from 0.569 to 0.508. Most likely cause: a span can't continue across a chunk boundary (see the code comment in `predict_spans`), so entities that straddle a chunk edge get fragmented into partial, non-matching predictions at eval time — a cost that outweighs the benefit of recovering training signal from previously-truncated tails. This is a useful negative result: naive fixed-window chunking is not a free win for this task, and entity-aware chunk placement (e.g. splitting on sentence boundaries, or padding a small overlap around chunk edges) would need to be tried before trusting a chunked eval number.

## Finding 2: Domain-adaptive CPT is the bigger lever

`src/pretrain_cpt_bert.py` continues MLM pretraining of `dbmdz/bert-base-italian-xxl-cased` on the same two Italian clinical corpora used for the LLM's Phase 1 CPT (`NLP-FBK/dyspnea-clinical-notes` + a 6,000-row sample of `praiselab-picuslab/DART`), producing `./bert_it_clinical_cpt/` (not committed — ~423 MB, gitignored). `src/train_evaluate_cv_bert_cpt.py` then runs the same 5-fold CV starting from that checkpoint instead of the vanilla one.

Compared apples-to-apples against the chunked (non-CPT) run, CPT gives **+0.080 absolute macro F1** (0.508 → 0.588) — the largest single effect of the two ablations tested here, consistent with the hypothesis in the main `FINAL_REPORT.md` that domain-adaptive pretraining should mirror the LLM pipeline's Phase 1 benefit.

`src/pretrain_cpt_bert.py` was also updated to prefer CUDA over MPS/CPU when available (this run was on an RTX 4070 Ti rather than the Apple M4 laptop the original BERT baseline was developed on), with `bf16` training and step-based (rather than ratio-based) warmup.

## Reproducing

```bash
# Domain-adaptive CPT (~80 min on an M4 laptop, faster on a CUDA GPU)
python src/pretrain_cpt_bert.py

# CPT + chunking-fixed CV -> cv_bert_cpt_results.json
python src/train_evaluate_cv_bert_cpt.py

# Chunking fix alone (no CPT), for the isolated ablation -> cv_bert_chunked_results.json
python src/train_evaluate_cv_bert_chunked_test.py
```
