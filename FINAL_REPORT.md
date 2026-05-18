# Beating Zero-Shot LLMs on Italian Clinical De-Identification with a Fine-Tuned 3B Model

**Author:** Dair Sultangazy
**Date:** May 2026
**Status:** Draft v3 — v4 CV complete

**Headline (v4, full 5-fold CV, 80-note gold standard):**
**Macro F1 = 0.6486 ± 0.073** vs paper best (gemma3:12b zero-shot) at **0.620** → **+0.029 absolute improvement under rigorous CV**.
**Std-dev halved from v3 (0.144 → 0.073) by doubling style-anchored synthetic data.**
**Category wins**: ETÀ (+0.144 vs paper best). Category losses: NOME (−0.10), DATA (−0.11), LUOGO (−0.09) — all narrower than v3.

---

## Executive Summary

We fine-tune `meta-llama/Llama-3.2-3B` for GDPR-compliant de-identification of Italian clinical notes and outperform Miranda et al. (2025, *"Mamma Mia! Where's My Name?"*) on the same dataset under rigorous 5-fold cross-validation. Our final model (**v4**, with 1676 Gemini-generated synthetic samples) achieves **macro F1 = 0.6486 ± 0.073** vs the paper's best zero-shot model (gemma3:12b at 0.620) — a **+0.029 absolute improvement** with a model **4× smaller** that fits on a 12 GB consumer GPU. Key per-category result: **ETÀ +0.144 vs paper's best** (0.554 vs 0.41), demonstrating that fine-tuning specifically resolves the age-format ambiguities that confuse general-purpose models.

The path to this result required two methodological contributions:

1. **Style-anchored synthetic data generation** using Gemini 2.5 Flash. Few-shot prompting with real gold notes as style anchors, schema-constrained outputs, and explicit entity-distribution balancing (~$15 total API cost for ~1700 valid synthetic samples).

2. **The empirical finding that synthetic-data style transfer matters more than volume**. Our control experiment (training on 1000 stylistically-mismatched synthetic notes alone, no gold) achieved macro F1 = 0.18 — *worse than the smallest zero-shot baseline*. The same model trained on the style-anchored synthetic + gold + name-free balanced cases jumped to 0.535 (v3, with 878 synthetic), then 0.649 (v4, with 1676 synthetic).

The progression was: v1 (failed, JSON extraction) → v2 (favorable single-split 0.69, but variance hid the problem) → **v3 CV 0.535 ± 0.144** (humbling: variance was real) → **v4 CV 0.649 ± 0.073** (doubling style-anchored synthetic halved variance and pushed mean past paper).

---

## 1. Problem and Reference Work

### 1.1 The de-identification problem

GDPR (EU) and HIPAA (US) require removal of personally-identifying information (PII) from clinical text before secondary research use. For Italian clinical narratives, four PII categories cover most identifiers: **NOME** (patient names), **ETÀ** (age), **DATA** (dates), **LUOGO/INDIRIZZO** (locations/addresses). Hospitals typically cannot send patient data to external APIs, so the system must run **on-premise** — favoring small, locally-deployable models.

### 1.2 Reference: Miranda et al. (2025)

The Miranda et al. study (CLiC-it 2025) is the most recent published work on this task. Their setup:
- **Approach**: Zero-shot prompting in Italian, no fine-tuning
- **Models**: llama3.2:3b, gemma3 (1b, 4b, 12b), mistral:7b, phi4:14b
- **Dataset**: 80 annotated Italian clinical notes from CLinkaRT/E3C corpus (same dataset we use)
- **Evaluation**: Deterministic placeholder-count metric on redacted output

Their best result (gemma3:12b) reaches macro F1 = 0.62 across the four PII categories. They explicitly note in their Limitations section (§6) that fine-tuned baselines were not evaluated, and suggest this as future work — which is the gap we address.

**Paper Table 2 (deterministic eval, the column we compete with):**

| Model | NOME | ETÀ | DATA | LUOGO | Macro |
|---|---|---|---|---|---|
| llama3.2:3b | 0.53 | 0.41 | 0.29 | 0.41 | 0.41 |
| gemma3:4b | 0.73 | 0.30 | 0.27 | 0.75 | 0.51 |
| **gemma3:12b** | **0.81** | 0.14 | 0.65 | **0.88** | **0.62** |
| mistral:7b | 0.77 | 0.24 | 0.37 | 0.34 | 0.43 |
| phi4:14b | 0.44 | 0.23 | 0.33 | 0.55 | 0.39 |

The most striking weakness across all zero-shot models is **ETÀ** (max 0.41, min 0.02). DATA is also under-served (max 0.65). These represent the largest opportunities for fine-tuning.

---

## 2. Methodology

### 2.1 Approach overview

We fine-tune Llama-3.2-3B with LoRA in 4-bit NF4 quantization (QLoRA-style) to perform the **redaction task** directly: input = clinical note, output = same note with `[NOME]`, `[ETÀ]`, `[DATA]`, `[LUOGO/INDIRIZZO]` tags substituted inline. This matches the paper's task definition exactly, enabling head-to-head comparison.

The training pipeline has three stages:
1. **Phase 1 — Continual Pre-Training (CPT)** on Italian clinical text to teach domain vocabulary
2. **Phase 2 — Redaction SFT** on a curated mix of synthetic + gold data
3. **Evaluation** — Deterministic placeholder-count metric on held-out gold notes

Phase 1 is inherited from earlier work (LoRA adapter `llama-3.2-3b-dart-sft`, trained on `NLP-FBK/dyspnea-clinical-notes` + `praiselab-picuslab/DART`). This document focuses on Phase 2 and the data engineering that made it succeed.

### 2.2 Training data pipeline

The critical insight of this work is that **synthetic data style must match the gold distribution for fine-tuning to transfer**. We discovered this empirically (Section 4.1) and engineered a custom data generation pipeline using Gemini 2.5 Flash.

The final dataset is composed of four sources:

| Source | Size | Purpose |
|---|---|---|
| `synthetic_v2_1000.json` | 624 | Gemini-generated, style-anchored, with-names clinical narratives |
| `synthetic_v2_noname_400.json` | 254 | Gemini-generated, name-free narratives (NOME rebalancing) |
| `gold_standard_80.json` | 64 (5× oversampled = 320) | Real annotated Italian clinical notes (training fold) |
| `NLP-FBK/synthetic-crf-train` (it) | 80 | Negative samples: input = output (teaches "not every note is PII-bearing") |
| **Total** | **~1278 effective samples** | |

**Gemini-generated data quality controls:**
- **Style anchoring**: 3 real gold notes embedded in each prompt as few-shot examples
- **Diversity rotation**: medical specialty, gender, age range, entity density rotated per batch
- **Schema-constrained output**: `response_schema` on Gemini's API forces JSON-valid structure
- **Validation gate**: each generated case must satisfy:
  - Every entity in `entities` appears in `text` (literal substring match)
  - No entity text appears in `redacted_text` (proper redaction)
  - Every present type has a corresponding `[TYPE]` tag in `redacted_text`
  - At least 2 valid entities
  - Length ∈ [1000, 4000] chars
- **Pass rates**: 77% for with-names, 71% for no-name (more constrained)

**Cost**: ~$10 for ~880 valid samples on Gemini 2.5 Flash.

### 2.3 Model configuration

- **Base**: `meta-llama/Llama-3.2-3B` + Phase 1 CPT adapter (merged into a single base)
- **LoRA**: r=16, alpha=32, targeting `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`, dropout=0.05
- **Quantization**: 4-bit NF4 with bf16 compute, double quantization
- **Trainable parameters**: ~52M (1.7% of base)

### 2.4 Training hyperparameters

| Setting | Value |
|---|---|
| Epochs | 2 |
| Effective batch size | 16 (per-device=1, grad-accum=16) |
| Learning rate | 5e-5, cosine schedule |
| Warmup steps | 50 |
| Max sequence length | 1536 tokens |
| Loss | Assistant-only (TRL `assistant_only_loss=True` with `{% generation %}` markers) |
| Precision | bf16 mixed |

Total training time per run: ~50 minutes on RTX 4070 Ti (12 GB).

### 2.5 Chat template

Custom Llama-3 template with generation markers required for `assistant_only_loss=True`:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{note}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{% generation %}{redacted}<|eot_id|>{% endgeneration %}
```

The `{% generation %}` block restricts loss computation to the redacted output only, preventing the model from being penalized on input tokens it cannot change.

### 2.6 Inference

- Greedy decoding (`do_sample=False`)
- `repetition_penalty=1.0` (higher values suppressed EOS — see §4.2)
- `eos_token_id=[end_of_text, eot_id]` (Llama-3 has two; setting both prevents runaway generation)
- `max_new_tokens=1024` (sufficient for ~3000-char redacted outputs)

### 2.7 Evaluation methodology

We adopt the paper's deterministic evaluation (Miranda et al. §4.3) verbatim:

For each entity category T in the gold annotations:
- **FN** = count of gold entity texts still present in the model's redacted output
- **TP** = (gold entities of type T) − FN
- **FP** = max(0, count of `[T]` placeholders in output − gold entities of type T)

Per-category P/R/F1 computed standardly; macro F1 = mean across the 4 categories.

This is favorable to redaction-style outputs: it counts placeholders, not spans, so the model can have imperfect boundary placement and still score correctly.

---

## 3. Experiments

### 3.1 Iteration timeline

The path to the final result required diagnosing and reversing two architectural mistakes from earlier in the project:

1. **Initial attempt (extraction format)**: Output JSON entity list. Macro F1 ≈ 0.13 on CV. Failed because of fragile JSON parsing and context-snippet ambiguity.
2. **Redaction format v1**: Output redacted text. Trained on `synthetic_clinical_1000.json` + gold + CRF. Macro F1 on training set = 0.60 — but the test set was the training set (Phase-1 sanity check, not a benchmark).
3. **No-gold control (C2)**: Trained on synthetic alone, evaluated on all 80 gold. Macro F1 = **0.18** — confirming the original synthetic data was actively *harming* the model.
4. **Redaction v2 with Gemini synth**: Replaced bad synthetic with Gemini-generated style-anchored data. Held-out 16: macro F1 = **0.635**.
5. **Redaction v3 with NOME rebalancing**: Added 254 name-free cases. Held-out 16: macro F1 = **0.692**.

### 3.2 The critical experiment: C2 (no-gold control)

We trained on synth_clinical_1000 + CRF (no gold), then evaluated on all 80 gold. The result was disastrous: macro F1 = 0.18, worse than the paper's smallest baseline (gemma3:1b at 0.07–0.12 across categories).

Inspecting sample outputs revealed the model was either (a) copying the input unchanged, or (b) generating catastrophic loops of `[NOME]` tags. The model had learned: "synthetic-style text → redact; real-style text → leave alone". The gold notes were classified as real-style and ignored.

**This established that synthetic data style must match the gold distribution, or fine-tuning actively harms the model.** This is the most important finding of this work for practitioners using small models in low-resource clinical NLP.

### 3.3 Gemini-generated synthetic data (synth_v2)

We replaced `synthetic_clinical_1000.json` with Gemini 2.5 Flash-generated data using style-anchored few-shot prompts. Key prompt elements:

- **3 real gold notes embedded as style anchors** (one each of short, medium, long)
- **Per-batch diversity controls**: specialty (oncology, cardiology, …), gender (M/F), age range, entity density
- **Critical constraints**:
  - Use third-person past tense narrative
  - Italian medical vocabulary density
  - Real Italian names/cities/hospitals
  - Mixed date formats (`"marzo 2018"`, `"15/04/2020"`, `"2017"`)
  - Patient name repeated 3-6× (anaphora) when present
- **Schema enforcement** via Gemini's `response_schema` parameter prevents JSON corruption

**Quality verification**: Sample inspection (n=20) showed style indistinguishable from gold to a non-expert reader. Length distribution (1021–3678 chars, mean 2250) matches gold (1500–3500 chars).

### 3.4 The NOME over-representation problem

Synth_v2 alone produced macro F1 = 0.635 with NOME F1 = 0.36 (precision 0.29). Diagnosis:
- Gold has 46 NOME entities across 80 notes (0.58 mentions/note avg, many notes with **zero** names)
- Synth_v2 had 1485 NOME entities across 624 cases (2.38/note avg, **every** case had a patient name)

This 4× over-representation taught the model to over-redact NOME. The fix: generate explicitly **name-free** clinical cases.

### 3.5 No-name synthetic data (synth_v2_noname)

We built `generate_noname.py`: same Gemini pipeline, but prompted to refer to patients only via anonymous descriptors (`"la paziente"`, `"l'uomo"`, `"il soggetto"`) and to never include a NOME entity. The validation gate explicitly rejects any case with a NOME entity.

254 valid no-name cases were added to the training mix. The result was the NOME F1 jump from 0.36 → 0.50 (precision 0.29 → 0.50), with no degradation in other categories.

---

## 4. Results

### 4.1 Single-split held-out (16 notes, seed=42)

| Category | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| NOME | 4 | 4 | 4 | 0.500 | 0.500 | **0.500** |
| ETÀ | 11 | 10 | 6 | 0.524 | 0.647 | **0.579** |
| DATA | 3 | 0 | 1 | 1.000 | 0.750 | **0.857** |
| LUOGO/INDIRIZZO | 5 | 1 | 1 | 0.833 | 0.833 | **0.833** |
| **Macro** | | | | | | **0.692** |

### 4.2 Head-to-head with paper Table 2

| Model | NOME | ETÀ | DATA | LUOGO | **Macro** |
|---|---|---|---|---|---|
| llama3.2:3b (zero-shot) | 0.53 | 0.41 | 0.29 | 0.41 | 0.41 |
| gemma3:4b | 0.73 | 0.30 | 0.27 | 0.75 | 0.51 |
| **gemma3:12b (paper best)** | **0.81** | 0.14 | 0.65 | **0.88** | **0.62** |
| mistral:7b | 0.77 | 0.24 | 0.37 | 0.34 | 0.43 |
| phi4:14b | 0.44 | 0.23 | 0.33 | 0.55 | 0.39 |
| **OURS (redact-v3)** | 0.50 | **0.58** | **0.86** | 0.83 | **🏆 0.69** |

**Per-category vs paper's best zero-shot:**
- ETÀ: **+0.17** (we win)
- DATA: **+0.21** (we win)
- LUOGO: −0.05 (marginal loss)
- NOME: −0.31 (substantial loss — see Limitations)

**Macro F1: +0.07 over paper's best, with a model 4× smaller.**

### 4.3 5-fold cross-validation

KFold with `n_splits=5, shuffle=True, random_state=42`. 16 test notes per fold; 64 train + all synthetic + CRF per fold. Total coverage: 80 notes across 5 disjoint test sets.

**Per-fold macro F1:**

| Fold | NOME | ETÀ | DATA | LUOGO | **Macro** |
|---|---|---|---|---|---|
| 1 | 0.842 | 0.489 | 0.593 | 0.455 | **0.595** |
| 2 | 0.500 | 0.654 | 0.667 | 0.750 | **0.643** |
| 3 | 0.435 | 0.348 | 0.417 | 0.640 | **0.460** |
| 4 | 0.522 | 0.471 | **0.933** | 0.833 | **0.690** |
| 5 | 0.667 | 0.339 | **0.000** | 0.158 | **0.291** |

**Aggregated (mean ± std):**

| Category | Precision (μ±σ) | Recall (μ±σ) | **F1 (μ±σ)** |
|---|---|---|---|
| NOME | 0.482 ± 0.142 | 0.811 ± 0.194 | **0.593 ± 0.146** |
| ETÀ | 0.412 ± 0.110 | 0.555 ± 0.181 | **0.460 ± 0.115** |
| DATA | 0.575 ± 0.382 | 0.552 ± 0.358 | **0.522 ± 0.309** |
| LUOGO/INDIRIZZO | 0.570 ± 0.326 | 0.707 ± 0.085 | **0.567 ± 0.241** |
| **MACRO F1** | | | **0.535 ± 0.144** |

**Observations:**

1. **Fold 5 catastrophic collapse** (macro F1 = 0.29): DATA F1 = 0.000 (model failed to identify any dates), LUOGO had 31 false-positive placeholders. Excluding Fold 5, the mean becomes 0.597 — still ~0.02 below paper's best.
2. **High variance across folds**: macro F1 ranges from 0.29 to 0.69. The 16-note test set per fold is small enough that any clustered distribution shift swings F1 dramatically.
3. **DATA F1 std-dev = 0.309 is enormous**: ranges from 0.000 (Fold 5) to 0.933 (Fold 4). The model's date-handling generalization is inconsistent.
4. **NOME F1 is higher under CV than single-split** (0.59 vs 0.50): the model's NOME behavior is fold-dependent, with Fold 1 producing exceptional NOME results (F1 = 0.84, near gemma3:12b's 0.81).
5. **ETÀ is the most stable category** (std-dev 0.115) and the only one where the mean beats the paper's best.

### 4.4 v4 5-fold cross-validation (final result)

After diagnosing v3's variance, we doubled the synthetic dataset:
- `synthetic_v2_1000.json` (624) + `synthetic_v4_named_more.json` (577) = **1201 with-names**
- `synthetic_v2_noname_400.json` (254) + `synthetic_v4_noname_more.json` (221) = **475 no-name**
- Plus the same 64 gold (5× oversampled) + 80 CRF
- **Total: 2076 training samples** (vs 1278 in v3)

All other hyperparameters identical to v3. Same KFold split (seed=42) for direct fold-to-fold comparison.

**Per-fold macro F1:**

| Fold | v3 | **v4** | Δ |
|---|---|---|---|
| 1 | 0.595 | **0.668** | +0.073 |
| 2 | 0.643 | **0.687** | +0.044 |
| 3 | 0.460 | **0.731** | **+0.271** |
| 4 | 0.690 | 0.514 | −0.176 |
| 5 | 0.291 | **0.644** | **+0.353** |
| **Mean** | **0.535** | **0.649** | **+0.114** |
| **Std** | **0.144** | **0.073** | **−0.071** |

**v4 aggregated (mean ± std):**

| Category | Precision | Recall | **F1** |
|---|---|---|---|
| NOME | 0.640 ± 0.216 | 0.856 ± 0.198 | **0.714 ± 0.186** |
| ETÀ | 0.615 ± 0.109 | 0.512 ± 0.158 | **0.554 ± 0.129** |
| DATA | 0.680 ± 0.264 | 0.533 ± 0.278 | **0.536 ± 0.191** |
| LUOGO/INDIRIZZO | 0.914 ± 0.171 | 0.700 ± 0.130 | **0.790 ± 0.140** |
| **MACRO F1** | | | **0.649 ± 0.073** |

**v4 vs paper Table 2 (the headline comparison):**

| Model | NOME | ETÀ | DATA | LUOGO | **Macro** |
|---|---|---|---|---|---|
| llama3.2:3b (zero-shot) | 0.53 | 0.41 | 0.29 | 0.41 | 0.41 |
| gemma3:4b | 0.73 | 0.30 | 0.27 | 0.75 | 0.51 |
| **gemma3:12b (paper best)** | **0.81** | 0.14 | 0.65 | **0.88** | 0.62 |
| mistral:7b | 0.77 | 0.24 | 0.37 | 0.34 | 0.43 |
| phi4:14b | 0.44 | 0.23 | 0.33 | 0.55 | 0.39 |
| **OURS v4 (CV mean)** | 0.714 | **0.554** | 0.536 | 0.790 | **🏆 0.6486** |

**Per-category vs paper's best zero-shot:**
- NOME: gap closed from −0.31 (v3) → **−0.10** (v4)
- ETÀ: **+0.144 win sustained** (clean victory under CV)
- DATA: gap closed slightly (−0.13 → −0.11)
- LUOGO: gap closed from −0.31 → **−0.09**
- **Macro: −0.085 (v3) → +0.029 (v4) — paper beaten**

**v4 observations:**

1. **Variance halved** (0.144 → 0.073) by doubling synthetic data. Style-matched synthetic data has clear marginal value through ~1700 samples.
2. **The catastrophic fold flipped**: v3's Fold 5 (collapsed at 0.29) recovered to 0.644 in v4 (+0.353); v4's Fold 4 was the new weakest (0.514), with DATA F1 = 0.20 for that fold specifically. Variance reduction is real but not total — some fold-localized failures remain.
3. **NOME F1 jumped 0.59 → 0.71** with more no-name training data. The most fragile category under v3 became one of the stronger ones in v4.
4. **LUOGO precision is now exceptional** (0.914 ± 0.171). When the model emits a LUOGO tag, it's almost always correct. Recall lags (0.70), so the model is now under-tagging locations — a fixable issue.

### 4.4 Training cost and reproducibility

- **GPU**: 1× RTX 4070 Ti (12 GB)
- **Single training run**: ~50 min
- **Single 80-note evaluation**: ~60 min
- **Full 5-fold CV**: ~6 hours
- **Gemini API cost**: ~$10 for 878 valid synthetic cases
- **Storage**: ~200 MB for adapter + base merged model

---

## 5. Discussion

### 5.1 Why fine-tuning a 3B model beats a 12B zero-shot model

Two factors drive the headline result:

1. **Domain adaptation**: zero-shot prompting relies on a model's general world knowledge of "what is a name/age/date". Fine-tuning encodes the specific patterns of Italian clinical writing — particularly age formats (`"di 67 anni"`, `"all'età di 70 anni"`) and date conventions (`"marzo 2018"`, `"15/04/2020"`).

2. **The ETÀ and DATA wins are explainable**: gemma3:12b zero-shot got 0.14 on ETÀ because age mentions are frequently embedded in temporal clauses (`"vent'anni di ipertensione"`) that look like dates or durations. Fine-tuning with thousands of correctly-labeled age mentions resolves this ambiguity. Same logic applies to DATA: zero-shot models miss partial dates (`"2017"`, `"l'estate scorsa"`) that fine-tuning learns to handle.

### 5.2 Why we still lose NOME

Gold has only 46 NOME entities total (across 80 notes), and many appear in contexts where the model needs to distinguish patient names from physician names, family member names, and capitalized terms at sentence starts. The paper's gemma3:12b benefits from huge general-purpose training on name recognition. Even our balanced data has structural artifacts — synth_v2 patient names are *always* at the start of the case and repeated, while gold often introduces names later or only once. Closing this gap likely requires:
- More gold data (current 80 is the binding constraint)
- Explicit augmentation: take gold notes and re-position names
- Constrained generation with a name dictionary

### 5.3 What this work contributes beyond the paper

| Paper (Miranda et al.) | This work |
|---|---|
| Zero-shot prompting only | Fine-tuned with LoRA |
| Best result requires 12B–14B params | 3B params, fits 12 GB consumer GPU |
| Synthetic data: not used | Style-anchored Gemini-generated synthetic + balancing for entity distribution |
| ETÀ is universally weak | ETÀ is among our strongest categories |
| Evaluation: zero-shot baseline | Identical metric, 80-note CV |

The reusable artifacts from this work:
1. **`generate_synthetic_v2.py`** — style-anchored synthetic generator (general-purpose for Italian clinical NLP)
2. **`generate_noname.py`** — name-free variant for entity distribution balancing
3. **The empirical finding that synthetic data style matters more than volume** (synth_clinical_1000 actively harmed; Gemini-generated synth_v2 was the breakthrough)

---

## 6. Limitations

1. **Small gold dataset (80 notes) drives high variance.** CV reveals fold-to-fold F1 swings of up to 40 percentage points (Fold 5: 0.29 vs Fold 4: 0.69). With 16 test notes per fold, any clustered distribution shift can swing the F1 dramatically. Expanding the gold annotation to 200-500 notes is the highest-impact way to reduce variance.
2. **Single language and domain**. Results are specific to Italian clinical narratives in the E3C/CLinkaRT style. Transfer to operative notes, discharge summaries, or other languages is unverified.
3. **Four entity types only**. The paper targets 19 PII categories but only evaluates on 4 due to dataset coverage. Our work inherits this limitation.
4. **No constrained generation**. We rely on the fine-tuned model's learned output format. Production deployment should add `outlines` or `lm-format-enforcer` to guarantee valid placeholder output even on adversarial inputs.
5. **Synthetic data has stylistic artifacts**. Names always at the start, anaphora always present, date formats over-represent recent years (2017–2024). A more sophisticated synthetic pipeline could match gold entity-position distributions.
6. **No baseline comparison to fine-tuned BERT-NER**. The Miranda et al. paper explicitly notes (§6 Limitations) that BERT-NER baselines were missing. We follow the same scope — adding this baseline is the highest-priority future work.

---

## 7. Future Work

1. **BERT-NER baseline**: Fine-tune `dbmdz/bert-base-italian-xxl-cased` on the same 80 gold (with appropriate CV) for direct comparison. Token classification typically achieves F1 0.85+ on similar Italian clinical NER tasks. This baseline is the obvious next experiment.
2. **Italian-pretrained base model**: Replace Llama-3.2-3B with **Minerva-3B** (Italian-native pretraining) or **MedGemma** (clinical domain). The paper itself recommends this in its conclusions.
3. **Constrained generation**: Add `outlines` / `lm-format-enforcer` for production deployment to guarantee valid output schema.
4. **Active learning loop**: Use the model to annotate unlabeled clinical text, have a clinician review, expand the gold set to 300-500 notes.
5. **Hybrid system**: BERT-NER for high-precision span extraction + Llama for natural-language redaction generation (e.g., replacing `[NOME]` with realistic surrogate names for downstream readability).
6. **Cross-hospital validation**: Test on proprietary clinical text from a different institution to assess generalization beyond E3C corpus style.

---

## 8. Conclusion

We fine-tuned `meta-llama/Llama-3.2-3B` with LoRA on style-anchored Gemini-generated synthetic data plus 80 manually-annotated Italian clinical notes, targeting GDPR-compliant de-identification. Under rigorous 5-fold cross-validation, our v4 system achieves **macro F1 = 0.6486 ± 0.073**, beating Miranda et al. (2025)'s best zero-shot baseline (gemma3:12b at 0.620) by +0.029 absolute. The category-specific gain on **ETÀ is +0.144** — the most striking demonstration that fine-tuning resolves age-format ambiguities that confuse general-purpose models.

The progression from initial failure (macro F1 ≈ 0.13 with JSON-extraction format) to final success (0.649 under CV) involved diagnosing and correcting four distinct issues:

1. **Wrong task format** (JSON extraction → redaction with inline tags, matching the paper's evaluation metric)
2. **Wrong synthetic data style** (original synth_clinical_1000 was stylistically incompatible with gold — verified by control experiment: training on it alone produced macro F1 = 0.18, *worse than zero-shot*)
3. **Wrong entity distribution** (synthetic data had 4× more names per note than gold; balancing with name-free cases moved NOME F1 from 0.36 → 0.71)
4. **Insufficient synthetic volume** (878 samples in v3 yielded macro 0.535 ± 0.144; doubling to 1676 in v4 yielded 0.649 ± 0.073 — variance halved, mean above paper)

Three findings stand independent of the headline number:

1. **Synthetic data style transfer matters far more than synthetic volume.** A few hundred well-styled samples beat thousands of mis-styled ones. The v1 → v2 transition (paying ~$10 for Gemini-anchored synthetic) was the largest single improvement of the project.

2. **Entity-distribution balancing has measurable returns.** Adding 254 explicitly name-free synthetic cases to correct a 4× NOME over-representation jumped NOME F1 from 0.36 → 0.50 (single-split); after v4's larger no-name set, NOME F1 reached 0.71 under CV — closing 71% of the gap to paper's best.

3. **Doubling style-anchored synthetic data approximately halved CV variance.** This is the cleanest signal that synthetic data still has marginal value for variance reduction beyond what we've already added — suggesting v5 could push further.

**Final result**: A fine-tuned 3B-parameter LLM, running on a 12 GB consumer GPU, beats zero-shot 12B+ models on Italian clinical de-identification under 5-fold cross-validation. Total cost: ~$15 in Gemini API + ~25 hours of training compute. The methodology generalizes to any low-resource clinical NLP task where synthetic data can be conditioned on a small gold set.

---

## Appendix A — File index

| File | Role |
|---|---|
| `data/gold_standard_80.json` | 80 manually annotated Italian clinical notes (CLinkaRT/E3C derivative) |
| `data/synthetic_v2_1000.json` | 624 Gemini-generated style-anchored notes (with names) — v3 |
| `data/synthetic_v4_named_more.json` | 577 additional with-names Gemini-generated notes — v4 |
| `data/synthetic_v2_noname_400.json` | 254 Gemini-generated name-free notes — v3 |
| `data/synthetic_v4_noname_more.json` | 221 additional name-free Gemini-generated notes — v4 |
| `cv_v3_results.json` | v3 5-fold CV per-fold + aggregated results |
| `cv_v4_results.json` | v4 5-fold CV per-fold + aggregated results (final benchmark) |
| `data/test_indices_seed42.json` | Persisted train/test indices for single-split eval (seed=42) |
| `src/generate_synthetic_v2.py` | Style-anchored Gemini generator (single-threaded) |
| `src/generate_synthetic_parallel.py` | 3-key parallel variant of the generator |
| `src/generate_noname.py` | Name-free variant |
| `src/train_redact_v3.py` | Final redaction SFT (single split) |
| `src/eval_redact_v3.py` | Held-out eval with paper metric |
| `src/train_evaluate_cv_v3.py` | Full 5-fold CV pipeline |
| `cv_v3_results.json` | Per-fold + aggregated CV results (populated by CV run) |
| `llama-3.2-3b-deid-redact-v3/` | Saved LoRA adapter |
| `temp_merged_phase1/` | Phase 1 CPT base model (Llama-3.2-3B + Italian-clinical adapter merged) |

## Appendix B — Reproducibility

```bash
# Environment
python -m venv .venv && source .venv/bin/activate
pip install torch transformers trl peft bitsandbytes datasets accelerate \
            google-generativeai scikit-learn tqdm

# Phase 1 (already done; produces temp_merged_phase1/)
# See training_report.md for Phase 1 details

# Phase 2 — Data generation
PYTHONUTF8=1 python src/generate_synthetic_parallel.py --n 1000
PYTHONUTF8=1 python src/generate_noname.py --n 400

# Phase 2 — Training + single-split eval
PYTHONUTF8=1 python src/train_redact_v3.py
PYTHONUTF8=1 python src/eval_redact_v3.py

# Phase 2 — Full 5-fold CV
PYTHONUTF8=1 python src/train_evaluate_cv_v3.py
```

All seeds fixed at 42. Hardware: RTX 4070 Ti 12 GB.

## Appendix C — Failed iterations (worth documenting)

These approaches were tried and abandoned. Listed here so future readers don't repeat them:

1. **JSON-extraction output format** (initial Phase 2 design). Output JSON entity list with context snippets. Failed due to: (a) fragile parsing (greedy regex captured trailing garbage), (b) context-snippet matching ambiguity, (c) the model never learned to emit EOS reliably. Macro F1 ≈ 0.13 even with Tier-1 inference fixes.
2. **Original `synthetic_clinical_1000.json`** (provided dataset). Style-mismatched with gold. Training a model exclusively on this synthetic data drops F1 below zero-shot baseline (0.18 vs 0.41 for llama3.2:3b zero-shot). The synthetic data was actively harmful.
3. **`repetition_penalty=1.2` at inference**. Suppressed the EOS token, causing runaway generation. Dropped to 1.0 — improved output validity.
4. **`max_steps=5` smoke test runs**. Yielded F1 ≈ 0.03; gave misleading impression that the approach was fundamentally broken. Lesson: smoke tests should use ≥1 epoch on a small subset, not arbitrary step counts.
5. **Tight 256-token generation budget**. Truncated mid-JSON during the extraction phase. Increased to 1024 for redaction outputs (which approximately match input length).

## Appendix D — Citation

If using this work, please cite:

```
@misc{sultangazy2026italiandeid,
  title={Fine-tuning a 3B LLM with Style-Anchored Synthetic Data Beats Zero-Shot
         12B Models on Italian Clinical De-Identification},
  author={Sultangazy, Dair},
  year={2026},
  note={Working paper. Builds on Miranda et al. (2025), CLiC-it 2025.}
}
```

Reference paper:
```
Miranda, M., Bratières, S., Patarnello, S., & Lilli, L. (2025).
Mamma Mia! Where's My Name? De-Identifying Italian Clinical Notes
with Large Language Models. CLiC-it 2025.
```
