"""
BERT-NER 5-fold CV baseline.

Fine-tunes an Italian BERT token classifier (dbmdz/bert-base-italian-xxl-cased)
on the same gold + synthetic data mix and the same 5-fold CV protocol
(KFold(n_splits=5, shuffle=True, random_state=42) over the 80 gold notes) used
by train_evaluate_cv_v4.py, so the result slots directly into the existing
comparison table.

Entities in gold_standard_80.json and the synthetic files carry no character
offsets, only (type, text) pairs. Spans are recovered by finding every
non-overlapping occurrence of each unique entity string in the note (longest
entities first) -- the same "replace all occurrences" assumption already used
by redact_gold() in train_evaluate_cv_v4.py to build training targets for the
LLM pipeline.

Scoring reuses paper_metric() verbatim from the LLM CV scripts: predicted spans
are spliced back into the note as [TYPE] placeholders and scored by
placeholder-count P/R/F1, not span-overlap, so the number is directly
comparable to the LLM headline table.
"""
import os, json, re, gc, time
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
)
from sklearn.model_selection import KFold
from tqdm import tqdm

MODEL_NAME = "dbmdz/bert-base-italian-xxl-cased"
CV_OUTPUT_BASE = "./cv_bert_outputs"
SYNTH_NAMED_FILES = [
    "./data/synthetic_v2_1000.json",
    "./data/synthetic_v4_named_more.json",
]
SYNTH_NONAME_FILES = [
    "./data/synthetic_v2_noname_400.json",
    "./data/synthetic_v4_noname_more.json",
]
GOLD_PATH = "./data/gold_standard_80.json"
RESULTS_PATH = "./cv_bert_results.json"
SEED = 42
MAX_LENGTH = 512
NUM_EPOCHS = 4
BATCH_SIZE = 8
LEARNING_RATE = 3e-5

VALID_TYPES = ["NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"]
LABELS = ["O"] + [f"{p}-{t}" for t in VALID_TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def sanitize_type(etype):
    etype = (etype or "").upper()
    if "LUOGO" in etype or "INDIRIZZO" in etype:
        return "LUOGO/INDIRIZZO"
    if etype in VALID_TYPES:
        return etype
    return None


def paper_metric(redacted, gold_entities):
    by_type = {t: [] for t in VALID_TYPES}
    for e in gold_entities:
        et = sanitize_type(e.get("type", ""))
        if et:
            by_type[et].append(e.get("text", ""))
    result = {}
    for t in VALID_TYPES:
        texts = by_type[t]
        n_gold = len(texts)
        fn = sum(1 for x in texts if x and x in redacted)
        tp = n_gold - fn
        fp = max(0, redacted.count(f"[{t}]") - n_gold)
        result[t] = (tp, fp, fn)
    return result


def entities_to_spans(text, entities):
    """Dedup by (type, text), longest first, then mark every non-overlapping
    occurrence of each entity string as a char span."""
    seen, unique = set(), []
    for e in entities:
        et = sanitize_type(e.get("type", ""))
        etext = e.get("text", "")
        if et and etext and (et, etext) not in seen:
            seen.add((et, etext))
            unique.append((et, etext))
    unique.sort(key=lambda x: -len(x[1]))

    occupied = []
    spans = []
    for et, etext in unique:
        for m in re.finditer(re.escape(etext), text):
            s, e = m.start(), m.end()
            if any(s < oe and e > os_ for os_, oe in occupied):
                continue
            occupied.append((s, e))
            spans.append((s, e, et))
    spans.sort(key=lambda x: x[0])
    return spans


def load_rows(paths, label):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  [{label}] WARNING: file missing, skipping: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            if r.get("text") and r.get("entities") is not None:
                rows.append({"text": r["text"], "entities": r["entities"]})
        print(f"  [{label}] loaded {p}: {len(data)} records")
    return rows


def load_crf_rows():
    crf_ds = load_dataset("NLP-FBK/synthetic-crf-train", split="it")
    rows = [{"text": r["clinical_note"], "entities": []}
            for r in crf_ds if r.get("clinical_note")]
    print(f"  [crf] {len(rows)} negative (no-entity) samples")
    return rows


def rows_to_dataset(rows):
    starts, ends, types, texts = [], [], [], []
    for r in rows:
        spans = entities_to_spans(r["text"], r["entities"])
        texts.append(r["text"])
        starts.append([s for s, e, t in spans])
        ends.append([e for s, e, t in spans])
        types.append([t for s, e, t in spans])
    return Dataset.from_dict({"text": texts, "starts": starts, "ends": ends, "types": types})


def build_char_tags(text, spans):
    tags = ["O"] * len(text)
    for s, e, t in spans:
        e = min(e, len(text))
        if s >= e:
            continue
        tags[s] = "B-" + t
        for i in range(s + 1, e):
            tags[i] = "I-" + t
    return tags


def make_tokenize_fn(tokenizer):
    """Splits each note into non-overlapping chunks of up to (MAX_LENGTH - 2)
    wordpieces instead of truncating, so notes longer than the model's window
    (55.8% of the synthetic set, 40% of gold) still contribute full-note
    training signal rather than having their tail silently dropped."""
    window = MAX_LENGTH - 2  # room for CLS/SEP
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id

    def tokenize_and_align(batch):
        all_input_ids, all_attention, all_labels = [], [], []
        for i in range(len(batch["text"])):
            text = batch["text"][i]
            spans = list(zip(batch["starts"][i], batch["ends"][i], batch["types"][i]))
            char_tags = build_char_tags(text, spans)
            full = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = full["offset_mapping"]
            for start in range(0, max(len(offsets), 1), window):
                chunk_offsets = offsets[start:start + window]
                if not chunk_offsets:
                    continue
                chunk_ids = full["input_ids"][start:start + window]
                input_ids = [cls_id] + chunk_ids + [sep_id]
                labels = [-100]
                for ts, te in chunk_offsets:
                    labels.append(-100 if ts == te else LABEL2ID[char_tags[ts]])
                labels.append(-100)
                all_input_ids.append(input_ids)
                all_attention.append([1] * len(input_ids))
                all_labels.append(labels)
        return {"input_ids": all_input_ids, "attention_mask": all_attention, "labels": all_labels}
    return tokenize_and_align


def get_train_dataset(train_gold, synth_rows, crf_rows):
    ds_synth = rows_to_dataset(synth_rows)
    ds_gold = rows_to_dataset(train_gold)
    ds_gold_x5 = concatenate_datasets([ds_gold] * 5)
    ds_crf = rows_to_dataset(crf_rows)
    print(f"  synth: {len(ds_synth)}, gold x5: {len(ds_gold_x5)}, crf: {len(ds_crf)}")
    return concatenate_datasets([ds_synth, ds_gold_x5, ds_crf])


@torch.no_grad()
def predict_spans(model, tokenizer, text, device):
    """Processes the whole note in non-overlapping (MAX_LENGTH - 2)-token
    chunks instead of truncating at MAX_LENGTH, so entities past the model's
    window get a real prediction instead of being silently excluded from the
    redacted output (which previously let them dodge scoring as a miss)."""
    window = MAX_LENGTH - 2
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    full = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = full["offset_mapping"]

    spans, cur = [], None
    real_ends = [0]
    for start in range(0, max(len(offsets), 1), window):
        chunk_offsets = offsets[start:start + window]
        if not chunk_offsets:
            continue
        chunk_ids = full["input_ids"][start:start + window]
        input_ids = torch.tensor([[cls_id] + chunk_ids + [sep_id]], device=device)
        attention_mask = torch.ones_like(input_ids)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
        pred_ids = logits.argmax(-1).tolist()[1:-1]  # drop CLS/SEP positions

        cur = None  # a span can't continue across a chunk boundary
        for (ts, te), pid in zip(chunk_offsets, pred_ids):
            if ts == te:
                continue
            real_ends.append(te)
            label = ID2LABEL[pid]
            if label == "O":
                if cur:
                    spans.append(tuple(cur)); cur = None
                continue
            prefix, etype = label.split("-", 1)
            if prefix == "B" or cur is None or cur[2] != etype:
                if cur:
                    spans.append(tuple(cur))
                cur = [ts, te, etype]
            else:
                cur[1] = te
        if cur:
            spans.append(tuple(cur))
    processed_end = max(real_ends)
    return spans, processed_end


def spans_to_redacted(text, spans, processed_end):
    spans = sorted(spans, key=lambda s: s[0])
    out, cursor = [], 0
    for s, e, t in spans:
        out.append(text[cursor:s])
        out.append(f"[{t}]")
        cursor = e
    out.append(text[cursor:processed_end])
    return "".join(out)


def evaluate_fold(model, tokenizer, test_gold, device):
    model.eval()
    totals = {t: [0, 0, 0] for t in VALID_TYPES}
    for row in tqdm(test_gold, desc="Eval fold"):
        text = row.get("text", "")
        spans, processed_end = predict_spans(model, tokenizer, text, device)
        redacted = spans_to_redacted(text, spans, processed_end)
        scores = paper_metric(redacted, row.get("entities", []))
        for t, (tp, fp, fn) in scores.items():
            totals[t][0] += tp; totals[t][1] += fp; totals[t][2] += fn
    out = {}
    for t in VALID_TYPES:
        tp, fp, fn = totals[t]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        out[t] = {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f1}
    return out


def run_cv():
    os.makedirs(CV_OUTPUT_BASE, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold = json.load(f)
    gold_arr = np.array(gold)

    print("Loading synthetic + CRF data (shared across folds)...")
    synth_rows = load_rows(SYNTH_NAMED_FILES, "named") + load_rows(SYNTH_NONAME_FILES, "noname")
    crf_rows = load_crf_rows()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    all_results = {"folds": [], "config": {
        "seed": SEED, "model": MODEL_NAME, "max_length": MAX_LENGTH,
        "num_epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
        "synth_named": SYNTH_NAMED_FILES, "synth_noname": SYNTH_NONAME_FILES,
    }}

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(gold_arr)):
        fold_num = fold_idx + 1
        t0 = time.time()
        print(f"\n{'='*70}\nFOLD {fold_num}/5\n{'='*70}")
        train_gold = gold_arr[train_idx].tolist()
        test_gold = gold_arr[test_idx].tolist()
        print(f"  train gold: {len(train_gold)}, test gold: {len(test_gold)}")

        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        ).to(device)

        train_ds = get_train_dataset(train_gold, synth_rows, crf_rows)
        tokenize_fn = make_tokenize_fn(tokenizer)
        train_ds_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
        print(f"  total train samples: {len(train_ds_tok)}")

        collator = DataCollatorForTokenClassification(tokenizer)
        steps_per_epoch = -(-len(train_ds_tok) // BATCH_SIZE)  # ceil div
        warmup_steps = int(steps_per_epoch * NUM_EPOCHS * 0.1)
        args = TrainingArguments(
            output_dir=f"{CV_OUTPUT_BASE}/fold_{fold_num}",
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="no",
            report_to=[],
            bf16=(device.type == "cuda"),
        )
        trainer = Trainer(model=model, args=args, train_dataset=train_ds_tok,
                           data_collator=collator, processing_class=tokenizer)
        trainer.train()
        t_train = time.time() - t0

        fold_scores = evaluate_fold(model, tokenizer, test_gold, device)
        macro = sum(s["f1"] for s in fold_scores.values()) / 4
        t_total = time.time() - t0
        print(f"\nFold {fold_num} per-category F1:")
        for t in VALID_TYPES:
            s = fold_scores[t]
            print(f"  {t:<22} P={s['p']:.4f} R={s['r']:.4f} F1={s['f1']:.4f}")
        print(f"  Macro F1: {macro:.4f} | time: train {t_train/60:.1f}min, total {t_total/60:.1f}min")

        all_results["folds"].append({"fold": fold_num, "train_size": len(train_gold),
                                      "test_size": len(test_gold), "scores": fold_scores,
                                      "macro_f1": macro})
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        del model, trainer
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*70}\nAGGREGATE (5 folds, mean ± std)\n{'='*70}")
    per_cat = {t: {"p": [], "r": [], "f1": []} for t in VALID_TYPES}
    macros = []
    for fr in all_results["folds"]:
        for t in VALID_TYPES:
            per_cat[t]["p"].append(fr["scores"][t]["p"])
            per_cat[t]["r"].append(fr["scores"][t]["r"])
            per_cat[t]["f1"].append(fr["scores"][t]["f1"])
        macros.append(fr["macro_f1"])
    agg = {}
    for t in VALID_TYPES:
        p_m, p_s = np.mean(per_cat[t]["p"]), np.std(per_cat[t]["p"])
        r_m, r_s = np.mean(per_cat[t]["r"]), np.std(per_cat[t]["r"])
        f_m, f_s = np.mean(per_cat[t]["f1"]), np.std(per_cat[t]["f1"])
        agg[t] = {"p_mean": p_m, "p_std": p_s, "r_mean": r_m, "r_std": r_s, "f1_mean": f_m, "f1_std": f_s}
        print(f"  {t:<22} F1 = {f_m:.4f} ± {f_s:.3f}")
    macro_m, macro_s = np.mean(macros), np.std(macros)
    print(f"  {'MACRO F1':<22} = {macro_m:.4f} ± {macro_s:.3f}")
    all_results["aggregate"] = {"per_category": agg, "macro_f1_mean": macro_m,
                                 "macro_f1_std": macro_s, "fold_macros": macros}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults → {RESULTS_PATH}")


if __name__ == "__main__":
    run_cv()
