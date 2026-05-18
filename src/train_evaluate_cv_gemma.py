"""
5-fold CV for Gemma 3 instruction-tuned models (1B or 4B).

Usage:
    python src/train_evaluate_cv_gemma.py --model_size 1b
    python src/train_evaluate_cv_gemma.py --model_size 4b

Same data mix as v4:
  - synthetic_v2_1000.json (624 named)
  - synthetic_v4_named_more.json (~577 named)
  - synthetic_v2_noname_400.json (254 no-name)
  - synthetic_v4_noname_more.json (221 no-name)
  - 80 gold (5x oversampled = 400 per fold ~64 train)
  - 80 CRF negatives

Gemma 3 uses its own built-in chat template; no manual template needed.
"""
import argparse, os, json, time, gc
import torch
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import KFold
from tqdm import tqdm

SYNTH_NAMED_FILES = [
    "./data/synthetic_v2_1000.json",
    "./data/synthetic_v4_named_more.json",
]
SYNTH_NONAME_FILES = [
    "./data/synthetic_v2_noname_400.json",
    "./data/synthetic_v4_noname_more.json",
]
GOLD_PATH = "./data/gold_standard_80.json"
SEED = 42

SYSTEM_PROMPT = (
    "Sei un assistente specializzato nella de-identificazione di referti clinici italiani, "
    "in conformità con il GDPR. Sostituisci tutte le informazioni sensibili con i tag "
    "[NOME], [ETÀ], [DATA], [LUOGO/INDIRIZZO]. "
    "Restituisci ESCLUSIVAMENTE il testo de-identificato, senza commenti o spiegazioni."
)

VALID_TYPES = ["NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"]

MODEL_IDS = {
    "1b": "google/gemma-3-1b-it",
    "4b": "google/gemma-3-4b-it",
}


def sanitize_type(etype):
    etype = (etype or "").upper()
    if "LUOGO" in etype or "INDIRIZZO" in etype:
        return "LUOGO/INDIRIZZO"
    if etype in VALID_TYPES:
        return etype
    return None


def redact_gold(text, entities):
    seen, unique = set(), []
    for e in entities:
        et = sanitize_type(e.get("type", ""))
        etext = e.get("text", "")
        if et and etext and (et, etext) not in seen:
            seen.add((et, etext))
            unique.append((et, etext))
    unique.sort(key=lambda x: -len(x[1]))
    out = text
    for et, etext in unique:
        out = out.replace(etext, f"[{et}]")
    return out


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


def load_synth_files(paths, label):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  [{label}] WARNING: missing {p}, skipping")
            continue
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for r in data:
            if r.get("text") and r.get("redacted_text"):
                rows.append({"input": r["text"], "output": r["redacted_text"]})
        print(f"  [{label}] {p}: {len(data)} records")
    print(f"  [{label}] total: {len(rows)}")
    return Dataset.from_list(rows)


def process_gold(gold_list):
    rows = []
    for r in gold_list:
        text = r.get("text", "")
        if text:
            rows.append({"input": text, "output": redact_gold(text, r.get("entities", []))})
    return Dataset.from_list(rows)


def get_train_dataset(train_gold):
    ds_named = load_synth_files(SYNTH_NAMED_FILES, "named")
    ds_noname = load_synth_files(SYNTH_NONAME_FILES, "noname")
    ds_gold = process_gold(train_gold)
    ds_gold_x5 = concatenate_datasets([ds_gold] * 5)
    print(f"  gold x5: {len(ds_gold_x5)}")
    crf_ds = load_dataset('NLP-FBK/synthetic-crf-train', split='it')
    crf_rows = [{"input": r["clinical_note"], "output": r["clinical_note"]}
                for r in crf_ds if r.get("clinical_note")]
    ds_crf = Dataset.from_list(crf_rows)
    print(f"  crf: {len(ds_crf)}")
    combined = concatenate_datasets([ds_named, ds_noname, ds_gold_x5, ds_crf])

    def to_messages(ex):
        return {"messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": ex["input"]},
            {"role": "assistant", "content": ex["output"]},
        ]}
    return combined.map(to_messages, remove_columns=combined.column_names)


def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    return get_peft_model(model, lora_cfg), tokenizer


def get_eos_ids(tokenizer):
    eot_candidates = ["<end_of_turn>", "<|eot_id|>", "<eos>"]
    ids = {tokenizer.eos_token_id}
    for tok in eot_candidates:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid and tid != tokenizer.unk_token_id:
            ids.add(tid)
    return list(ids)


def evaluate_fold(model, tokenizer, test_gold):
    model.eval()
    eos_ids = get_eos_ids(tokenizer)
    totals = {t: [0, 0, 0] for t in VALID_TYPES}
    for row in tqdm(test_gold, desc="Eval fold"):
        text = row.get("text", "")
        ids = tokenizer.encode(text, add_special_tokens=False)
        note = text if len(ids) <= 700 else tokenizer.decode(ids[:700])
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": note}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1024, do_sample=False,
                repetition_penalty=1.0, eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        redacted = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
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


def run_cv(model_size):
    model_id = MODEL_IDS[model_size]
    cv_output_base = f"./cv_gemma{model_size}_outputs"
    results_path = f"./cv_gemma{model_size}_results.json"
    os.makedirs(cv_output_base, exist_ok=True)

    print(f"\nModel: {model_id}")
    with open(GOLD_PATH, 'r', encoding='utf-8') as f:
        gold = json.load(f)
    gold_arr = np.array(gold)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    all_results = {"folds": [], "config": {"seed": SEED, "model_id": model_id,
                                            "model_size": model_size, "version": f"gemma{model_size}"}}

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(gold_arr)):
        fold_num = fold_idx + 1
        t0 = time.time()
        print(f"\n{'='*70}\nFOLD {fold_num}/5\n{'='*70}")
        train_gold = gold_arr[train_idx].tolist()
        test_gold = gold_arr[test_idx].tolist()
        print(f"  train gold: {len(train_gold)}, test gold: {len(test_gold)}")

        model, tokenizer = get_model_and_tokenizer(model_id)
        train_ds = get_train_dataset(train_gold)
        print(f"  total train samples: {len(train_ds)}")

        args = SFTConfig(
            output_dir=f"{cv_output_base}/fold_{fold_num}",
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            logging_steps=20,
            save_strategy="no",
            bf16=True,
            max_grad_norm=1.0,
            max_length=1536,
            assistant_only_loss=True,
        )
        trainer = SFTTrainer(model=model, train_dataset=train_ds, args=args, processing_class=tokenizer)
        trainer.train()
        t_train = time.time() - t0

        fold_scores = evaluate_fold(model, tokenizer, test_gold)
        macro = sum(s["f1"] for s in fold_scores.values()) / 4
        t_total = time.time() - t0

        print(f"\nFold {fold_num} per-category F1:")
        for t in VALID_TYPES:
            s = fold_scores[t]
            print(f"  {t:<22} P={s['p']:.4f} R={s['r']:.4f} F1={s['f1']:.4f}  (TP={s['tp']} FP={s['fp']} FN={s['fn']})")
        print(f"  Macro F1: {macro:.4f} | train {t_train/60:.1f}min, total {t_total/60:.1f}min")

        all_results["folds"].append({"fold": fold_num, "train_size": len(train_gold),
                                     "test_size": len(test_gold), "scores": fold_scores,
                                     "macro_f1": macro})
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        del model, trainer, tokenizer
        gc.collect()
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

    print(f"\n{'Category':<22}{'P (mean+/-std)':>20}{'R (mean+/-std)':>20}{'F1 (mean+/-std)':>20}")
    print("-" * 80)
    agg = {}
    for t in VALID_TYPES:
        p_m, p_s = np.mean(per_cat[t]["p"]), np.std(per_cat[t]["p"])
        r_m, r_s = np.mean(per_cat[t]["r"]), np.std(per_cat[t]["r"])
        f_m, f_s = np.mean(per_cat[t]["f1"]), np.std(per_cat[t]["f1"])
        agg[t] = {"p_mean": p_m, "p_std": p_s, "r_mean": r_m, "r_std": r_s, "f1_mean": f_m, "f1_std": f_s}
        print(f"{t:<22}{p_m:>12.4f} ± {p_s:.3f}{r_m:>12.4f} ± {r_s:.3f}{f_m:>12.4f} ± {f_s:.3f}")

    macro_m, macro_s = np.mean(macros), np.std(macros)
    print("-" * 80)
    print(f"{'MACRO F1':<22}{'':>40}{macro_m:>12.4f} ± {macro_s:.3f}")

    all_results["aggregate"] = {"per_category": agg, "macro_f1_mean": macro_m,
                                "macro_f1_std": macro_s, "fold_macros": macros}
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults -> {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", choices=["1b", "4b"], required=True,
                        help="Gemma 3 model size: 1b or 4b")
    args = parser.parse_args()
    run_cv(args.model_size)
