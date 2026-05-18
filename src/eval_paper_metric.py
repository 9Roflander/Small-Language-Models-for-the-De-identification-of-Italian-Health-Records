"""
Phase 1: Reproduce the paper's deterministic evaluation on our v2 adapter.

Paper's metric (Section 4.3):
- TP per category = (#gold entities of type T) - (#gold entity texts still present in redacted output)
- FN per category = #gold entity texts still present in redacted output
- FP per category = max(0, #[T] placeholders in output - #gold entities of type T)

Pipeline:
1. Run our extractor (v2 adapter, Tier 1 inference settings) -> JSON entities
2. Build a redacted version by replacing each predicted entity's text with [TYPE] in original note
3. Apply paper's metric

CAVEAT: v2 was trained on all 80 gold notes. These numbers are training-set scores
(upper bound). For a clean comparison vs paper (zero-shot), need 5-fold CV next phase.
But this establishes whether the approach can in principle beat their numbers.
"""

import json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

MERGED_PHASE1_DIR = "./temp_merged_phase1"
PHASE2_ADAPTER_DIR = "./llama-3.2-3b-deid-sft-v2"
GOLD_PATH = "./data/gold_standard_80.json"

SYSTEM_PROMPT = (
    "Sei un assistente medico esperto specializzato in de-identificazione. "
    "Estrai tutte le informazioni sensibili dal seguente referto clinico. "
    "Restituisci ESCLUSIVAMENTE un array JSON di oggetti, dove ogni oggetto ha due chiavi: "
    "'text' (l'entità sensibile con 2-3 parole di contesto circostante per disambiguare) e "
    "'type' (rigorosamente uno tra NOME, ETÀ, DATA, LUOGO/INDIRIZZO). Se non ci sono entità, restituisci []."
)

LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)

VALID_TYPES = ["NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"]


def sanitize_type(etype):
    etype = etype.upper()
    if "LUOGO" in etype or "INDIRIZZO" in etype:
        return "LUOGO/INDIRIZZO"
    if etype in VALID_TYPES:
        return etype
    return None


def parse_entities(gen_text):
    """Tier 1 parser: raw_decode at first '['."""
    pred = []
    cleaned = gen_text.replace('\n', ' ')
    decoder = json.JSONDecoder()
    idx = cleaned.find('[')
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
            if isinstance(obj, list):
                for item in obj:
                    if not isinstance(item, dict):
                        continue
                    etype = sanitize_type(item.get("type", ""))
                    snippet = item.get("text", "")
                    if etype and snippet:
                        pred.append({"text": snippet, "type": etype})
                return pred
        except json.JSONDecodeError:
            pass
        idx = cleaned.find('[', idx + 1)
    return pred


def redact_text(original, pred_entities):
    """Build redacted text by replacing each predicted entity snippet with [TYPE]."""
    # Sort by length descending so longer snippets get replaced first (avoid partial overlaps)
    sorted_ents = sorted(pred_entities, key=lambda e: -len(e["text"]))
    redacted = original
    for ent in sorted_ents:
        snippet = ent["text"]
        placeholder = f"[{ent['type']}]"
        if snippet and snippet in redacted:
            redacted = redacted.replace(snippet, placeholder)
    return redacted


def paper_metric_for_note(original, redacted, gold_entities):
    """
    Paper's deterministic metric, per category.
    Returns dict: {type: (tp, fp, fn)}
    """
    # Group gold by type (paper allows duplicates: same entity repeated twice counts twice)
    gold_by_type = {t: [] for t in VALID_TYPES}
    for e in gold_entities:
        etype = sanitize_type(e.get("type", ""))
        if not etype:
            continue
        gold_by_type[etype].append(e.get("text", ""))

    result = {}
    for etype in VALID_TYPES:
        gold_texts = gold_by_type[etype]
        n_gold = len(gold_texts)

        # FN: count how many gold texts are still present in redacted output
        # Paper example: "if the input has 2 annotated NAME entities and the text is found
        # only once in the output, this last one is the counter for FN, TP = 2-1 = 1"
        fn = 0
        for t in gold_texts:
            if t and t in redacted:
                fn += 1
        tp = n_gold - fn

        # FP: number of [TYPE] placeholders exceeding gold count
        placeholder = f"[{etype}]"
        n_placeholders = redacted.count(placeholder)
        fp = max(0, n_placeholders - n_gold)

        result[etype] = (tp, fp, fn)
    return result


def main():
    print("Loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_PHASE1_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(MERGED_PHASE1_DIR, quantization_config=quant, device_map="auto")
    model = PeftModel.from_pretrained(model, PHASE2_ADAPTER_DIR)
    model.eval()

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tokenizer.eos_token_id, eot_id] if eot_id != tokenizer.eos_token_id else [tokenizer.eos_token_id]

    with open(GOLD_PATH, 'r', encoding='utf-8') as f:
        gold = json.load(f)
    print(f"Loaded {len(gold)} gold notes\n")

    # Aggregate counts per category
    totals = {t: [0, 0, 0] for t in VALID_TYPES}  # [tp, fp, fn]
    sample_dumps = []

    for i, row in enumerate(tqdm(gold, desc="Eval (paper metric)")):
        text = row.get("text", "")
        gold_entities = row.get("entities", [])

        # Truncate note for inference (keep redaction eval on FULL note)
        system_overhead = len(tokenizer.encode(SYSTEM_PROMPT)) + 30
        max_note_tokens = 900 - system_overhead
        note_for_prompt = text
        note_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(note_ids) > max_note_tokens:
            note_for_prompt = tokenizer.decode(note_ids[:max_note_tokens])

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": note_for_prompt},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                repetition_penalty=1.0, eos_token_id=eos_ids,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        pred_ents = parse_entities(gen_text)
        redacted = redact_text(text, pred_ents)

        scores = paper_metric_for_note(text, redacted, gold_entities)
        for etype, (tp, fp, fn) in scores.items():
            totals[etype][0] += tp
            totals[etype][1] += fp
            totals[etype][2] += fn

        if i < 3:
            sample_dumps.append({
                "note_idx": i,
                "n_pred_ents": len(pred_ents),
                "redacted_preview": redacted[:300],
                "scores": scores,
            })

    print("\n" + "=" * 70)
    print("SAMPLE OUTPUTS (first 3 notes)")
    print("=" * 70)
    for s in sample_dumps:
        print(f"\nNote {s['note_idx']}: {s['n_pred_ents']} entities extracted")
        print(f"  Redacted preview: {s['redacted_preview']!r}")
        print(f"  Per-category (TP, FP, FN): {s['scores']}")

    print("\n" + "=" * 70)
    print("PAPER-METRIC RESULTS (v2 adapter on all 80 gold notes)")
    print("=" * 70)
    print(f"{'Category':<22}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>10}{'R':>10}{'F1':>10}")
    print("-" * 70)
    macro_f1 = 0
    for etype in VALID_TYPES:
        tp, fp, fn = totals[etype]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        macro_f1 += f1
        print(f"{etype:<22}{tp:>6}{fp:>6}{fn:>6}{p:>10.4f}{r:>10.4f}{f1:>10.4f}")
    macro_f1 /= len(VALID_TYPES)
    print("-" * 70)
    print(f"{'MACRO F1':<22}{'':>18}{'':>10}{'':>10}{macro_f1:>10.4f}")

    print("\n" + "=" * 70)
    print("COMPARISON vs Paper Table 2 (Deterministic Evaluation)")
    print("=" * 70)
    paper = {
        "llama3.2:3b":  {"NOME": 0.53, "ETÀ": 0.41, "LUOGO/INDIRIZZO": 0.41, "DATA": 0.29},
        "gemma3:1b":    {"NOME": 0.07, "ETÀ": 0.02, "LUOGO/INDIRIZZO": 0.07, "DATA": 0.12},
        "gemma3:4b":    {"NOME": 0.73, "ETÀ": 0.30, "LUOGO/INDIRIZZO": 0.75, "DATA": 0.27},
        "gemma3:12b":   {"NOME": 0.81, "ETÀ": 0.14, "LUOGO/INDIRIZZO": 0.88, "DATA": 0.65},
        "mistral:7b":   {"NOME": 0.77, "ETÀ": 0.24, "LUOGO/INDIRIZZO": 0.34, "DATA": 0.37},
        "phi4:14b":     {"NOME": 0.44, "ETÀ": 0.23, "LUOGO/INDIRIZZO": 0.55, "DATA": 0.33},
    }
    header = f"{'Model':<18}" + "".join(f"{t:>10}" for t in VALID_TYPES)
    print(header)
    print("-" * 70)
    for model_name, scores in paper.items():
        row = f"{model_name:<18}" + "".join(f"{scores[t]:>10.2f}" for t in VALID_TYPES)
        print(row)
    ours_row = f"{'OURS (v2 FT)':<18}"
    for etype in VALID_TYPES:
        tp, fp, fn = totals[etype]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        ours_row += f"{f1:>10.4f}"
    print(ours_row)
    print("\nNote: OURS evaluated on training-set (v2 saw all 80 in SFT). Upper-bound number.")
    print("Phase 2 will run proper 5-fold CV with redaction-format training for clean comparison.")


if __name__ == "__main__":
    main()
