"""Eval the redact-v3 adapter (balanced data) on the 16 HELD-OUT gold notes."""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

MERGED_PHASE1_DIR = "./temp_merged_phase1"
REDACT_ADAPTER_DIR = "./llama-3.2-3b-deid-redact-v3"
GOLD_PATH = "./data/gold_standard_80.json"
SPLIT_INDICES_PATH = "./data/test_indices_seed42.json"

SYSTEM_PROMPT = (
    "Sei un assistente specializzato nella de-identificazione di referti clinici italiani, "
    "in conformità con il GDPR. Sostituisci tutte le informazioni sensibili con i tag "
    "[NOME], [ETÀ], [DATA], [LUOGO/INDIRIZZO]. "
    "Restituisci ESCLUSIVAMENTE il testo de-identificato, senza commenti o spiegazioni."
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
    etype = (etype or "").upper()
    if "LUOGO" in etype or "INDIRIZZO" in etype:
        return "LUOGO/INDIRIZZO"
    if etype in VALID_TYPES:
        return etype
    return None


def paper_metric(redacted, gold_entities):
    gold_by_type = {t: [] for t in VALID_TYPES}
    for e in gold_entities:
        et = sanitize_type(e.get("type", ""))
        if et:
            gold_by_type[et].append(e.get("text", ""))
    result = {}
    for t in VALID_TYPES:
        texts = gold_by_type[t]
        n_gold = len(texts)
        fn = sum(1 for x in texts if x and x in redacted)
        tp = n_gold - fn
        fp = max(0, redacted.count(f"[{t}]") - n_gold)
        result[t] = (tp, fp, fn)
    return result


def main():
    with open(SPLIT_INDICES_PATH, 'r', encoding='utf-8') as f:
        split = json.load(f)
    with open(GOLD_PATH, 'r', encoding='utf-8') as f:
        gold = json.load(f)
    test_gold = [gold[i] for i in split["test"]]
    print(f"Test set: {len(test_gold)} held-out gold notes (seed={split['seed']})\n")

    tokenizer = AutoTokenizer.from_pretrained(MERGED_PHASE1_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MERGED_PHASE1_DIR, quantization_config=quant, device_map="auto")
    model = PeftModel.from_pretrained(model, REDACT_ADAPTER_DIR)
    model.eval()

    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tokenizer.eos_token_id, eot] if eot != tokenizer.eos_token_id else [tokenizer.eos_token_id]

    totals = {t: [0, 0, 0] for t in VALID_TYPES}
    for i, row in enumerate(tqdm(test_gold, desc="Held-out 16 (v3)")):
        text = row.get("text", "")
        ids = tokenizer.encode(text, add_special_tokens=False)
        note = text if len(ids) <= 700 else tokenizer.decode(ids[:700])
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": note}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                                     repetition_penalty=1.0, eos_token_id=eos_ids,
                                     pad_token_id=tokenizer.eos_token_id)
        redacted = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        scores = paper_metric(redacted, row.get("entities", []))
        for t, (tp, fp, fn) in scores.items():
            totals[t][0] += tp; totals[t][1] += fp; totals[t][2] += fn

    print("\n" + "=" * 70)
    print("HELD-OUT 16 RESULTS — redact-v3 (with no-name balancing)")
    print("=" * 70)
    print(f"{'Category':<22}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>10}{'R':>10}{'F1':>10}")
    print("-" * 70)
    macro = 0
    ours = {}
    for t in VALID_TYPES:
        tp, fp, fn = totals[t]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        macro += f1
        ours[t] = f1
        print(f"{t:<22}{tp:>6}{fp:>6}{fn:>6}{p:>10.4f}{r:>10.4f}{f1:>10.4f}")
    macro /= 4
    print("-" * 70)
    print(f"{'MACRO F1':<46}{'':>10}{'':>10}{macro:>10.4f}")

    print("\n" + "=" * 70)
    print("vs PAPER + v2 history")
    print("=" * 70)
    rows = [
        ("llama3.2:3b (paper)",  {"NOME":0.53,"ETÀ":0.41,"DATA":0.29,"LUOGO/INDIRIZZO":0.41}),
        ("gemma3:4b (paper)",    {"NOME":0.73,"ETÀ":0.30,"DATA":0.27,"LUOGO/INDIRIZZO":0.75}),
        ("gemma3:12b (paper)",   {"NOME":0.81,"ETÀ":0.14,"DATA":0.65,"LUOGO/INDIRIZZO":0.88}),
        ("mistral:7b (paper)",   {"NOME":0.77,"ETÀ":0.24,"DATA":0.37,"LUOGO/INDIRIZZO":0.34}),
        ("phi4:14b (paper)",     {"NOME":0.44,"ETÀ":0.23,"DATA":0.33,"LUOGO/INDIRIZZO":0.55}),
        ("OURS v2 (16-split)",   {"NOME":0.36,"ETÀ":0.55,"DATA":0.86,"LUOGO/INDIRIZZO":0.77}),
    ]
    print(f"{'Model':<22}" + "".join(f"{t:>12}" for t in VALID_TYPES) + f"{'MACRO':>10}")
    print("-" * 84)
    for name, sc in rows:
        m = sum(sc.values()) / 4
        print(f"{name:<22}" + "".join(f"{sc[t]:>12.2f}" for t in VALID_TYPES) + f"{m:>10.3f}")
    print(f"{'OURS v3 (16-split)':<22}" + "".join(f"{ours[t]:>12.4f}" for t in VALID_TYPES) + f"{macro:>10.4f}")


if __name__ == "__main__":
    main()
