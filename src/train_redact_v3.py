"""
Redact-v3: balanced training with 624 with-names + 400 no-name + gold + CRF.
Goal: fix NOME over-tagging by adding name-free synthetic cases.
"""
import os, json, torch, random
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

TEMP_MERGED_DIR = "./temp_merged_phase1"
OUTPUT_DIR = "./llama-3.2-3b-deid-redact-v3"
SYNTH_WITH_NAMES = "./data/synthetic_v2_1000.json"
SYNTH_NONAME = "./data/synthetic_v2_noname_400.json"
GOLD_PATH = "./data/gold_standard_80.json"
SPLIT_INDICES_PATH = "./data/test_indices_seed42.json"
SEED = 42

LLAMA3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] | trim }}<|eot_id|>{% endgeneration %}"
    "{% else %}"
    "{{ message['content'] | trim }}<|eot_id|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)

SYSTEM_PROMPT = (
    "Sei un assistente specializzato nella de-identificazione di referti clinici italiani, "
    "in conformità con il GDPR. Sostituisci tutte le informazioni sensibili con i tag "
    "[NOME], [ETÀ], [DATA], [LUOGO/INDIRIZZO]. "
    "Restituisci ESCLUSIVAMENTE il testo de-identificato, senza commenti o spiegazioni."
)

VALID_TYPES = {"NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"}


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
        etype = sanitize_type(e.get("type", ""))
        etext = e.get("text", "")
        if etype and etext and (etype, etext) not in seen:
            seen.add((etype, etext))
            unique.append((etype, etext))
    unique.sort(key=lambda x: -len(x[1]))
    out = text
    for etype, etext in unique:
        out = out.replace(etext, f"[{etype}]")
    return out


def load_split():
    with open(GOLD_PATH, 'r', encoding='utf-8') as f:
        gold = json.load(f)
    indices = list(range(len(gold)))
    random.Random(SEED).shuffle(indices)
    test_idx = sorted(indices[:16])
    train_idx = sorted(indices[16:])
    print(f"Gold split: train={len(train_idx)}, test={len(test_idx)}")
    return [gold[i] for i in train_idx]


def load_synth(path, label):
    print(f"Loading {label} from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for r in data:
        if r.get("text") and r.get("redacted_text"):
            rows.append({"input": r["text"], "output": r["redacted_text"]})
    print(f"  {len(rows)} samples")
    return Dataset.from_list(rows)


def process_gold_train(train_gold):
    print("Processing 64 gold train notes...")
    rows = []
    for r in train_gold:
        text = r.get("text", "")
        ents = r.get("entities", [])
        if text:
            rows.append({"input": text, "output": redact_gold(text, ents)})
    print(f"  {len(rows)} gold-train samples")
    return Dataset.from_list(rows)


def process_crf():
    print("Loading CRF negatives...")
    ds = load_dataset('NLP-FBK/synthetic-crf-train', split='it')
    rows = [{"input": r["clinical_note"], "output": r["clinical_note"]} for r in ds if r.get("clinical_note")]
    print(f"  {len(rows)} CRF samples")
    return Dataset.from_list(rows)


def assemble_dataset(train_gold):
    ds_synth_named = load_synth(SYNTH_WITH_NAMES, "synth_with_names")
    ds_synth_noname = load_synth(SYNTH_NONAME, "synth_noname")
    ds_gold = process_gold_train(train_gold)
    ds_gold_x5 = concatenate_datasets([ds_gold] * 5)
    print(f"  Gold x5 oversampled: {len(ds_gold_x5)}")
    ds_crf = process_crf()

    combined = concatenate_datasets([ds_synth_named, ds_synth_noname, ds_gold_x5, ds_crf])

    def to_messages(ex):
        return {"messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": ex["input"]},
            {"role": "assistant", "content": ex["output"]},
        ]}
    return combined.map(to_messages, remove_columns=combined.column_names)


def setup_and_train():
    train_gold = load_split()
    print("\nLoading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(TEMP_MERGED_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(TEMP_MERGED_DIR, quantization_config=quant, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)

    print("\nAssembling dataset...")
    train_ds = assemble_dataset(train_gold)
    print(f"  Total training samples: {len(train_ds)}")

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        max_grad_norm=1.0,
        max_length=1536,
        assistant_only_loss=True,
    )
    trainer = SFTTrainer(model=model, train_dataset=train_ds, args=args, processing_class=tokenizer)
    print("\nStarting training...")
    trainer.train()
    print("Saving adapter...")
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    setup_and_train()
