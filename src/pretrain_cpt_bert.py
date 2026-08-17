"""
Domain-adaptive continual pretraining (MLM) for the BERT-NER baseline.

Mirrors the LLM pipeline's Phase 1 CPT step: before any task-specific
fine-tuning, adapt the base model to Italian clinical/pharmaceutical language
via unsupervised masked-language-modeling on domain text. The LLM's own Phase 1
script and merged checkpoint were never committed to this repo (only the
corpora it used are named in FINAL_REPORT.md / thesis.tex), so this is a fresh
implementation for BERT using the same two named corpora, not a port of
missing code.

Corpora:
  - NLP-FBK/dyspnea-clinical-notes ('it' split): 2,667 Italian clinical notes.
  - praiselab-picuslab/DART: 16,029 Italian drug package inserts (SmPC/RCP-style
    regulatory text -- indications, dosage, contraindications, interactions,
    adverse effects), one row per medicinal product. Gated on HF Hub -- requires
    `hf auth login` with an account that has been granted access.

DART is deterministically subsampled (seed=42) to keep training wall-clock time
reasonable on this hardware (Apple M4, MPS, no CUDA) -- documented here rather
than silently truncated.
"""
import os, random
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)

MODEL_NAME = "dbmdz/bert-base-italian-xxl-cased"
OUTPUT_DIR = "./bert_it_clinical_cpt"
SEED = 42
MAX_LENGTH = 512
NUM_EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
DART_SAMPLE_SIZE = 6000

DART_FIELDS = [
    "04.1 Indicazioni terapeutiche",
    "04.2 Posologia e modo di somministrazione",
    "04.3 Controindicazioni",
    "04.5 Interazioni con altri medicinali ed altre forme di interazione",
    "04.8 Effetti indesiderati",
]


def load_dyspnea_texts():
    ds = load_dataset("NLP-FBK/dyspnea-clinical-notes", split="it")
    texts = [r["clinical_note"] for r in ds if r.get("clinical_note")]
    print(f"  dyspnea-clinical-notes: {len(texts)} documents")
    return texts


def load_dart_texts():
    ds = load_dataset("praiselab-picuslab/DART", split="DART")
    rng = random.Random(SEED)
    idxs = sorted(rng.sample(range(len(ds)), min(DART_SAMPLE_SIZE, len(ds))))
    sub = ds.select(idxs)
    texts = []
    for r in sub:
        parts = [r.get("Nome Medicinale", "") or ""]
        for f in DART_FIELDS:
            v = r.get(f)
            if v and str(v).strip():
                parts.append(str(v).strip())
        text = "\n".join(p for p in parts if p)
        if text.strip():
            texts.append(text)
    print(f"  DART (subsampled, seed={SEED}, n={DART_SAMPLE_SIZE}): {len(texts)} documents")
    return texts


def main():
    print("Loading CPT corpora...")
    texts = load_dyspnea_texts() + load_dart_texts()
    print(f"  TOTAL: {len(texts)} documents")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = Dataset.from_dict({"text": texts})

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    print(f"  tokenized: {len(ds_tok)} examples")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
    steps_per_epoch = -(-len(ds_tok) // BATCH_SIZE)  # ceil div
    warmup_steps = int(steps_per_epoch * NUM_EPOCHS * 0.05)
    args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}_ckpts",
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
    trainer = Trainer(model=model, args=args, train_dataset=ds_tok,
                       data_collator=collator, processing_class=tokenizer)
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved domain-adapted checkpoint -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
