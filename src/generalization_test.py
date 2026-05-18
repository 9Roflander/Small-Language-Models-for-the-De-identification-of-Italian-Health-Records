"""
Generalization (OOD) test: take gold note 0 and swap all entity values to
non-Italian, never-seen-in-training names/cities/hospitals/dates.

If the model still correctly redacts → it learned the CONCEPT of entity types.
If it fails → it memorized specific Italian-clinical-name patterns.
"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MERGED_PHASE1_DIR = "./temp_merged_phase1"
REDACT_ADAPTER_DIR = "./llama-3.2-3b-deid-redact-v3"

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


def make_ood_note():
    """Note 0 (Anna) with ALL entity values swapped to never-seen-in-training values."""
    # Original gold note 0 + first part of text. Truncated to fit in ~700 tokens.
    # Substitutions (NEVER appearing in any training data):
    #   Anna -> Yusuf, donna -> uomo, 47 -> 62
    #   Bologna -> Almaty (Kazakhstan)
    #   Istituto Rizzoli -> Istituto Saryarka (made-up Kazakh hospital)
    #   gennaio 2017 -> ottobre 2034
    #   marzo 2017 -> dicembre 2034
    #   Trieste -> Tashkent (Uzbekistan)
    #   ASUGI -> KAZMU (made-up Kazakh medical authority)
    text = (
        "Yusuf è un uomo di 62 anni, vive con il figlio in un piccolo appartamento "
        "in un paese di periferia, lavora nel campo della ristorazione e, nonostante "
        "una patologia genetica familiare che provoca delle alterazioni scheletriche, "
        "cardiologiche, oculari e cutanee, vive una vita serena e tranquilla. Ha una "
        "buona rete familiare e informale che supporta la famiglia. Viene seguito da "
        "un centro di riferimento specifico per patologia rara ad Almaty e si "
        "sottopone a controlli clinicostrumentali periodici e regolari. Nell'ottobre "
        "del 2034 esegue un intervento chirurgico di piede torto congenito presso "
        "l'Istituto Saryarka. Durante la degenza, a seguito di un dolore retrosternale "
        "e di una sincope, viene sottoposto a un intervento di endoprotesi di aorta "
        "per dissezione acuta. Al rientro a casa, nel dicembre del 2034, deve "
        "proseguire i controlli cardiologici ed eseguire la fisioterapia per gli "
        "esiti di intervento al piede e regolari controlli ematici. Non essendo "
        "autonomo nella deambulazione, viene preso in carico dai servizi del "
        "Distretto n. 1 della KAZMU di Tashkent. Yusuf sembra, in un primo momento, "
        "una persona collaborante al piano assistenziale predisposto."
    )

    expected_entities = [
        ("NOME", "Yusuf"),
        ("ETÀ", "62"),
        ("LUOGO/INDIRIZZO", "Almaty"),
        ("LUOGO/INDIRIZZO", "Istituto Saryarka"),
        ("DATA", "ottobre del 2034"),
        ("DATA", "dicembre del 2034"),
        ("LUOGO/INDIRIZZO", "Tashkent"),
        ("LUOGO/INDIRIZZO", "KAZMU"),
    ]
    return text, expected_entities


def make_normal_note():
    """A purely-Italian note in similar style — for comparison baseline."""
    text = (
        "Marco è un uomo di 58 anni, vive con la figlia in un piccolo appartamento "
        "in un paese di periferia, lavora nel campo della ristorazione e, nonostante "
        "una patologia genetica familiare, vive una vita serena. Viene seguito da un "
        "centro di riferimento specifico a Roma e si sottopone a controlli periodici. "
        "Nel febbraio del 2019 esegue un intervento chirurgico presso il Policlinico "
        "Gemelli. Al rientro a casa, nel maggio del 2019, deve proseguire i controlli "
        "cardiologici. Non essendo autonomo nella deambulazione, viene preso in carico "
        "dai servizi del Distretto Sanitario di Milano. Marco sembra una persona "
        "collaborante al piano assistenziale predisposto."
    )
    expected = [
        ("NOME", "Marco"),
        ("ETÀ", "58"),
        ("LUOGO/INDIRIZZO", "Roma"),
        ("DATA", "febbraio del 2019"),
        ("LUOGO/INDIRIZZO", "Policlinico Gemelli"),
        ("DATA", "maggio del 2019"),
        ("LUOGO/INDIRIZZO", "Milano"),
    ]
    return text, expected


def evaluate_redaction(text, expected_entities, redacted):
    """Compute TP/FP/FN per category using paper's metric."""
    by_type = {"NOME": [], "ETÀ": [], "DATA": [], "LUOGO/INDIRIZZO": []}
    for t, v in expected_entities:
        by_type[t].append(v)
    results = {}
    for t, vals in by_type.items():
        n_gold = len(vals)
        fn = sum(1 for v in vals if v in redacted)
        tp = n_gold - fn
        fp = max(0, redacted.count(f"[{t}]") - n_gold)
        results[t] = {"tp": tp, "fp": fp, "fn": fn,
                      "redacted_correctly": [v for v in vals if v not in redacted],
                      "missed": [v for v in vals if v in redacted]}
    return results


def main():
    print("Loading model...")
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

    print("Model loaded.\n")

    for label, (text, expected) in [
        ("IN-DISTRIBUTION (Italian names/cities)", make_normal_note()),
        ("OUT-OF-DISTRIBUTION (Kazakh/Uzbek names+cities, year 2034)", make_ood_note()),
    ]:
        print("=" * 80)
        print(f"TEST: {label}")
        print("=" * 80)
        print(f"\nInput note (first 300 chars):\n{text[:300]}...\n")
        print(f"Expected entities to redact:")
        for t, v in expected:
            print(f"  [{t}] {v}")

        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                                     repetition_penalty=1.0, eos_token_id=eos_ids,
                                     pad_token_id=tokenizer.eos_token_id)
        redacted = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\nModel redacted output (first 400 chars):\n{redacted[:400]}...\n")

        results = evaluate_redaction(text, expected, redacted)
        print("Per-category result:")
        tp_total, fp_total, fn_total = 0, 0, 0
        for t, r in results.items():
            tp_total += r['tp']; fp_total += r['fp']; fn_total += r['fn']
            if r['tp'] + r['fp'] + r['fn'] == 0:
                continue
            print(f"  {t:<22} TP={r['tp']} FP={r['fp']} FN={r['fn']}")
            if r['redacted_correctly']:
                print(f"    ✓ Redacted: {r['redacted_correctly']}")
            if r['missed']:
                print(f"    ✗ Missed:   {r['missed']}")
        total = tp_total + fp_total + fn_total
        p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        r_ = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        f1 = 2 * p * r_ / (p + r_) if (p + r_) > 0 else 0
        print(f"\n  TOTAL: TP={tp_total} FP={fp_total} FN={fn_total}  P={p:.3f} R={r_:.3f} F1={f1:.3f}\n")


if __name__ == "__main__":
    main()
