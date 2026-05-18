"""
Generate gold-style synthetic Italian clinical notes via Gemini 2.5 Flash.

Pipeline:
1. Load 3 gold notes as style anchors (few-shot)
2. Prompt Gemini in batches with diversity controls (specialty, gender, age, density)
3. Validate every output: JSON parses, entities present in text, redaction consistent
4. Drop bad cases, keep good ones in JSON array

Output schema (one record):
  {"text": str, "redacted_text": str,
   "entities": [{"text": str, "type": NOME|ETÀ|DATA|LUOGO/INDIRIZZO}, ...]}
"""
import json, os, random, time, itertools, argparse
from pathlib import Path
import google.generativeai as genai

# 3 keys for round-robin (user provided)
API_KEYS = [
    "AIzaSyBN8IBQmDZ2K_pS0H_uMNZRLYoF7O5Rtvs",
    "AIzaSyAroqhMxOEMTEg5clEcqLTWqOPFoR434B4",
    "AIzaSyBHCP6ZNPvIlyOew_16yyZv2EqsDA1rY-M",
]
KEY_CYCLE = itertools.cycle(API_KEYS)

GOLD_PATH = "./data/gold_standard_80.json"
OUTPUT_PATH = "./data/synthetic_v2_test20.json"
MODEL_NAME = "gemini-2.5-flash"
CASES_PER_REQUEST = 3

# JSON schema to force structured output (Gemini guarantees JSON-valid escaping)
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "cases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "redacted_text": {"type": "string"},
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "type": {
                                    "type": "string",
                                    "enum": ["NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"],
                                },
                            },
                            "required": ["text", "type"],
                        },
                    },
                },
                "required": ["text", "redacted_text", "entities"],
            },
        }
    },
    "required": ["cases"],
}

VALID_TYPES = {"NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"}

SPECIALTIES = [
    "oncologia", "cardiologia", "nefrologia", "neurologia",
    "ortopedia", "ginecologia", "pneumologia", "gastroenterologia",
    "endocrinologia", "geriatria",
]
GENDERS = ["donna", "uomo"]
AGE_RANGES = ["20-40 anni (giovane adulto)", "40-65 anni (adulto)", "65-90 anni (anziano)"]
DENSITIES = ["bassa (2-4 entità)", "media (5-8 entità)", "alta (9-12 entità)"]


def load_style_anchors(n=3):
    with open(GOLD_PATH, 'r', encoding='utf-8') as f:
        gold = json.load(f)
    # Pick diverse anchors: short, medium, long
    sorted_by_len = sorted(gold, key=lambda r: len(r.get("text", "")))
    anchors = [sorted_by_len[len(sorted_by_len) // 4],     # short-ish
               sorted_by_len[len(sorted_by_len) // 2],     # medium
               sorted_by_len[3 * len(sorted_by_len) // 4]] # long-ish
    return anchors[:n]


def build_prompt(anchors, n_cases, specialty, gender, age_range, density):
    examples_block = ""
    for i, a in enumerate(anchors, 1):
        ents = a.get("entities", [])
        ents_compact = json.dumps(
            [{"text": e.get("text", ""), "type": e.get("type", "")} for e in ents],
            ensure_ascii=False
        )
        examples_block += f"\n--- ESEMPIO {i} ---\nTESTO: {a.get('text', '')}\nENTITÀ: {ents_compact}\n"

    return f"""Sei un medico italiano esperto nella redazione di casi clinici. Genera {n_cases} nuovi casi clinici originali in italiano, nello stile dei seguenti esempi reali tratti dalla letteratura medica italiana.

ESEMPI DI STILE (USA QUESTO STESSO REGISTRO LINGUISTICO):
{examples_block}

PARAMETRI DI DIVERSIFICAZIONE PER QUESTO BATCH:
- Specialità medica: {specialty}
- Genere paziente: {gender}
- Fascia d'età: {age_range}
- Densità entità sensibili: {density}

REQUISITI PER CIASCUN CASO:
1. Lunghezza tra 1500 e 3000 caratteri
2. Stile narrativo: terza persona, tempi passati (presente storico ammesso). Densa terminologia medica italiana.
3. Includere entità sensibili realistiche e VARIE:
   - NOME: nome italiano realistico (es. "Mario Rossi", "Giulia Ferri") menzionato 3-6 volte nel testo (anaphora naturale)
   - ETÀ: nel testo l'età compare come "di 67 anni" o "67 anni", ma in `entities` il `text` deve essere SOLO IL NUMERO (es. "67"), SENZA la parola "anni". Età coerente con la patologia.
   - DATA: usa formati MISTI nel testo ("marzo 2018", "15/04/2020", "novembre 2019", "2017", "il 4 maggio 2021")
   - LUOGO/INDIRIZZO: città italiane reali + strutture sanitarie realistiche (es. "Policlinico Gemelli", "Ospedale San Raffaele", "Istituto Rizzoli", "ASUGI di Trieste")

FORMATO DI USCITA — JSON valido (un solo oggetto con un array di casi):
{{
  "cases": [
    {{
      "text": "...testo originale del caso clinico...",
      "redacted_text": "...stesso testo con tag [NOME], [ETÀ], [DATA], [LUOGO/INDIRIZZO] al posto delle entità sensibili...",
      "entities": [
        {{"text": "Mario Rossi", "type": "NOME"}},
        {{"text": "67", "type": "ETÀ"}},
        {{"text": "marzo 2018", "type": "DATA"}},
        {{"text": "Roma", "type": "LUOGO/INDIRIZZO"}}
      ]
    }}
  ]
}}

VINCOLI CRITICI (la mancata osservanza causerà lo scarto del caso):
- Ogni entità in `entities` DEVE apparire letteralmente nel `text`.
- Il `redacted_text` deve essere identico al `text` tranne per la sostituzione delle entità con i rispettivi tag.
- Nessuna entità deve sopravvivere in `redacted_text` (deve essere stata sostituita).
- I tipi devono essere ESATTAMENTE uno tra: NOME, ETÀ, DATA, LUOGO/INDIRIZZO."""


def sanitize_type(etype):
    etype = (etype or "").upper()
    if "LUOGO" in etype or "INDIRIZZO" in etype:
        return "LUOGO/INDIRIZZO"
    if etype in VALID_TYPES:
        return etype
    return None


def validate_case(case):
    """Return (valid: bool, reason: str). Drops cases that fail."""
    if not isinstance(case, dict):
        return False, "not a dict"
    text = case.get("text", "")
    redacted = case.get("redacted_text", "")
    ents = case.get("entities", [])
    if not text or not redacted or not isinstance(ents, list):
        return False, "missing fields"
    if not (1000 <= len(text) <= 4000):
        return False, f"length {len(text)} out of range"

    clean_ents = []
    for e in ents:
        if not isinstance(e, dict):
            continue
        etype = sanitize_type(e.get("type", ""))
        etext = (e.get("text", "") or "").strip()
        if not etype or not etext:
            continue
        if etext not in text:
            return False, f"entity {etext!r} not in text"
        # Redacted should not contain the entity text any more
        if etext in redacted:
            return False, f"entity {etext!r} still in redacted_text"
        clean_ents.append({"text": etext, "type": etype})

    if len(clean_ents) < 2:
        return False, "fewer than 2 valid entities"

    # Redacted must contain at least one tag of each entity's type present
    present_types = {e["type"] for e in clean_ents}
    for t in present_types:
        if f"[{t}]" not in redacted:
            return False, f"no [{t}] tag in redacted_text"

    case["entities"] = clean_ents
    return True, "ok"


def call_gemini(prompt, key, retries=3):
    genai.configure(api_key=key)
    model = genai.GenerativeModel(MODEL_NAME)
    cfg = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,   # force schema-compliant output
        temperature=0.7,
        max_output_tokens=12288,
    )
    last_err = None
    for attempt in range(retries):
        try:
            resp = model.generate_content(prompt, generation_config=cfg)
            return resp.text
        except Exception as e:
            last_err = e
            print(f"  [retry {attempt+1}] {type(e).__name__}: {str(e)[:120]}")
            time.sleep(2 * (attempt + 1))
    raise last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="number of cases to generate")
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    print(f"Loading style anchors from gold...")
    anchors = load_style_anchors(3)
    for i, a in enumerate(anchors):
        print(f"  anchor {i+1}: {len(a.get('text', ''))} chars, {len(a.get('entities', []))} entities")

    n_target = args.n
    n_batches = (n_target + CASES_PER_REQUEST - 1) // CASES_PER_REQUEST

    print(f"\nGenerating {n_target} cases via {n_batches} batches of ~{CASES_PER_REQUEST}...")

    all_valid = []
    all_rejected = []

    for batch_idx in range(n_batches):
        # Rotate diversity params each batch
        specialty = SPECIALTIES[batch_idx % len(SPECIALTIES)]
        gender = GENDERS[batch_idx % len(GENDERS)]
        age_range = AGE_RANGES[batch_idx % len(AGE_RANGES)]
        density = DENSITIES[batch_idx % len(DENSITIES)]
        key = next(KEY_CYCLE)

        print(f"\nBatch {batch_idx+1}/{n_batches}: {specialty}, {gender}, {age_range}, density={density}")
        prompt = build_prompt(anchors, CASES_PER_REQUEST, specialty, gender, age_range, density)

        try:
            raw = call_gemini(prompt, key)
        except Exception as e:
            print(f"  ERROR after retries: {e}")
            continue

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}; saving full raw for inspection")
            # Save full raw output to a debug file
            debug_path = f"./data/_debug_raw_batch{batch_idx}.txt"
            with open(debug_path, 'w', encoding='utf-8') as df:
                df.write(raw)
            all_rejected.append({"batch": batch_idx, "error": str(e), "raw_path": debug_path})
            continue

        cases = parsed.get("cases", [])
        print(f"  received {len(cases)} cases")

        for case in cases:
            ok, reason = validate_case(case)
            if ok:
                all_valid.append(case)
                print(f"    [+] OK: {len(case['text'])} chars, {len(case['entities'])} entities")
            else:
                all_rejected.append({"reason": reason, "preview": str(case)[:200]})
                print(f"    [-] REJECT: {reason}")

        if len(all_valid) >= n_target:
            break

    all_valid = all_valid[:n_target]
    print(f"\n{'='*60}")
    print(f"FINAL: {len(all_valid)} valid / {len(all_rejected)} rejected")
    print(f"{'='*60}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_valid, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_valid)} cases to {args.output}")

    if all_rejected:
        rej_path = args.output.replace(".json", "_rejected.json")
        with open(rej_path, 'w', encoding='utf-8') as f:
            json.dump(all_rejected, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(all_rejected)} rejected to {rej_path}")


if __name__ == "__main__":
    main()
