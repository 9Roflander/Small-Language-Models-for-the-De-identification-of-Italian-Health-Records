"""
Generate NO-NAME synthetic clinical notes via Gemini 2.5 Flash.
Goal: balance the synth_v2 dataset which currently has 4x more names than gold.

These notes contain ONLY: ETÀ, DATA, LUOGO/INDIRIZZO. No patient names.
Patients are referred to anonymously: 'la paziente', 'l'uomo', 'il soggetto', 'la donna', etc.
"""
import json, os, time, itertools, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import google.generativeai as genai

from generate_synthetic_v2 import (
    API_KEYS, MODEL_NAME, CASES_PER_REQUEST, RESPONSE_SCHEMA,
    SPECIALTIES, AGE_RANGES, DENSITIES,
    load_style_anchors, validate_case,
)


def build_noname_prompt(anchors, n_cases, specialty, age_range, density):
    """Prompt explicitly forbids patient names."""
    examples_block = ""
    for i, a in enumerate(anchors, 1):
        ents = a.get("entities", [])
        # Filter out NOME from examples shown to model (so it doesn't copy)
        ents_compact = json.dumps(
            [{"text": e.get("text", ""), "type": e.get("type", "")}
             for e in ents if e.get("type") != "NOME"],
            ensure_ascii=False
        )
        examples_block += f"\n--- ESEMPIO {i} ---\nTESTO: {a.get('text', '')}\nENTITÀ (nessun nome): {ents_compact}\n"

    return f"""Sei un medico italiano esperto. Genera {n_cases} casi clinici originali in italiano IN CUI IL PAZIENTE NON VIENE MAI MENZIONATO PER NOME. Lo stile deve seguire questi esempi reali (ignora i nomi che eventualmente compaiono):

{examples_block}

PARAMETRI:
- Specialità medica: {specialty}
- Fascia d'età: {age_range}
- Densità entità: {density}

REGOLE ASSOLUTE:
1. Il paziente NON deve mai avere un nome proprio. Riferirsi sempre con descrizioni anonime: "il paziente", "la paziente", "l'uomo", "la donna", "il soggetto", "una signora di X anni", "un anziano", "il bambino", "l'adolescente". MAI nomi propri come Mario, Anna, Giulia, Marco, ecc.
2. NESSUNA entità di tipo NOME deve apparire in `entities`. L'array contiene SOLO entità di tipo ETÀ, DATA, LUOGO/INDIRIZZO.
3. Lunghezza testo: 1500-3000 caratteri
4. Stile narrativo: terza persona, tempo passato, denso di terminologia medica italiana
5. Entità da includere (con valori realistici italiani):
   - ETÀ: nel testo "di 67 anni" o "67 anni", ma in `entities` SOLO IL NUMERO (es. "67")
   - DATA: formati misti ("marzo 2018", "15/04/2020", "novembre 2019", "2017")
   - LUOGO/INDIRIZZO: città italiane + strutture sanitarie (es. "Policlinico Gemelli", "Ospedale Niguarda di Milano")

FORMATO USCITA (JSON valido, schema fissato):
{{
  "cases": [
    {{
      "text": "...caso clinico SENZA nome del paziente...",
      "redacted_text": "...stesso testo con tag [ETÀ], [DATA], [LUOGO/INDIRIZZO]...",
      "entities": [
        {{"text": "67", "type": "ETÀ"}},
        {{"text": "marzo 2018", "type": "DATA"}},
        {{"text": "Roma", "type": "LUOGO/INDIRIZZO"}}
      ]
    }}
  ]
}}

VINCOLI:
- Ogni entità DEVE apparire nel `text`
- Il `redacted_text` deve avere TUTTE le entità sostituite con i tag
- NESSUN nome proprio del paziente nel testo (verifica scrupolosamente)
- Solo tipi: ETÀ, DATA, LUOGO/INDIRIZZO (mai NOME)"""


def call_gemini_once(prompt, key, retries=3):
    genai.configure(api_key=key)
    model = genai.GenerativeModel(MODEL_NAME)
    cfg = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        temperature=0.7,
        max_output_tokens=12288,
    )
    for attempt in range(retries):
        try:
            return model.generate_content(prompt, generation_config=cfg).text
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def validate_noname_case(case):
    """Same as validate_case but also rejects any NOME entity."""
    valid, reason = validate_case(case)
    if not valid:
        return False, reason
    for e in case.get("entities", []):
        if e.get("type") == "NOME":
            return False, "contains NOME entity (should be name-free)"
    return True, "ok"


def run_one_batch(batch_idx, anchors, key):
    specialty = SPECIALTIES[batch_idx % len(SPECIALTIES)]
    age_range = AGE_RANGES[batch_idx % len(AGE_RANGES)]
    density = DENSITIES[batch_idx % len(DENSITIES)]
    prompt = build_noname_prompt(anchors, CASES_PER_REQUEST, specialty, age_range, density)
    try:
        raw = call_gemini_once(prompt, key)
    except Exception as e:
        return {"batch": batch_idx, "ok": [], "rejected": [{"reason": f"api: {e}"}]}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"batch": batch_idx, "ok": [], "rejected": [{"reason": f"json: {e}"}]}
    ok, bad = [], []
    for case in parsed.get("cases", []):
        valid, reason = validate_noname_case(case)
        if valid:
            ok.append(case)
        else:
            bad.append({"reason": reason})
    return {"batch": batch_idx, "ok": ok, "rejected": bad, "tags": f"{specialty}/{age_range}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--output", default="./data/synthetic_v2_noname_400.json")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    anchors = load_style_anchors(3)
    n_batches = (args.n + CASES_PER_REQUEST - 1) // CASES_PER_REQUEST + 30  # bigger buffer for noname rejections
    print(f"Target {args.n} no-name cases via up to {n_batches} batches (workers={args.workers})")

    all_valid, all_rejected = [], []
    key_cycle = itertools.cycle(API_KEYS)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one_batch, i, anchors, next(key_cycle)) for i in range(n_batches)]
        done = 0
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            all_valid.extend(res["ok"])
            all_rejected.extend(res["rejected"])
            elapsed = time.time() - t0
            print(f"[{done}/{n_batches}] batch {res['batch']:>3} ({res.get('tags','?')}): "
                  f"+{len(res['ok'])} ok, -{len(res['rejected'])} rej | "
                  f"total valid={len(all_valid)} | {elapsed:.0f}s")
            if len(all_valid) >= args.n:
                for f in futures:
                    if not f.done(): f.cancel()
                break

    all_valid = all_valid[:args.n]
    print(f"\nFINAL: {len(all_valid)} valid / {len(all_rejected)} rejected")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_valid, f, ensure_ascii=False, indent=2)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
