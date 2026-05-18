"""
Parallel Gemini 2.5 Flash generator. Uses 3 API keys concurrently for ~3x speedup.
Reuses prompt + schema + validator from generate_synthetic_v2.
"""
import json, os, time, itertools, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import google.generativeai as genai

from generate_synthetic_v2 import (
    API_KEYS, MODEL_NAME, CASES_PER_REQUEST, RESPONSE_SCHEMA,
    SPECIALTIES, GENDERS, AGE_RANGES, DENSITIES,
    load_style_anchors, build_prompt, validate_case,
)


def call_gemini_once(prompt, key, retries=3):
    genai.configure(api_key=key)
    model = genai.GenerativeModel(MODEL_NAME)
    cfg = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
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
            time.sleep(2 * (attempt + 1))
    raise last_err


def run_one_batch(batch_idx, anchors, key):
    specialty = SPECIALTIES[batch_idx % len(SPECIALTIES)]
    gender = GENDERS[batch_idx % len(GENDERS)]
    age_range = AGE_RANGES[batch_idx % len(AGE_RANGES)]
    density = DENSITIES[batch_idx % len(DENSITIES)]

    prompt = build_prompt(anchors, CASES_PER_REQUEST, specialty, gender, age_range, density)
    try:
        raw = call_gemini_once(prompt, key)
    except Exception as e:
        return {"batch": batch_idx, "ok": [], "rejected": [{"reason": f"api_error: {e}"}]}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        debug_path = f"./data/_debug_raw_batch{batch_idx}.txt"
        with open(debug_path, 'w', encoding='utf-8') as df:
            df.write(raw)
        return {"batch": batch_idx, "ok": [], "rejected": [{"reason": f"json_error: {e}", "raw_path": debug_path}]}

    cases = parsed.get("cases", [])
    ok_list, bad_list = [], []
    for case in cases:
        valid, reason = validate_case(case)
        if valid:
            ok_list.append(case)
        else:
            bad_list.append({"reason": reason, "preview": str(case)[:200]})
    return {"batch": batch_idx, "ok": ok_list, "rejected": bad_list, "tags": f"{specialty}/{gender}/{age_range}/{density}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--output", default="./data/synthetic_v2_1000.json")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    anchors = load_style_anchors(3)
    print(f"Anchors: {[len(a['text']) for a in anchors]} chars")

    n_target = args.n
    n_batches = (n_target + CASES_PER_REQUEST - 1) // CASES_PER_REQUEST + 10  # 10 buffer for rejects
    print(f"Generating {n_target} cases via up to {n_batches} batches of {CASES_PER_REQUEST} (workers={args.workers})")

    all_valid, all_rejected = [], []
    futures = []
    key_cycle = itertools.cycle(API_KEYS)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for batch_idx in range(n_batches):
            key = next(key_cycle)
            futures.append(pool.submit(run_one_batch, batch_idx, anchors, key))

        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            res = fut.result()
            all_valid.extend(res["ok"])
            all_rejected.extend(res["rejected"])
            elapsed = time.time() - t0
            print(f"[{done_count}/{n_batches}] batch {res['batch']:>3} ({res.get('tags','?')}): "
                  f"+{len(res['ok'])} ok, -{len(res['rejected'])} rej | "
                  f"total valid={len(all_valid)} | {elapsed:.0f}s elapsed")
            if len(all_valid) >= n_target:
                # Cancel remaining
                for f in futures:
                    if not f.done():
                        f.cancel()
                break

    all_valid = all_valid[:n_target]
    print(f"\n{'='*60}\nFINAL: {len(all_valid)} valid / {len(all_rejected)} rejected\n{'='*60}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_valid, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_valid)} → {args.output}")

    if all_rejected:
        rej_path = args.output.replace(".json", "_rejected.json")
        with open(rej_path, 'w', encoding='utf-8') as f:
            json.dump(all_rejected, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(all_rejected)} rejections → {rej_path}")


if __name__ == "__main__":
    main()
