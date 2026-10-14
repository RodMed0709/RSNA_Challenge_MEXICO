"""exp054 - L5: soft report labels from OpenAI with k=3 prompt variants, averaged.

Usage:
  python generar_L5.py --data DIR --cache DIR [--n 20] [--workers 12] [--budget 14]
  python generar_L5.py --data DIR --cache DIR --build OUT.csv

Reads OPENAI_API_KEY from the environment or from RSNA_Challenge/.secrets.env (never printed).
Cache: one jsonl per prompt variant; reruns skip studies already done (resumable).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

MODEL = "gpt-4.1-mini"
PRICE_IN, PRICE_CACHED, PRICE_OUT = 0.40, 0.10, 1.60  # USD per 1M tokens (gpt-4.1-mini)
SECRETS = Path(r"C:/Users/medro/Documents/_PERSONAL_PROJECTS/RSNA_Challenge/.secrets.env")

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
KEYS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa", "pf_oa",
        "effusion", "synovitis", "bakers", "contusion", "fracture"]

# Official competition definitions (Kaggle discussion 733343), same content as the exp010/exp011 glossary.
DEFS = {
    "acl": "ACL: high-grade partial or complete tear of the anterior cruciate ligament (fiber discontinuity, >50% of fibers torn). Mild signal change, mucoid degeneration or thickening without discontinuity is negative.",
    "mcl": "MCL: acute high-grade partial or complete tear of the medial collateral ligament, with disrupted fibers and edema in/around the ligament. Mild sprains or chronic changes are negative.",
    "medial_meniscus": "Medial Meniscus: tear of the medial meniscus (abnormal signal definitely reaching the articular surface, or a truncated, small or displaced fragment). Intrasubstance degeneration not reaching the surface is negative.",
    "lateral_meniscus": "Lateral Meniscus: same tear criteria applied to the lateral meniscus.",
    "medial_oa": "Medial OA: moderate or large area (about 1 cm or more) of high-grade (>50% thickness) cartilage loss in the medial tibiofemoral compartment, with or without subchondral changes.",
    "lateral_oa": "Lateral OA: same osteoarthritis criteria applied to the lateral tibiofemoral compartment.",
    "pf_oa": "PF OA: same osteoarthritis criteria applied to the patellofemoral compartment (patella/trochlea).",
    "effusion": "Effusion: moderate or large joint fluid distending the joint.",
    "synovitis": "Synovitis: inflammation and thickening of the synovial lining of the joint.",
    "bakers": "Baker's: moderate or large fluid collection in the typical posterior location of a Baker's (popliteal) cyst.",
    "contusion": "Contusion: bone contusion / bone bruise, impaction-type bone marrow edema without a defined fracture line.",
    "fracture": "Fracture: acute fracture with cortical disruption or a fracture line.",
}
DEF_TEXT = "\n".join(f"- {DEFS[k]}" for k in KEYS)

SYS_A = f"""You are an expert musculoskeletal radiologist extracting structured findings from a knee MRI report.
The report may be written in any language (English, Spanish, German, French, Dutch, Greek, Bulgarian, Russian, Turkish, Croatian, Polish...). Read it in its original language.
Use only what the report states. Do not infer a finding from symptoms or from another finding. Medial and lateral are distinct.

For each finding choose one status:
- present: the report clearly states the finding and it meets the definition below.
- present_low_grade: the report states the finding but explicitly below the definition threshold (e.g. small/minimal effusion, low-grade sprain, mild/early chondral thinning, small Baker's cyst, grade 1-2 chondropathy).
- uncertain: possible, probable, suspected, equivocal, cannot be excluded.
- absent: the report explicitly denies it or describes the structure as normal/intact/preserved.
- not_mentioned: the report says nothing about it.

Definitions:
{DEF_TEXT}
"""

SYS_B = f"""You are a senior knee radiologist. You will read one knee MRI report (any language) and estimate, for each of twelve findings, the probability (integer 0-100) that an expert applying the official definitions below would label the finding as PRESENT for this knee.

Definitions:
{DEF_TEXT}

Procedure: first write brief notes (at most 60 words, in English) quoting or translating the key positive and negative statements of the report. Then give the twelve probabilities.
Calibration guide: clearly stated and meeting the definition 90-100; stated but severity unclear 60-85; probable/suspected 50-70; possible/cannot be excluded 25-45; stated but explicitly below the threshold (small, mild, low-grade, early) 10-35; not mentioned 3-8; explicitly normal/intact/denied 0-2.
Use only the report. Do not infer one finding from another. Medial and lateral are distinct."""

SYS_C = f"""Task: knee MRI report -> 12 finding probabilities.
Step 1: paraphrase the report in plain English (at most 80 words), keeping every statement about ligaments, menisci, cartilage per compartment, joint fluid, synovium, popliteal cyst, bone marrow edema and fractures, including negations, uncertainty and severity words. The source may be in any language; translate faithfully.
Step 2: using ONLY your paraphrase, give for each finding an integer 0-100 = probability that the finding is present according to these definitions:
{DEF_TEXT}
Meaning of the numbers: 95 = definitely present and meets definition; 60 = probable or severity unspecified; 35 = possible / cannot exclude; 20 = present but explicitly mild/small/low-grade (below definition); 5 = not mentioned; 1 = explicitly normal or absent."""

STATUS_P = {"present": 0.95, "present_low_grade": 0.25, "uncertain": 0.45, "absent": 0.01, "not_mentioned": 0.05}


def _obj(props, req=None):
    return {"type": "object", "properties": props, "required": req or list(props), "additionalProperties": False}


SCHEMA_A = _obj({k: {"type": "string", "enum": list(STATUS_P)} for k in KEYS})
_PROBS = _obj({k: {"type": "integer"} for k in KEYS})
SCHEMA_B = _obj({"notes": {"type": "string"}, "probabilities": _PROBS})
SCHEMA_C = _obj({"english_paraphrase": {"type": "string"}, "probabilities": _PROBS})

PROMPTS = {
    "A_extract": (SYS_A, SCHEMA_A, 400),
    "B_reason": (SYS_B, SCHEMA_B, 500),
    "C_paraphrase": (SYS_C, SCHEMA_C, 600),
}


def to_probs(name, obj):
    if name == "A_extract":
        return [STATUS_P[obj[k]] for k in KEYS]
    p = obj["probabilities"]
    return [min(max(float(p[k]), 0.0), 100.0) / 100.0 for k in KEYS]


def api_key():
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OPENAI_API_KEY not found")


def load_cache(path):
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["uid"]] = d
    return out


def cost(recs):
    ti = sum(r["usage"]["in"] for r in recs)
    tc = sum(r["usage"]["cached"] for r in recs)
    to = sum(r["usage"]["out"] for r in recs)
    usd = ((ti - tc) * PRICE_IN + tc * PRICE_CACHED + to * PRICE_OUT) / 1e6
    return {"n": len(recs), "in": ti, "cached": tc, "out": to, "usd": round(usd, 4)}


def run(args):
    from openai import OpenAI
    client = OpenAI(api_key=api_key(), max_retries=6, timeout=120)
    df = pd.read_csv(Path(args.data) / "train.csv", encoding="utf-8")
    if args.n:
        df = df.sample(n=args.n, random_state=54) if args.sample else df.head(args.n)
    cache_dir = Path(args.cache); cache_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    caches = {p: load_cache(cache_dir / f"{p}.jsonl") for p in PROMPTS}
    spent = [sum(cost(list(c.values()))["usd"] for c in caches.values())]
    stop = threading.Event()
    jobs = [(p, r.StudyInstanceUID, r.Report) for p in PROMPTS for r in df.itertuples()
            if r.StudyInstanceUID not in caches[p]]
    print(f"todo {len(jobs)} calls; already spent ${spent[0]:.4f}", flush=True)

    def one(p, uid, report):
        if stop.is_set():
            return
        sys_msg, schema, max_tok = PROMPTS[p]
        for attempt in range(3):
            resp = client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=max_tok, seed=54,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": "REPORT:\n" + str(report).strip()[:12000]}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": p, "schema": schema, "strict": True}},
            )
            txt = resp.choices[0].message.content
            try:
                obj = json.loads(txt)
                probs = to_probs(p, obj)
                break
            except Exception as e:  # truncated or malformed output -> retry
                if attempt == 2:
                    print(f"FAIL {p} {uid}: {e}", flush=True)
                    return
        u = resp.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        rec = {"uid": uid, "prompt": p, "model": resp.model, "probs": probs, "raw": obj,
               "usage": {"in": u.prompt_tokens, "cached": cached, "out": u.completion_tokens}}
        c = ((u.prompt_tokens - cached) * PRICE_IN + cached * PRICE_CACHED + u.completion_tokens * PRICE_OUT) / 1e6
        with lock:
            with (cache_dir / f"{p}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            spent[0] += c
            if spent[0] > args.budget:
                stop.set()

    t0 = time.time(); done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(one, *j) for j in jobs]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print("ERR", repr(e)[:300], flush=True)
            done += 1
            if done % 200 == 0:
                print(f"{done}/{len(jobs)} ${spent[0]:.3f} {time.time()-t0:.0f}s", flush=True)
    if stop.is_set():
        print(f"BUDGET STOP at ${spent[0]:.3f}", flush=True)
    caches = {p: load_cache(cache_dir / f"{p}.jsonl") for p in PROMPTS}
    rep = {p: cost(list(c.values())) for p, c in caches.items()}
    rep["total_usd"] = round(sum(v["usd"] for v in rep.values()), 4)
    print(json.dumps(rep, indent=1))


def build(args):
    df = pd.read_csv(Path(args.data) / "train.csv", encoding="utf-8")
    cache_dir = Path(args.cache)
    per = {}
    for p in PROMPTS:
        c = load_cache(cache_dir / f"{p}.jsonl")
        per[p] = pd.DataFrame({u: r["probs"] for u, r in c.items()}, index=LABELS).T
    uids = df.StudyInstanceUID
    stack = [per[p].reindex(uids) for p in PROMPTS]
    mean = sum(s.fillna(0) for s in stack) / sum(s.notna().astype(float) for s in stack)
    mean.index.name = "StudyInstanceUID"
    out = mean.reset_index()
    out.to_csv(args.build, index=False)
    for p, s in zip(PROMPTS, stack):
        s.index.name = "StudyInstanceUID"
        s.reset_index().to_csv(Path(args.build).with_name(Path(args.build).stem + f"__{p}.csv"), index=False)
    print("written", args.build, out.shape, "complete rows:", int(mean.notna().all(axis=1).sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--budget", type=float, default=14.0)
    ap.add_argument("--build", default="")
    a = ap.parse_args()
    build(a) if a.build else run(a)
