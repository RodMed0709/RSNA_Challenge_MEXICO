"""sub056_2 - L5v2: the 3 L5 readings (gpt-4.1-mini, cache reused) + 2 new prompts (D_strict, E_checklist) on NEW_MODEL, averaged.

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

MODEL = "gpt-4.1-mini"          # L5 readings A/B/C (unchanged, cached)
NEW_MODEL = os.environ.get("L5V2_MODEL", "gpt-5-mini")
NEW_EFFORT = os.environ.get("L5V2_EFFORT", "low")
PRICES = {  # USD per 1M tokens: in, cached in, out
    "gpt-4.1-mini": (0.40, 0.10, 1.60), "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-5-mini": (0.25, 0.025, 2.00), "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),  # exp011 (OpenAI pricing page 2026-10-08)
    "gpt-5.4-nano": (0.20, 0.02, 1.25),   # assumed list price; checked against billing below
}
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

SYS_D = """You are a fellowship-trained musculoskeletal radiologist grading a knee MRI report for a research dataset that uses the RSNA/Kaggle official label definitions STRICTLY:
- ACL and MCL count ONLY if the report describes a high-grade partial tear (>50% of fibers) or a complete tear/rupture. Sprain, low-grade or grade 1-2 injury, signal change, mucoid degeneration, thickening, or an intact post-surgical graft do NOT count.
- Effusion counts ONLY if moderate or large (or described as distending the joint). Small, minimal, trace or physiological fluid does NOT count.
- Baker's counts ONLY for a moderate or large popliteal/Baker's cyst. A small cyst does NOT count.
- Medial OA / Lateral OA / PF OA count ONLY with high-grade cartilage loss (>50% of thickness, grade 3-4, full-thickness, denuded bone, or advanced/severe osteoarthritis) in that compartment (medial tibiofemoral, lateral tibiofemoral, patellofemoral). Mild chondropathy, grade 1-2, superficial fissuring, or osteophytes alone do NOT count.
- Medial/Lateral Meniscus count ONLY for a tear (signal reaching an articular surface, displaced/flap/bucket-handle/radial/complex tear, truncation). Degeneration or intrasubstance signal does NOT count.
- Synovitis: synovial thickening/inflammation, hypertrophic or proliferative synovium.
- Contusion: bone bruise/contusion/impaction marrow edema without a fracture line.
- Fracture: acute fracture line or cortical break (including osteochondral or avulsion fracture).
The report may be in any language; read it in the original.
For each finding give the severity category the report states and your probability (0-100) that the strict definition is met.
Probabilities: strict criteria explicitly met 90-100; finding present but grade/size not stated 45-70; probable/suspected 40-60; possible/cannot exclude 15-35; explicitly below threshold 3-15; not mentioned 2-5; explicitly normal/absent 0-2."""

SYS_E = f"""You audit knee MRI reports (any language) with a fixed checklist. For EACH of the 12 findings answer four questions about the report text only:
1. mentioned: does the report say anything about this structure/finding? (true/false)
2. negated: does the report say it is absent, normal, intact or preserved? (true/false)
3. hedged: is the statement uncertain (possible, probable, suspected, likely, cannot be excluded, differential)? (true/false)
4. meets_definition: does the stated finding meet the official definition below? ("yes", "no", "unclear")
Official definitions:
{DEF_TEXT}
Rules: medial and lateral are distinct; do not infer one finding from another; if not mentioned: negated=false, hedged=false, meets_definition="no"."""

STATUS_P = {"present": 0.95, "present_low_grade": 0.25, "uncertain": 0.45, "absent": 0.01, "not_mentioned": 0.05}


def _obj(props, req=None):
    return {"type": "object", "properties": props, "required": req or list(props), "additionalProperties": False}


SCHEMA_A = _obj({k: {"type": "string", "enum": list(STATUS_P)} for k in KEYS})
_PROBS = _obj({k: {"type": "integer"} for k in KEYS})
SCHEMA_B = _obj({"notes": {"type": "string"}, "probabilities": _PROBS})
SCHEMA_C = _obj({"english_paraphrase": {"type": "string"}, "probabilities": _PROBS})

_SEV = {"type": "string", "enum": ["not_mentioned", "normal_or_absent", "below_threshold", "unspecified", "meets_threshold"]}
SCHEMA_D = _obj({k: _obj({"severity": _SEV, "probability": {"type": "integer"}}) for k in KEYS})
SCHEMA_E = _obj({k: _obj({"mentioned": {"type": "boolean"}, "negated": {"type": "boolean"}, "hedged": {"type": "boolean"},
                          "meets_definition": {"type": "string", "enum": ["yes", "no", "unclear"]}}) for k in KEYS})


def checklist_p(c):
    if not c["mentioned"]:
        return 0.05
    if c["negated"]:
        return 0.01
    m, h = c["meets_definition"], c["hedged"]
    if m == "yes":
        return 0.55 if h else 0.95
    if m == "unclear":
        return 0.40 if h else 0.60
    return 0.10 if h else 0.15


PROMPTS = {  # name: (system, schema, max output tokens, model)
    "A_extract": (SYS_A, SCHEMA_A, 400, MODEL),
    "B_reason": (SYS_B, SCHEMA_B, 500, MODEL),
    "C_paraphrase": (SYS_C, SCHEMA_C, 600, MODEL),
    "D_strict": (SYS_D, SCHEMA_D, 6000, NEW_MODEL),
    "E_checklist": (SYS_E, SCHEMA_E, 6000, NEW_MODEL),
}


def to_probs(name, obj):
    if name == "A_extract":
        return [STATUS_P[obj[k]] for k in KEYS]
    if name == "D_strict":
        return [min(max(float(obj[k]["probability"]), 0.0), 100.0) / 100.0 for k in KEYS]
    if name == "E_checklist":
        return [checklist_p(obj[k]) for k in KEYS]
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
    pi, pc, po = PRICES[recs[0].get("model_req", "gpt-4.1-mini")] if recs else PRICES["gpt-4.1-mini"]
    usd = ((ti - tc) * pi + tc * pc + to * po) / 1e6
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
    spent = [sum(cost(list(caches[q].values()))["usd"] for q in (args.only.split(",") if args.only else PROMPTS))]
    stop = threading.Event()
    run_p = args.only.split(",") if args.only else list(PROMPTS)
    jobs = [(p, r.StudyInstanceUID, r.Report) for p in run_p for r in df.itertuples()
            if r.StudyInstanceUID not in caches[p]]
    print(f"todo {len(jobs)} calls; already spent ${spent[0]:.4f}", flush=True)

    def one(p, uid, report):
        if stop.is_set():
            return
        sys_msg, schema, max_tok, model = PROMPTS[p]
        kw = dict(model=model, seed=54)
        if model.startswith("gpt-5"):
            kw.update(max_completion_tokens=max_tok, reasoning_effort=NEW_EFFORT)
        else:
            kw.update(temperature=0, max_tokens=max_tok)
        for attempt in range(3):
            resp = client.chat.completions.create(
                **kw,
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
        rec = {"uid": uid, "prompt": p, "model": resp.model, "model_req": model, "probs": probs, "raw": obj,
               "usage": {"in": u.prompt_tokens, "cached": cached, "out": u.completion_tokens}}
        pi, pc, po = PRICES[model]
        c = ((u.prompt_tokens - cached) * pi + cached * pc + u.completion_tokens * po) / 1e6
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
    ap.add_argument("--only", default="", help="comma list of prompt names to run (default all)")
    ap.add_argument("--build", default="")
    a = ap.parse_args()
    build(a) if a.build else run(a)
