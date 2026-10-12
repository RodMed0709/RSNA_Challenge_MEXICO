"""sub056_3 - L5v3 = the 5 L5v2 readings (cached A-E) + 2 new readings, averaged per cell.

New readings:
  F_focus3  gpt-5.4 (reasoning none): only Contusion, Synovitis, Baker's (the noisiest findings), focused definitions.
  G_quote   gpt-5.4-mini (reasoning none): all 12, shortest evidence quote per finding + probability 0-100.
F covers 3 columns (NaN elsewhere): those 3 cells average 7 readings, the other 9 average 6.

No openai package needed (plain HTTPS via requests). Key: /root/.openai_key (never printed, never on /workspace).
Exit codes: 0 ok, 3 insufficient quota (writes L5V3_NO_QUOTA), 4 projected/actual cost over budget (L5V3_OVER_BUDGET).

  python generar_L5v3.py --root DIR pilot   [--dry-run]   # 20 reports, projects full cost
  python generar_L5v3.py --root DIR run     [--dry-run]   # all reports, resumable, budget guard
  python generar_L5v3.py --root DIR build OUT.csv
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

KEY_FILE = Path("/root/.openai_key")
TRAIN_CSV = Path("/workspace/data/raw/train.csv")
URL = "https://api.openai.com/v1/chat/completions"
PRICES = {  # USD per 1M tokens (in, cached in, out)
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),   # exp011 (OpenAI pricing page 2026-10-08)
    "gpt-5.4-nano": (0.20, 0.02, 1.25),    # assumed list price
    "gpt-5.4": (2.50, 0.25, 15.00),        # assumed list price
}
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
KEYS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa", "pf_oa",
        "effusion", "synovitis", "bakers", "contusion", "fracture"]
OLD = ["A_extract", "B_reason", "C_paraphrase", "D_strict", "E_checklist"]  # cached, read-only

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

FOCUS = ["synovitis", "bakers", "contusion"]
SYS_F = """You are a senior musculoskeletal radiologist. Read one knee MRI report (any language: read it in the original) and judge ONLY three findings. Think about the exact wording; these three are often described indirectly.
- synovitis: synovial thickening, synovial hypertrophy/proliferation, inflamed or enhancing synovium, Hoffa (infrapatellar fat pad) synovitis, villonodular or hemorrhagic synovitis. Joint effusion ALONE is NOT synovitis. Plicae alone are NOT synovitis.
- bakers: popliteal / Baker's / semimembranosus-gastrocnemius bursal cyst, moderate or large. Small or minimal cyst counts only weakly. Ruptured Baker's cyst counts. Parameniscal or ganglion cysts are NOT Baker's.
- contusion: bone bruise, bone contusion, trabecular microfracture, impaction or kissing-contusion pattern, post-traumatic bone marrow edema WITHOUT a fracture line. Degenerative subchondral edema next to OA, bone infarct, and edema around a fracture line are NOT contusion.
Return for each an integer probability 0-100 that the finding is present by these definitions.
Guide: explicitly present 90-100; probable 60-80; possible / cannot exclude 25-45; present but explicitly small/minimal (bakers) 20-35; not mentioned 2-5; explicitly absent 0-2."""

SYS_G = f"""You are an expert musculoskeletal radiologist extracting labels from a knee MRI report (any language; read it in the original).
For each of the 12 findings: first copy the SHORTEST verbatim phrase from the report that decides it (empty string if the report says nothing about it), then give an integer probability 0-100 that the finding is present by the official definitions below.
Definitions:
{DEF_TEXT}
Probabilities: explicit and meets the definition 90-100; present but grade/size unstated 50-75; probable/suspected 45-65; possible/cannot exclude 20-40; explicitly below threshold (mild, small, low-grade) 5-20; not mentioned 2-5; explicitly normal/intact/absent 0-2.
Medial and lateral are distinct. Do not infer one finding from another."""


def _obj(props):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


SCHEMA_F = _obj({k: {"type": "integer"} for k in FOCUS})
SCHEMA_G = _obj({k: _obj({"evidence": {"type": "string"}, "probability": {"type": "integer"}}) for k in KEYS})
NEW = {  # name: (system, schema, max_completion_tokens, model)
    "F_focus3": (SYS_F, SCHEMA_F, 2000, "gpt-5.4"),
    "G_quote": (SYS_G, SCHEMA_G, 3000, "gpt-5.4-mini"),
}
# per-report token estimates for the dry run (from the gpt-5.4-mini D_strict pilot: 1316 in / 192 out per report)
EST_TOKENS = {"F_focus3": (900, 80), "G_quote": (1400, 420)}


def to_probs(name, obj):
    if name == "F_focus3":
        return [min(max(float(obj[k]), 0.0), 100.0) / 100.0 if k in FOCUS else None for k in KEYS]
    return [min(max(float(obj[k]["probability"]), 0.0), 100.0) / 100.0 for k in KEYS]


def load_cache(path):
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["uid"]] = d
    return out


def usd(model, tin, tcached, tout):
    pi, pc, po = PRICES[model]
    return ((tin - tcached) * pi + tcached * pc + tout * po) / 1e6


class NoQuota(Exception):
    pass


def call(session, key, name, report):
    sys_msg, schema, max_tok, model = NEW[name]
    body = {"model": model, "seed": 54, "max_completion_tokens": max_tok, "reasoning_effort": "none",
            "messages": [{"role": "system", "content": sys_msg},
                         {"role": "user", "content": "REPORT:\n" + str(report).strip()[:12000]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}}
    for attempt in range(8):
        r = session.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=180)
        if r.status_code == 200:
            d = r.json()
            obj = json.loads(d["choices"][0]["message"]["content"])
            u = d["usage"]
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            return obj, d.get("model", model), {"in": u["prompt_tokens"], "cached": cached, "out": u["completion_tokens"]}
        try:
            err = r.json().get("error", {}) or {}
        except Exception:
            err = {}
        if err.get("code") == "insufficient_quota" or err.get("type") == "insufficient_quota":
            raise NoQuota(err.get("message", "insufficient_quota"))
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(60, 2 ** attempt + random.random()))
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {str(err)[:300]}")
    raise RuntimeError("too many retries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("cmd", choices=["pilot", "run", "build"])
    ap.add_argument("out", nargs="?")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--budget", type=float, default=29.0, help="USD cap for F+G (whole run, incl. pilot)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", default="", help="comma list of new readings to run (default all)")
    a = ap.parse_args()
    root = Path(a.root); cache = root / "cache"
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    if a.cmd == "build":
        return build(df, cache, Path(a.out))
    if a.cmd == "pilot":
        df = df.sample(n=20, random_state=54)
    caches = {n: load_cache(cache / f"{n}.jsonl") for n in NEW}
    spent0 = sum(usd(r.get("model_req", NEW[n][3]), r["usage"]["in"], r["usage"]["cached"], r["usage"]["out"])
                 for n in NEW for r in caches[n].values())
    run_n = a.only.split(",") if a.only else list(NEW)
    jobs = [(n, r.StudyInstanceUID, r.Report) for n in run_n for r in df.itertuples() if r.StudyInstanceUID not in caches[n]]
    old_ok = {n: len(load_cache(cache / f"{n}.jsonl")) for n in OLD}
    print(f"[{a.cmd}] cached old readings: {old_ok}; new done: { {n: len(c) for n, c in caches.items()} }; "
          f"todo {len(jobs)} calls; spent so far ${spent0:.3f}; budget ${a.budget}", flush=True)
    if a.dry_run:
        full = sum(usd(NEW[n][3], *[EST_TOKENS[n][0], 0, EST_TOKENS[n][1]]) for n in NEW) * 4407
        print(f"[dry-run] key file present: {KEY_FILE.is_file()} · estimated full cost F+G ${full:.2f} "
              f"(F ${usd('gpt-5.4', 900, 0, 80) * 4407:.2f}, G ${usd('gpt-5.4-mini', 1400, 0, 420) * 4407:.2f}) · no API calls made")
        return 0
    import requests
    key = KEY_FILE.read_text().strip()
    session = requests.Session()
    lock = threading.Lock(); spent = [spent0]; stop = threading.Event(); noquota = threading.Event()

    def one(n, uid, rep):
        if stop.is_set():
            return
        try:
            obj, model, use = call(session, key, n, rep)
        except NoQuota:
            noquota.set(); stop.set(); return
        rec = {"uid": uid, "prompt": n, "model": model, "model_req": NEW[n][3], "probs": to_probs(n, obj), "raw": obj, "usage": use}
        with lock:
            with (cache / f"{n}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            spent[0] += usd(NEW[n][3], use["in"], use["cached"], use["out"])
            if spent[0] > a.budget:
                stop.set()

    t0 = time.time(); done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, *j) for j in jobs]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print("ERR", repr(e)[:300], flush=True)
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(jobs)} ${spent[0]:.2f} {time.time() - t0:.0f}s", flush=True)
    if noquota.is_set():
        (root / "L5V3_NO_QUOTA").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        print("INSUFFICIENT QUOTA -> L5V3_NO_QUOTA", flush=True); return 3
    caches = {n: load_cache(cache / f"{n}.jsonl") for n in NEW}
    per = {}
    for n, c in caches.items():
        ti = sum(r["usage"]["in"] for r in c.values()); to = sum(r["usage"]["out"] for r in c.values())
        tc = sum(r["usage"]["cached"] for r in c.values())
        per[n] = {"n": len(c), "in": ti, "cached": tc, "out": to, "usd": round(usd(NEW[n][3], ti, tc, to), 4)}
    total = sum(v["usd"] for v in per.values())
    print(json.dumps({"per_prompt": per, "total_usd": round(total, 4)}, indent=1), flush=True)
    if a.cmd == "pilot":
        projs = {n: round(v["usd"] / max(v["n"], 1) * 4407, 2) for n, v in per.items()}
        proj = sum(projs.values())
        print(f"PILOT projection for 4407 reports: ${proj:.2f} {projs} (budget ${a.budget})", flush=True)
        (root / "pilot_projection.json").write_text(json.dumps({"total": proj, **projs}))
        if proj > a.budget:
            (root / "L5V3_OVER_BUDGET").write_text(f"projected {proj:.2f} > {a.budget}\n"); return 4
        return 0
    if stop.is_set():
        (root / "L5V3_OVER_BUDGET").write_text(f"spent {spent[0]:.2f} > {a.budget}\n")
        print("BUDGET STOP", flush=True); return 4
    return 0


def build(df, cache, out):
    uids = df.StudyInstanceUID.astype(str)
    stacks = {}
    for n in OLD + list(NEW):
        c = load_cache(cache / f"{n}.jsonl")
        if len(set(c) & set(uids)) < 0.99 * len(uids):
            print(f"build: reading {n} skipped (covers {len(c)} studies)")
            continue
        m = pd.DataFrame({u: [math.nan if v is None else v for v in r["probs"]] for u, r in c.items()}, index=LABELS).T
        stacks[n] = m.reindex(uids)
    num = sum(s.fillna(0) for s in stacks.values()); den = sum(s.notna().astype(float) for s in stacks.values())
    mean = num / den
    mean.index.name = "StudyInstanceUID"
    mean.reset_index().to_csv(out, index=False)
    for n, s in stacks.items():
        s.index.name = "StudyInstanceUID"
        s.reset_index().to_csv(out.with_name(out.stem + f"__{n}.csv"), index=False)
    complete = int(mean.notna().all(axis=1).sum())
    print(f"written {out} {mean.shape} complete rows: {complete}; readings per cell min {int(den.min().min())} max {int(den.max().max())}")
    return 0 if complete >= 0.99 * len(df) else 5


if __name__ == "__main__":
    sys.exit(main())
