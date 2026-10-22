"""exp086 - L6: soft report labels from Claude Sonnet 5.5 (k=1, L5 prompt B_reason) + Claude Opus 5.5
adjudication of L5-vs-L6 cell disagreements -> consenso_v6 (training target; gold-58 = hard gold).

CPU + API only (no training). Commands:
  python generar_L6.py pilot   --out DIR [--n 50]       sync pilot: L6 on n reports + adjudication, cost projection
  python generar_L6.py l6      --out DIR                L6 on the remaining reports (Batch API; sync fallback)
  python generar_L6.py adjud   --out DIR                Opus adjudication of the disagreeing cells (Batch; sync fallback)
  python generar_L6.py build   --out DIR                L6 CSV, consenso_v6.csv, metrics (AUC vs gold, informative), READY.json
  python generar_L6.py all     --out DIR                pilot -> l6 -> adjud -> build

Secrets: ANTHROPIC_API_KEY and ANTHROPIC_WORKSPACE_ID from the environment or from --secrets (KEY=VALUE file).
They are never printed. Every request carries the `anthropic-workspace-id` header.
Caches are jsonl (one line per study), resumable; reruns skip what is done.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Amendment 18-oct ~20:00 UTC (Rodrigo): L6 = Opus 5.5 effort high (Sonnet 5.5 reading kept as informative L6s),
# adjudication = Opus 5.5 effort max, budget US$100 total.
L6_MODEL = "claude-opus-5-5"
L6S_MODEL = "claude-sonnet-5-5"
ADJ_MODEL = "claude-opus-5-5"
PRICE = {L6_MODEL: (4.00, 20.00), L6S_MODEL: (2.00, 10.00)}   # USD / 1M tokens (in, out); Batch API = 50 %
CACHE_READ_FRAC, CACHE_WRITE_FRAC = 0.10, 1.25
BUDGET_USD = 100.0         # hard cap for the whole experiment (pilots and the archived v1 included)
BATCH_MAX_WAIT_S = 2 * 3600  # > 2 h -> cancel and fall back to concurrent sync calls
AGREE_DELTA = 0.30         # SPEC: agreement = same side of 0.5 and |L5 - L6| <= 0.30
SHARP_POS, SHARP_NEG = 0.95, 0.05
SEED = 86

DATA_TRAIN = "/workspace/data/raw/train.csv"
L5_CSV = "/workspace/data/labels/ours/L5_20261011.csv"
SRC = {
    "raptor": "/workspace/data/labels/public/dreaddevelopment/rsna-knee-labels/labels_llm_soft.parquet",
    "L2": "/workspace/data/labels/public/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
    "L3": "/workspace/data/labels/public/pilkwang/rsna-knee-llm-labels/report_labels_v2.csv",
    "L4": "/workspace/data/labels/public/lixin73/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv",
}

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
KEYS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa", "pf_oa",
        "effusion", "synovitis", "bakers", "contusion", "fracture"]
K2L = dict(zip(KEYS, LABELS))

# ---- prompt: L5 B_reason verbatim (exp054/generar_L5.py), the best single L5 prompt vs the 4-source consensus ----
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
CALIB = ("Calibration guide: clearly stated and meeting the definition 90-100; stated but severity unclear 60-85; "
         "probable/suspected 50-70; possible/cannot be excluded 25-45; stated but explicitly below the threshold "
         "(small, mild, low-grade, early) 10-35; not mentioned 3-8; explicitly normal/intact/denied 0-2.")

SYS_B = f"""You are a senior knee radiologist. You will read one knee MRI report (any language) and estimate, for each of twelve findings, the probability (integer 0-100) that an expert applying the official definitions below would label the finding as PRESENT for this knee.

Definitions:
{DEF_TEXT}

Procedure: first write brief notes (at most 60 words, in English) quoting or translating the key positive and negative statements of the report. Then give the twelve probabilities.
{CALIB}
Use only the report. Do not infer one finding from another. Medial and lateral are distinct."""


def _obj(props, req=None):
    return {"type": "object", "properties": props, "required": req or list(props), "additionalProperties": False}


SCHEMA_B = _obj({"notes": {"type": "string"}, "probabilities": _obj({k: {"type": "integer"} for k in KEYS})})

SYS_ADJ = f"""You are a senior musculoskeletal radiologist acting as adjudicator. Two independent readers read the same knee MRI report (any language) and each estimated, for some findings, the probability (integer 0-100) that an expert applying the official definitions below would label the finding as PRESENT for this knee. On the findings you receive they disagree. Neither reader is more reliable than the other, and both can be wrong.

Definitions:
{DEF_TEXT}

For EACH finding in dispute:
1. Argue against Reading 1: give the strongest reason, grounded in the report text, why its number could be wrong (at most 40 words).
2. Argue against Reading 2 in the same way (at most 40 words).
3. Decide. Give your own probability (integer 0-100) that an expert applying the definitions would label the finding PRESENT, using this scale. {CALIB}
4. If the report is genuinely ambiguous for this finding (the text supports both readings and careful re-reading does not resolve it), set verdict to "ambiguous"; otherwise "decided". Always give your best probability.
Use only the report. Do not infer one finding from another. Medial and lateral are distinct. Return one entry per finding in dispute."""

SCHEMA_ADJ = _obj({"adjudications": {"type": "array", "items": _obj({
    "finding": {"type": "string", "enum": KEYS},
    "against_reading_1": {"type": "string"},
    "against_reading_2": {"type": "string"},
    "verdict": {"type": "string", "enum": ["decided", "ambiguous"]},
    "probability": {"type": "integer"},
})}})

# temperature cannot be set on Sonnet 5.5 / Opus 5.5 (non-default -> 400): declared in the SPEC.
# v1 (archived, informative L6s): Sonnet 5.5, thinking "between_tools" (off), max_tokens 1500.
L6_PARAMS = dict(model=L6_MODEL, max_tokens=16000, system=SYS_B, thinking={"type": "adaptive"},
                 output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA_B}})
ADJ_EFFORT = os.environ.get("L6_ADJ_EFFORT", "max")
ADJ_HIGH_AFTER = datetime(2026, 10, 19, 3, 15, tzinfo=timezone.utc)   # sync adjudications after this -> effort high
HARD_STOP = datetime(2026, 10, 19, 4, 20, tzinfo=timezone.utc)        # no new API calls after this; build and deliver


def now_utc():
    return datetime.now(timezone.utc)


def adj_params(effort):
    return dict(model=ADJ_MODEL, max_tokens=64000, system=SYS_ADJ, thinking={"type": "adaptive"},
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA_ADJ}})


# ------------------------------------------------------------------ utils
def log(out, msg):
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} [labels_claude exp086] {msg}"
    print(line, flush=True)
    with (Path(out) / "status.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def secrets(path):
    env = {}
    if path and Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    key = os.environ.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_API_KEY")
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID") or env.get("ANTHROPIC_WORKSPACE_ID")
    return key, ws


def client(args):
    import anthropic
    key, ws = secrets(args.secrets)
    if not key or not ws:
        raise SystemExit("NO_WORKSPACE_ID: falta ANTHROPIC_API_KEY o ANTHROPIC_WORKSPACE_ID")
    return anthropic.Anthropic(api_key=key, default_headers={"anthropic-workspace-id": ws}, max_retries=6, timeout=600)


def usage_dict(u):
    return {"in": u.input_tokens or 0, "out": u.output_tokens or 0,
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0}


def usd(model, u, batch):
    pi, po = PRICE[model]
    c = (u["in"] * pi + u["cache_read"] * pi * CACHE_READ_FRAC + u["cache_write"] * pi * CACHE_WRITE_FRAC
         + u["out"] * po) / 1e6
    return c * (0.5 if batch else 1.0)


def load_jsonl(path):
    out = {}
    if Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["uid"]] = d
    return out


def append_jsonl(path, rec, lock):
    with lock:
        with Path(path).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def spent(out):
    tot = 0.0
    for path in Path(out).rglob("*.jsonl"):     # includes the archived v1 and the Sonnet L6s
        for r in load_jsonl(path).values():
            tot += r.get("usd", 0.0)
    return tot


def text_of(msg):
    return next((b.text for b in msg.content if b.type == "text"), None)


def reports():
    df = pd.read_csv(DATA_TRAIN, encoding="utf-8")
    df["StudyInstanceUID"] = df.StudyInstanceUID.astype(str)
    return df


def report_msg(report):
    return "REPORT:\n" + str(report).strip()[:12000]


def read_soft(path):
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, low_memory=False)
    df["StudyInstanceUID"] = df.StudyInstanceUID.astype(str)
    return df.set_index("StudyInstanceUID")[LABELS].apply(pd.to_numeric, errors="coerce").astype(float)


# ------------------------------------------------------------------ parsing
def parse_l6(msg):
    if msg.stop_reason == "refusal":
        raise ValueError("refusal")
    obj = json.loads(text_of(msg))
    p = obj["probabilities"]
    return obj, [min(max(float(p[k]), 0.0), 100.0) / 100.0 for k in KEYS]


def parse_adj(msg, disputed):
    if msg.stop_reason == "refusal":
        raise ValueError("refusal")
    obj = json.loads(text_of(msg))
    dec = {}
    for a in obj["adjudications"]:
        if a["finding"] in disputed and a["finding"] not in dec:
            dec[a["finding"]] = {"verdict": a["verdict"],
                                 "p": min(max(float(a["probability"]), 0.0), 100.0) / 100.0}
    return obj, dec


# ------------------------------------------------------------------ disagreement
def l5_table():
    return read_soft(L5_CSV)


def l6_table(out):
    c = load_jsonl(Path(out) / "l6.jsonl")
    return pd.DataFrame({u: r["probs"] for u, r in c.items()}, index=LABELS).T if c else pd.DataFrame(columns=LABELS)


def agree_mask(a, b):
    """True = agreement cell (same side of 0.5 and |a-b| <= AGREE_DELTA)."""
    return ((a > 0.5) == (b > 0.5)) & ((a - b).abs() <= AGREE_DELTA + 1e-9)


def disputes(out, uids=None):
    l5, l6 = l5_table(), l6_table(out)
    ids = [u for u in (uids if uids is not None else l6.index) if u in l6.index and u in l5.index]
    a, b = l5.loc[ids, LABELS], l6.loc[ids, LABELS]
    ok = agree_mask(a, b)
    res = {}
    for u in ids:
        bad = [KEYS[i] for i, lab in enumerate(LABELS) if not ok.at[u, lab]]
        if bad:
            res[u] = {k: (float(a.at[u, K2L[k]]), float(b.at[u, K2L[k]])) for k in bad}
    return res


def adj_user(uid, report, cells):
    """Two readings in a per-study random order (seed 86 + uid); the adjudicator is not told which model is which."""
    flip = random.Random(int(hashlib.sha256(f"{SEED}:{uid}".encode()).hexdigest(), 16)).random() < 0.5
    lines = []
    for k, (p5, p6) in cells.items():
        r1, r2 = (p6, p5) if flip else (p5, p6)
        lines.append(f"- {k} ({K2L[k]}): Reading 1 = {round(r1 * 100)}, Reading 2 = {round(r2 * 100)}")
    txt = report_msg(report) + "\n\nFINDINGS IN DISPUTE (finding key: readings):\n" + "\n".join(lines)
    return txt, flip


# ------------------------------------------------------------------ runners
def run_sync(cl, jobs, worker, workers=8):
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(worker, cl, *j) for j in jobs]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print("ERR", repr(e)[:300], flush=True)
            done += 1
            if done % 100 == 0:
                print(f"  sync {done}/{len(jobs)}", flush=True)


def l6_worker_factory(out, lock):
    def one(cl, uid, report):
        if now_utc() > HARD_STOP:
            return
        for attempt in range(3):
            with cl.messages.stream(messages=[{"role": "user", "content": report_msg(report)}], **L6_PARAMS) as st:
                msg = st.get_final_message()
            try:
                obj, probs = parse_l6(msg)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"FAIL l6 {uid}: {e}", flush=True)
                    return
        u = usage_dict(msg.usage)
        append_jsonl(Path(out) / "l6.jsonl", {"uid": uid, "model": msg.model, "probs": probs, "raw": obj, "usage": u,
                                              "batch": False, "usd": usd(L6_MODEL, u, False)}, lock)
    return one


def adj_worker_factory(out, lock, rep, effort):
    def one(cl, uid, cells):
        if now_utc() > HARD_STOP:
            return
        eff = "high" if (effort == "max" and now_utc() > ADJ_HIGH_AFTER) else effort
        txt, flip = adj_user(uid, rep[uid], cells)
        for attempt in range(3):
            with cl.messages.stream(messages=[{"role": "user", "content": txt}], **adj_params(eff)) as st:
                msg = st.get_final_message()
            try:
                obj, dec = parse_adj(msg, set(cells))
                break
            except Exception as e:
                if attempt == 2:
                    print(f"FAIL adj {uid}: {e}", flush=True)
                    return
        u = usage_dict(msg.usage)
        append_jsonl(Path(out) / "adj.jsonl", {"uid": uid, "model": msg.model, "cells": cells, "flip": flip, "dec": dec,
                                               "raw": obj, "effort": eff, "usage": u, "batch": False,
                                               "usd": usd(ADJ_MODEL, u, False)}, lock)
    return one


def run_batch(cl, out, tag, requests, on_result):
    """Submit one batch (or resume the one in state), poll; > BATCH_MAX_WAIT_S -> cancel. Returns custom_ids not done."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    state_p = Path(out) / f"batch_{tag}.json"
    st = json.loads(state_p.read_text()) if state_p.is_file() else None
    want = {r[0] for r in requests}
    if st is None or not set(st["ids"]) >= want:
        if not requests:
            return set()
        b = cl.messages.batches.create(requests=[Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**p))
                                                 for cid, p in requests])
        st = {"id": b.id, "created": time.time(), "ids": sorted(want)}
        state_p.write_text(json.dumps(st))
        log(out, f"batch {tag} created {b.id} n={len(requests)}")
    while True:
        b = cl.messages.batches.retrieve(st["id"])
        if b.processing_status == "ended":
            break
        if time.time() - st["created"] > BATCH_MAX_WAIT_S or now_utc() > ADJ_HIGH_AFTER:
            log(out, f"batch {tag} > 2 h ({b.request_counts}); cancel -> sync fallback")
            cl.messages.batches.cancel(st["id"])
            for _ in range(60):
                time.sleep(30)
                if cl.messages.batches.retrieve(st["id"]).processing_status == "ended":
                    break
            break
        print(f"  batch {tag} {b.processing_status} {b.request_counts}", flush=True)
        time.sleep(60)
    got = set()
    for r in cl.messages.batches.results(st["id"]):
        if r.result.type == "succeeded":
            try:
                on_result(r.custom_id, r.result.message)
                got.add(r.custom_id)
            except Exception as e:
                print(f"FAIL batch parse {r.custom_id}: {e}", flush=True)
    log(out, f"batch {tag} {st['id']} ended: ok {len(got)}/{len(st['ids'])}")
    return want - got


def cid_of(uid):
    return "s" + hashlib.sha1(uid.encode()).hexdigest()[:40]


# ------------------------------------------------------------------ commands
def cmd_pilot(args):
    out = Path(args.out)
    if (out / "pilot.json").is_file():
        log(out, "pilot.json exists, skip pilot")
        return json.loads((out / "pilot.json").read_text())
    cl = client(args)
    df = reports()
    pil = df.sample(n=args.n, random_state=SEED)
    lock = threading.Lock()
    cache = load_jsonl(out / "l6.jsonl")
    jobs = [(r.StudyInstanceUID, r.Report) for r in pil.itertuples() if r.StudyInstanceUID not in cache]
    log(out, f"pilot L6 sync n={args.n} (todo {len(jobs)})")
    run_sync(cl, jobs, l6_worker_factory(out, lock))
    l6c = load_jsonl(out / "l6.jsonl")
    pr = [l6c[u] for u in pil.StudyInstanceUID if u in l6c]
    l6_full_usd = np.mean([usd(L6_MODEL, r["usage"], True) for r in pr]) * len(df)   # batch price
    disp = disputes(out, list(pil.StudyInstanceUID))
    rep = dict(zip(df.StudyInstanceUID, df.Report))
    adjc = load_jsonl(out / "adj.jsonl")
    ajobs = [(u, c) for u, c in disp.items() if u not in adjc]
    log(out, f"pilot adjudication sync effort={ADJ_EFFORT}: {len(disp)} studies with disputes (todo {len(ajobs)})")
    run_sync(cl, ajobs, adj_worker_factory(out, lock, rep, ADJ_EFFORT))
    adjc = load_jsonl(out / "adj.jsonl")
    ar = [adjc[u] for u in disp if u in adjc]
    per_adj = np.mean([usd(ADJ_MODEL, r["usage"], True) for r in ar]) if ar else 0.0
    frac = len(disp) / len(pr)
    adj_full = per_adj * frac * len(df)
    so_far = spent(out)
    proj = so_far + l6_full_usd + adj_full
    rep_ = {"n": len(pr), "frac_studies_disputed": frac,
            "cells_disputed_frac": sum(len(c) for c in disp.values()) / (12 * len(pr)),
            "l6_tokens_mean": {k: float(np.mean([r["usage"][k] for r in pr])) for k in ("in", "out")},
            "adj_tokens_mean": {k: float(np.mean([r["usage"][k] for r in ar])) for k in ("in", "out")} if ar else None,
            "spent_pilot_usd": so_far, "proj_l6_batch_usd": l6_full_usd, "proj_adj_batch_usd": adj_full,
            "proj_total_usd": proj, "adj_effort": ADJ_EFFORT, "budget_usd": BUDGET_USD}
    (out / "pilot.json").write_text(json.dumps(rep_, indent=1))
    log(out, f"pilot: spent ${so_far:.3f}; projection L6 ${l6_full_usd:.2f} + adj ${adj_full:.2f} = ${proj:.2f} "
             f"(disputed studies {frac:.0%}, cells {rep_['cells_disputed_frac']:.1%})")
    return rep_


def cmd_pilot2(args):
    """Amendment pilot: Opus L6 (effort high) on the same 50-report sample + up to 20 adjudications at effort max."""
    out = Path(args.out)
    if (out / "pilot2.json").is_file():
        log(out, "pilot2.json exists, skip")
        return json.loads((out / "pilot2.json").read_text())
    cl = client(args)
    df = reports()
    pil = df.sample(n=args.n, random_state=SEED)
    lock = threading.Lock()
    cache = load_jsonl(out / "l6.jsonl")
    jobs = [(r.StudyInstanceUID, r.Report) for r in pil.itertuples() if r.StudyInstanceUID not in cache]
    log(out, f"pilot2 L6 Opus high sync n={args.n} (todo {len(jobs)}); spent so far ${spent(out):.2f}")
    run_sync(cl, jobs, l6_worker_factory(out, lock))
    l6c = load_jsonl(out / "l6.jsonl")
    pr = [l6c[u] for u in pil.StudyInstanceUID if u in l6c]
    per_l6 = float(np.mean([usd(L6_MODEL, r["usage"], True) for r in pr]))
    disp = disputes(out, list(pil.StudyInstanceUID))
    sel = dict(list(disp.items())[:20])
    rep_ = dict(zip(df.StudyInstanceUID, df.Report))
    adjc = load_jsonl(out / "adj.jsonl")
    ajobs = [(u, c) for u, c in sel.items() if u not in adjc]
    log(out, f"pilot2 adjudication sync effort={ADJ_EFFORT}: {len(sel)} of {len(disp)} disputed studies")
    run_sync(cl, ajobs, adj_worker_factory(out, lock, rep_, ADJ_EFFORT))
    adjc = load_jsonl(out / "adj.jsonl")
    ar = [adjc[u] for u in sel if u in adjc]
    per_adj = float(np.mean([usd(ADJ_MODEL, r["usage"], True) for r in ar]))
    frac = len(disp) / len(pr)
    so_far = spent(out)
    l6_rest = per_l6 * (len(df) - len(pr))
    adj_rest = per_adj * max(frac * len(df) - len(ar), 0)
    proj = so_far + l6_rest + adj_rest
    r = {"n": len(pr), "frac_studies_disputed": frac,
         "cells_disputed_frac": sum(len(c) for c in disp.values()) / (12 * len(pr)),
         "l6_tokens_mean": {k: float(np.mean([x["usage"][k] for x in pr])) for k in ("in", "out")},
         "adj_tokens_mean": {k: float(np.mean([x["usage"][k] for x in ar])) for k in ("in", "out")},
         "l6_usd_per_report_batch": per_l6, "adj_usd_per_study_batch": per_adj, "n_adj_pilot": len(ar),
         "spent_usd": so_far, "proj_l6_rest_batch_usd": l6_rest, "proj_adj_rest_batch_usd": adj_rest,
         "proj_total_usd": proj, "adj_effort": ADJ_EFFORT, "budget_usd": BUDGET_USD}
    (out / "pilot2.json").write_text(json.dumps(r, indent=1))
    log(out, f"pilot2: spent ${so_far:.2f}; projection L6 Opus rest ${l6_rest:.2f} + adj max rest ${adj_rest:.2f} "
             f"= ${proj:.2f} (disputed studies {frac:.0%}; L6 out tok {r['l6_tokens_mean']['out']:.0f}, "
             f"adj out tok {r['adj_tokens_mean']['out']:.0f})")
    return r


def cmd_l6(args):
    out = Path(args.out)
    cl = client(args)
    df = reports()
    lock = threading.Lock()
    cache = load_jsonl(out / "l6.jsonl")
    todo = [(r.StudyInstanceUID, r.Report) for r in df.itertuples() if r.StudyInstanceUID not in cache]
    log(out, f"L6: {len(todo)} reports to do (cache {len(cache)}); spent so far ${spent(out):.2f}")
    if not todo:
        return
    by_cid = {cid_of(u): u for u, _ in todo}

    def on_result(cid, msg):
        uid = by_cid[cid]
        obj, probs = parse_l6(msg)
        u = usage_dict(msg.usage)
        append_jsonl(out / "l6.jsonl", {"uid": uid, "model": msg.model, "probs": probs, "raw": obj, "usage": u,
                                        "batch": True, "usd": usd(L6_MODEL, u, True)}, lock)

    reqs = [(cid_of(u), dict(messages=[{"role": "user", "content": report_msg(rp)}], **L6_PARAMS)) for u, rp in todo]
    left = run_batch(cl, out, "l6", reqs, on_result) if not args.sync else set(by_cid)
    cache = load_jsonl(out / "l6.jsonl")
    rest = [(u, rp) for u, rp in todo if u not in cache]
    if rest:
        log(out, f"L6 sync fallback for {len(rest)} reports")
        run_sync(cl, rest, l6_worker_factory(out, lock), workers=args.workers)
    log(out, f"L6 done: {len(load_jsonl(out / 'l6.jsonl'))}/{len(df)}; spent ${spent(out):.2f}")


def cmd_adjud(args):
    out = Path(args.out)
    cl = client(args)
    df = reports()
    rep = dict(zip(df.StudyInstanceUID, df.Report))
    lock = threading.Lock()
    effort = ADJ_EFFORT
    disp = disputes(out)
    adjc = load_jsonl(out / "adj.jsonl")
    todo = {u: c for u, c in disp.items() if u not in adjc}
    # budget guard: projected cost of the remaining adjudications at batch price
    per = np.mean([usd(ADJ_MODEL, r["usage"], True) for r in adjc.values()]) if adjc else 0.01
    proj = spent(out) + per * len(todo)
    log(out, f"adjudication: {len(disp)} studies with disputes, {sum(len(c) for c in disp.values())} cells; "
             f"todo {len(todo)}; effort {effort}; projection total ${proj:.2f}")
    if proj > BUDGET_USD * 0.97:
        # SPEC rule: adjudicate studies in decreasing order of the largest |L5 - L6| gap until the budget is used
        room = max(BUDGET_USD * 0.97 - spent(out), 0) / max(per, 1e-6)
        order = sorted(todo, key=lambda u: -max(abs(a - b) for a, b in todo[u].values()))
        todo = {u: todo[u] for u in order[:int(room)]}
        log(out, f"BUDGET: only {len(todo)} studies adjudicated (largest gaps first); rest -> mean(L5, L6)")
    if not todo:
        return
    by_cid = {cid_of(u): u for u in todo}
    flips = {}

    def on_result(cid, msg):
        uid = by_cid[cid]
        obj, dec = parse_adj(msg, set(todo[uid]))
        u = usage_dict(msg.usage)
        append_jsonl(out / "adj.jsonl", {"uid": uid, "model": msg.model, "cells": todo[uid], "flip": flips[uid],
                                         "dec": dec, "raw": obj, "effort": effort, "usage": u, "batch": True,
                                         "usd": usd(ADJ_MODEL, u, True)}, lock)

    reqs = []
    for u, c in todo.items():
        txt, flips[u] = adj_user(u, rep[u], c)
        reqs.append((cid_of(u), dict(messages=[{"role": "user", "content": txt}], **adj_params(effort))))
    if not args.sync:
        run_batch(cl, out, "adj", reqs, on_result)
    adjc = load_jsonl(out / "adj.jsonl")
    rest = [(u, c) for u, c in todo.items() if u not in adjc]
    if rest:
        log(out, f"adjudication sync fallback for {len(rest)} studies")
        run_sync(cl, rest, adj_worker_factory(out, lock, rep, effort), workers=args.workers)
    log(out, f"adjudication done: {len(load_jsonl(out / 'adj.jsonl'))} studies; spent ${spent(out):.2f}")


def auc_table(gold, preds):
    from sklearn.metrics import roc_auc_score
    res = {}
    for name, p in preds.items():
        per = {}
        for lab in LABELS:
            y = gold[lab]
            s = p.reindex(gold.index)[lab]
            m = y.notna() & s.notna()
            per[lab] = float(roc_auc_score(y[m], s[m])) if m.sum() > 1 and y[m].nunique() > 1 else None
        vals = [v for v in per.values() if v is not None]
        res[name] = {"macro": float(np.mean(vals)) if vals else None, "per_label": per,
                     "n_gold": int(p.index.isin(gold.index).sum())}
    return res


def macro_auc_vs(ref_bin, p, ids):
    from sklearn.metrics import roc_auc_score
    per = {}
    for lab in LABELS:
        y, s = ref_bin.reindex(ids)[lab], p.reindex(ids)[lab]
        m = y.notna() & s.notna()
        per[lab] = float(roc_auc_score(y[m], s[m])) if y[m].nunique() > 1 else None
    vals = [v for v in per.values() if v is not None]
    return float(np.mean(vals)), per


def cmd_build(args):
    out = Path(args.out)
    df = reports()
    uids = list(df.StudyInstanceUID)
    gold = df.dropna(subset=LABELS, how="all").set_index("StudyInstanceUID")[LABELS].astype(float)
    assert len(gold) == 58
    l5 = l5_table().reindex(uids)
    l6 = l6_table(out).reindex(uids)
    l6.index.name = "StudyInstanceUID"
    l6.reset_index().to_csv(out / "L6_opus55.csv", index=False)
    _s = load_jsonl(out / "l6s_sonnet.jsonl")
    l6s = (pd.DataFrame({u: r["probs"] for u, r in _s.items()}, index=LABELS).T if _s
           else pd.DataFrame(columns=LABELS)).reindex(uids)
    adjc = load_jsonl(out / "adj.jsonl")

    ok = agree_mask(l5, l6)
    have6 = l6.notna()
    tgt = pd.DataFrame(np.nan, index=l5.index, columns=LABELS)
    src = pd.DataFrame("", index=l5.index, columns=LABELS)
    agree = ok & have6
    for lab in LABELS:
        a = agree[lab]
        tgt.loc[a, lab] = np.where(l6.loc[a, lab] > 0.5, SHARP_POS, SHARP_NEG)
        src.loc[a, lab] = "agree"
        no6 = ~have6[lab]
        tgt.loc[no6, lab] = l5.loc[no6, lab]
        src.loc[no6, lab] = "l5_only"
    mean56 = (l5 + l6) / 2
    n_dec = n_amb = n_missing = 0
    for u in l5.index:
        for i, lab in enumerate(LABELS):
            if src.at[u, lab]:
                continue
            k = KEYS[i]
            d = adjc.get(u, {}).get("dec", {}).get(k)
            if d is None:
                tgt.at[u, lab] = mean56.at[u, lab]; src.at[u, lab] = "adj_missing"; n_missing += 1
            elif d["verdict"] == "ambiguous":
                tgt.at[u, lab] = mean56.at[u, lab]; src.at[u, lab] = "ambiguous"; n_amb += 1
            else:
                tgt.at[u, lab] = d["p"]; src.at[u, lab] = "adjudicated"; n_dec += 1
    cons_v6_nogoldfix = tgt.copy()
    final = tgt.copy()
    final.loc[gold.index, LABELS] = gold[LABELS].to_numpy()
    assert final.notna().all().all() and final.stack().between(0, 1).all()
    final.index.name = "StudyInstanceUID"
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    final.reset_index().to_csv(dest / "consenso_v6.csv", index=False)
    src.index.name = "StudyInstanceUID"
    src.reset_index().to_csv(out / "consenso_v6_source.csv", index=False)

    # ---- informative metrics (nothing is selected with gold)
    S = {n: read_soft(p) for n, p in SRC.items()}
    def mean_of(names):
        st = np.stack([S[n].reindex(uids)[LABELS].to_numpy(float) for n in names])
        n = np.isfinite(st).sum(0)
        return pd.DataFrame(np.where(n > 0, np.nansum(st, 0) / np.maximum(n, 1), np.nan), index=uids, columns=LABELS)
    cons4 = mean_of(["raptor", "L2", "L3", "L4"])
    st5 = np.stack([S[n].reindex(uids)[LABELS].to_numpy(float) for n in ["raptor", "L2", "L3", "L4"]] + [l5.to_numpy()])
    n5 = np.isfinite(st5).sum(0)
    cons5 = pd.DataFrame(np.where(n5 > 0, np.nansum(st5, 0) / np.maximum(n5, 1), np.nan), index=uids, columns=LABELS)
    gold_auc = auc_table(gold, {"L5": l5, "L6": l6, "L6s_sonnet": l6s, "consenso_actual_4src": cons4, "consenso_5src_L5": cons5,
                                "consenso_v6": cons_v6_nogoldfix})
    nongold = [u for u in uids if u not in set(gold.index)]
    c4bin = (cons4 > 0.5).astype(float).where(cons4.notna())
    health_macro, health_per = macro_auc_vs(c4bin, l6, nongold)
    l6_vs = {n: macro_auc_vs((S[n] > 0.5).astype(float).where(S[n].notna()), l6, nongold)[0] for n in S}
    l6_vs["L5"] = macro_auc_vs((l5 > 0.5).astype(float).where(l5.notna()), l6, nongold)[0]
    l6_vs["L6s_sonnet"] = macro_auc_vs((l6s > 0.5).astype(float).where(l6s.notna()), l6, nongold)[0]
    l6s_health = macro_auc_vs(c4bin, l6s, nongold)[0]
    cells = int(have6.to_numpy().sum())
    agree_frac = float(agree.to_numpy().sum() / max(cells, 1))
    agree_per = {lab: float(agree[lab].sum() / max(have6[lab].sum(), 1)) for lab in LABELS}
    coverage = float(l6.notna().all(axis=1).mean())
    cost = spent(out)
    l6c = load_jsonl(out / "l6.jsonl")
    tok = {"l6_in": sum(r["usage"]["in"] for r in l6c.values()), "l6_out": sum(r["usage"]["out"] for r in l6c.values()),
           "adj_in": sum(r["usage"]["in"] for r in adjc.values()), "adj_out": sum(r["usage"]["out"] for r in adjc.values())}
    healthy = (coverage >= 0.99 and health_macro >= 0.93 and min(v for v in health_per.values() if v is not None) >= 0.80
               and cost <= BUDGET_USD)
    metrics = {
        "schema_version": 1, "exp": "exp086_etiquetas_l6_claude", "parent": "exp054", "phase": "labels",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "models": {"L6": L6_MODEL + " effort high", "L6s_informative": L6S_MODEL,
                   "adjudicator": ADJ_MODEL + " effort " + ADJ_EFFORT},
        "l6s_sonnet_macro_auc_vs_cons4_nongold": l6s_health,
        "adj_effort_counts": pd.Series([r.get("effort") for r in adjc.values()]).value_counts().to_dict(),
        "agreement_rule": {"same_side_of": 0.5, "max_abs_diff": AGREE_DELTA, "sharp": [SHARP_NEG, SHARP_POS]},
        "n_studies": len(uids), "l6_coverage_full_rows": coverage,
        "l6_health": {"macro_auc_vs_cons4_nongold": health_macro, "per_label": health_per,
                      "macro_auc_vs_source_nongold": l6_vs, "healthy": healthy},
        "agreement_cells_frac": agree_frac, "agreement_per_label": agree_per,
        "n_cells_disputed": int((src.isin(["adjudicated", "ambiguous", "adj_missing"])).to_numpy().sum()),
        "n_cells_adjudicated_decided": n_dec, "n_cells_ambiguous": n_amb, "n_cells_adj_missing": n_missing,
        "n_studies_adjudicated": len(adjc), "n_cells_l5_only": int((src == "l5_only").to_numpy().sum()),
        "mean_target": {"consenso_v6": float(np.nanmean(cons_v6_nogoldfix.to_numpy())),
                        "L5": float(np.nanmean(l5.to_numpy())), "L6": float(np.nanmean(l6.to_numpy())),
                        "cons4": float(np.nanmean(cons4.to_numpy()))},
        "gold_auc_informative": gold_auc,
        "cost_usd": round(cost, 4), "tokens": tok,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    warn = None
    g6, g4 = gold_auc["consenso_v6"]["macro"], gold_auc["consenso_actual_4src"]["macro"]
    if g6 is not None and g4 is not None and g6 < g4 - 0.02:
        warn = f"consenso_v6 gold macro AUC {g6:.4f} < consenso actual {g4:.4f} - 0.02 (informative)"
    if healthy:
        ready = {"path": str(dest / "consenso_v6.csv"), "n_studies": int(len(final)),
                 "n_adjudicated": n_dec + n_amb, "n_studies_adjudicated": len(adjc), "cost_usd": round(cost, 2),
                 "created_at": datetime.now(timezone.utc).isoformat(), "gold_rows": "hard gold 0/1 (58)",
                 "format": "StudyInstanceUID + 12 LABELS, prob in [0,1]", "warning": warn}
        (dest / "READY.json").write_text(json.dumps(ready, indent=1))
    log(out, f"build: coverage {coverage:.4f} health {health_macro:.4f} agree {agree_frac:.3f} decided {n_dec} "
             f"ambiguous {n_amb} missing {n_missing} cost ${cost:.2f} healthy={healthy} "
             f"gold macro L5 {gold_auc['L5']['macro']:.4f} L6 {gold_auc['L6']['macro']:.4f} "
             f"cons4 {g4:.4f} v6 {g6:.4f}")
    return metrics


def cmd_all(args):
    p = cmd_pilot(args)
    out = Path(args.out)
    if "adj_effort_final" not in p:
        if p["proj_total_usd"] <= BUDGET_USD:
            p["adj_effort_final"] = p["adj_effort"]
        elif p["spent_pilot_usd"] + p["proj_l6_batch_usd"] <= BUDGET_USD:
            # SPEC rule: over budget -> adjudicator effort low + budget guard (largest gaps first, rest = mean(L5, L6))
            p["adj_effort_final"] = "low"
            log(out, f"projection ${p['proj_total_usd']:.2f} > ${BUDGET_USD} -> adjudicator effort low + budget guard")
        else:
            log(out, f"OVER_BUDGET: L6 alone projects ${p['proj_l6_batch_usd']:.2f}; stop")
            raise SystemExit(2)
        (out / "pilot.json").write_text(json.dumps(p, indent=1))
    cmd_l6(args)
    cmd_adjud(args)
    cmd_build(args)


def cmd_all2(args):
    p = cmd_pilot2(args)
    out = Path(args.out)
    if p["proj_total_usd"] > BUDGET_USD and not (out / "GO_OVER_BUDGET").is_file():
        log(out, f"OVER_BUDGET_100: projection ${p['proj_total_usd']:.2f} > ${BUDGET_USD}; waiting for the coordinator "
                 f"(create work/GO_OVER_BUDGET to continue with the budget guard)")
        raise SystemExit(3)
    cmd_l6(args)
    cmd_adjud(args)
    cmd_build(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "pilot2", "l6", "adjud", "build", "all", "all2"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dest", default="/workspace/users/rodrigo/orq/labels_claude")
    ap.add_argument("--secrets", default="/root/.anthropic_env")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sync", action="store_true", help="skip the Batch API")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"pilot": cmd_pilot, "pilot2": cmd_pilot2, "l6": cmd_l6, "adjud": cmd_adjud, "build": cmd_build, "all": cmd_all,
     "all2": cmd_all2}[a.cmd](a)
