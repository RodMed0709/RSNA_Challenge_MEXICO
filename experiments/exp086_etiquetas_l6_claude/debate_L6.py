"""exp086 amendment 2 - adversarial debate L5-side (OpenAI) vs L6 (Claude Opus 5.5) on the disagreeing cells.

Per study with >= 1 disputed cell (all its disputed cells in one conversation turn):
  round r (max 3): GPT turn (attacks the other reader's arguments, defends/revises, quotes the report)
                   -> agreement check -> Claude Opus turn (effort high, same task) -> agreement check.
  agreement = same side of 0.5 and |pA - pB| <= 0.30 -> cell leaves the debate, target 0.95 / 0.05.
  after 3 rounds without agreement -> final judge Claude Opus effort max with the whole transcript;
  'ambiguous' -> mean of the two last probabilities.
GPT starts from its previous reading = L5; Claude from L6 (Opus) with its notes as argument.
Readers are named neutrally ("the other reader"); the judge sees "Reader A" (GPT) / "Reader B" (Claude).

  python debate_L6.py pilot --out WORK [--n 10]   GPT cost probe on 10 disputed studies (round-1 GPT turn)
  python debate_L6.py run   --out WORK            full debate (resumable, state in WORK/debate_state.json)
  python debate_L6.py build --out WORK --dest D   consenso_v6.csv (+ READY.json), metrics
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import generar_L6 as g

GPT_MODEL = g.os.environ.get("L6_GPT_MODEL", "gpt-5.6-sol")
GPT_FALLBACK = "gpt-5.6-terra"
GPT_PRICE = {"gpt-5.6-sol": (4.00, 0.40, 20.00), "gpt-5.6-terra": (2.00, 0.20, 12.00), "gpt-5.4": (2.50, 0.25, 15.00)}
GPT_EFFORT = "medium"
OPENAI_BUDGET = 45.0
ANTHROPIC_BUDGET = 100.0
MAX_ROUNDS = 3
CLAUDE_EFFORT = "high"
BATCH_WAIT_S = 3600            # Claude turn batch: > 1 h -> cancel, sync fallback
LATE = datetime(2026, 10, 19, 2, 30, tzinfo=timezone.utc)     # after this: no round 3 (limit to 2 rounds)
STOP = datetime(2026, 10, 19, 4, 15, tzinfo=timezone.utc)     # no new calls after this; build and deliver

SYS_DEBATE = f"""You are a senior musculoskeletal radiologist in an adversarial review. You and another reader independently read the same knee MRI report (any language) and estimated, per finding, the probability (integer 0-100) that an expert applying the official definitions below would label the finding PRESENT for this knee. On the findings you receive you disagree.

Definitions:
{g.DEF_TEXT}

For EACH finding listed:
1. Attack the other reader's argument: say concretely where it misreads, over-reads or under-reads the report (or concede that it is right).
2. Defend or revise your own probability. Revise only if the report text supports it; do not move just to agree.
3. Quote the exact words of the report (original language) that decide the finding, or "" if the report says nothing about it.
Give your probability (integer 0-100) with this scale. {g.CALIB}
Use only the report. Do not infer one finding from another. Medial and lateral are distinct. Return one entry per finding listed, each argument in 1-2 sentences."""

SCHEMA_TURN = g._obj({"cells": {"type": "array", "items": g._obj({
    "finding": {"type": "string", "enum": g.KEYS},
    "attack_on_other": {"type": "string"},
    "probability": {"type": "integer"},
    "quote": {"type": "string"},
    "argument": {"type": "string"},
})}})

SYS_JUDGE = f"""You are a senior musculoskeletal radiologist acting as final judge. Two readers (Reader A and Reader B) read the same knee MRI report (any language) and debated for several rounds, per finding, the probability (integer 0-100) that an expert applying the official definitions below would label the finding PRESENT. They did not reach agreement on the findings you receive. Neither reader is more reliable than the other, and both can be wrong.

Definitions:
{g.DEF_TEXT}

For EACH finding: argue against Reader A's final position (at most 40 words), argue against Reader B's final position (at most 40 words), then decide your own probability (integer 0-100) with this scale. {g.CALIB}
If the report is genuinely ambiguous for this finding (the text supports both positions and careful re-reading does not resolve it), set verdict to "ambiguous"; otherwise "decided". Always give your best probability.
Use only the report. Return one entry per finding. Field against_reading_1 = against Reader A, against_reading_2 = against Reader B."""


def now():
    return datetime.now(timezone.utc)


def log(out, msg):
    g.log(out, "debate: " + msg)


def openai_client(args):
    from openai import OpenAI
    env = {}
    for line in Path(args.secrets).read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return OpenAI(api_key=env["OPENAI_API_KEY"], max_retries=6, timeout=600)


def gpt_usd(model, u):
    pi, pc, po = GPT_PRICE[model]
    return ((u["in"] - u["cached"]) * pi + u["cached"] * pc + u["out"] * po) / 1e6


# ------------------------------------------------------------------ state
def state_path(out):
    return Path(out) / "debate_state.json"


def load_state(out):
    p = state_path(out)
    return json.loads(p.read_text()) if p.is_file() else None


def save_state(out, st):
    tmp = state_path(out).with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False))
    tmp.replace(state_path(out))


def init_state(out):
    disp = g.disputes(out)                                  # L5 vs L6 (Opus)
    l6c = g.load_jsonl(Path(out) / "l6.jsonl")
    st = {"gpt_model": GPT_MODEL, "studies": {}, "phases_done": []}
    for u, cells in disp.items():
        notes = l6c[u]["raw"].get("notes", "")
        st["studies"][u] = {"cells": {k: {"p5": p5, "p6": p6, "pA": p5, "pB": p6,
                                          "lastA": {"argument": "(initial reading, no argument)", "quote": ""},
                                          "lastB": {"argument": notes, "quote": ""},
                                          "turns": [], "status": "open", "round_agreed": None}
                                      for k, (p5, p6) in cells.items()}}
    return st


def agree(a, b):
    return ((a > 0.5) == (b > 0.5)) and abs(a - b) <= g.AGREE_DELTA + 1e-9


def open_cells(st):
    return {u: [k for k, c in s["cells"].items() if c["status"] == "open"] for u, s in st["studies"].items()
            if any(c["status"] == "open" for c in s["cells"].values())}


def turn_user(report, cells, who, keys):
    if who == "A":
        mine, oth_last, oth_p = "pA", "lastB", "pB"
    else:
        mine, oth_last, oth_p = "pB", "lastA", "pA"
    lines = []
    for k in keys:
        c = cells[k]
        lines.append(f"- {k} ({g.K2L[k]}): your probability = {round(c[mine] * 100)}; other reader = "
                     f"{round(c[oth_p] * 100)}; other reader's argument: {c[oth_last]['argument']!s}"
                     + (f" | other reader's quote: \"{c[oth_last]['quote']}\"" if c[oth_last]["quote"] else ""))
    return g.report_msg(report) + "\n\nFINDINGS IN DISPUTE:\n" + "\n".join(lines)


def parse_turn(obj, keys):
    res = {}
    for a in obj["cells"]:
        if a["finding"] in keys and a["finding"] not in res:
            res[a["finding"]] = {"p": min(max(float(a["probability"]), 0.0), 100.0) / 100.0, "quote": a["quote"],
                                 "argument": a["argument"], "attack": a["attack_on_other"]}
    return res


def apply_turn(st, u, who, rnd, res):
    s = st["studies"][u]
    for k, r in res.items():
        c = s["cells"][k]
        if c["status"] != "open":
            continue
        c["p" + who] = r["p"]
        c["last" + who] = {"argument": r["argument"], "quote": r["quote"]}
        c["turns"].append({"round": rnd, "who": who, **r})
        if agree(c["pA"], c["pB"]):
            c["status"] = "agreed"
            c["round_agreed"] = rnd
            c["agreed_after"] = who


# ------------------------------------------------------------------ GPT turn (sync, concurrent)
def gpt_turn(args, st, rnd, out, ids=None, model=None):
    model = model or st["gpt_model"]
    oc = openai_client(args)
    rep = dict(zip(*[g.reports()[c] for c in ["StudyInstanceUID", "Report"]]))
    oc_cells = open_cells(st)
    if ids is not None:
        oc_cells = {u: oc_cells[u] for u in ids if u in oc_cells}
    lock = threading.Lock()
    path = Path(out) / f"debate_gpt_r{rnd}.jsonl"
    done = g.load_jsonl(path)

    def one(u, keys):
        if now() > STOP:
            return
        if u in done:
            return
        txt = turn_user(rep[u], st["studies"][u]["cells"], "A", keys)
        for attempt in range(3):
            r = oc.chat.completions.create(
                model=model, reasoning_effort=GPT_EFFORT, max_completion_tokens=16000,
                messages=[{"role": "system", "content": SYS_DEBATE}, {"role": "user", "content": txt}],
                response_format={"type": "json_schema", "json_schema": {"name": "turn", "schema": SCHEMA_TURN,
                                                                        "strict": True}})
            try:
                obj = json.loads(r.choices[0].message.content)
                res = parse_turn(obj, set(keys))
                break
            except Exception as e:
                if attempt == 2:
                    print(f"FAIL gpt {u}: {e}", flush=True)
                    return
        uu = r.usage
        cached = getattr(getattr(uu, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        usage = {"in": uu.prompt_tokens, "cached": cached, "out": uu.completion_tokens}
        g.append_jsonl(path, {"uid": u, "round": rnd, "model": r.model, "keys": keys, "res": res, "usage": usage,
                              "usd_openai": gpt_usd(model, usage)}, lock)

    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(one, u, k) for u, k in oc_cells.items()]
        for i, f in enumerate(as_completed(futs)):
            try:
                f.result()
            except Exception as e:
                print("ERR gpt", repr(e)[:300], flush=True)
            if (i + 1) % 200 == 0:
                print(f"  gpt r{rnd} {i + 1}/{len(futs)} ${spent_openai(out):.2f}", flush=True)
    done = g.load_jsonl(path)
    for u in oc_cells:
        if u in done:
            apply_turn(st, u, "A", rnd, done[u]["res"])
    return len(oc_cells)


# ------------------------------------------------------------------ Claude turn (batch, sync fallback)
def claude_params(user_txt):
    return dict(model=g.ADJ_MODEL, max_tokens=16000, system=SYS_DEBATE, thinking={"type": "adaptive"},
                output_config={"effort": CLAUDE_EFFORT, "format": {"type": "json_schema", "schema": SCHEMA_TURN}},
                messages=[{"role": "user", "content": user_txt}])


def claude_turn(args, st, rnd, out):
    cl = g.client(args)
    rep = dict(zip(*[g.reports()[c] for c in ["StudyInstanceUID", "Report"]]))
    oc_cells = open_cells(st)
    path = Path(out) / f"debate_claude_r{rnd}.jsonl"
    lock = threading.Lock()
    done = g.load_jsonl(path)
    todo = {u: k for u, k in oc_cells.items() if u not in done}
    txts = {u: turn_user(rep[u], st["studies"][u]["cells"], "B", k) for u, k in todo.items()}

    def rec(u, msg, batch):
        if msg.stop_reason == "refusal":
            raise ValueError("refusal")
        obj = json.loads(g.text_of(msg))
        res = parse_turn(obj, set(todo[u]))
        usage = g.usage_dict(msg.usage)
        g.append_jsonl(path, {"uid": u, "round": rnd, "model": msg.model, "keys": todo[u], "res": res,
                              "usage": usage, "batch": batch, "usd": g.usd(g.ADJ_MODEL, usage, batch)}, lock)

    if todo and now() < STOP:
        by_cid = {g.cid_of(u): u for u in todo}
        reqs = [(g.cid_of(u), claude_params(txts[u])) for u in todo]
        old = g.BATCH_MAX_WAIT_S
        g.BATCH_MAX_WAIT_S = BATCH_WAIT_S
        try:
            g.run_batch(cl, out, f"debate_claude_r{rnd}", reqs, lambda cid, m: rec(by_cid[cid], m, True))
        finally:
            g.BATCH_MAX_WAIT_S = old
        done = g.load_jsonl(path)
        rest = [u for u in todo if u not in done]
        if rest:
            log(out, f"claude r{rnd} sync fallback {len(rest)}")

            def one(u):
                if now() > STOP:
                    return
                for attempt in range(3):
                    with cl.messages.stream(**claude_params(txts[u])) as s_:
                        m = s_.get_final_message()
                    try:
                        rec(u, m, False)
                        return
                    except Exception as e:
                        if attempt == 2:
                            print(f"FAIL claude {u}: {e}", flush=True)
            with ThreadPoolExecutor(args.workers) as ex:
                for f in as_completed([ex.submit(one, u) for u in rest]):
                    try:
                        f.result()
                    except Exception as e:
                        print("ERR claude", repr(e)[:300], flush=True)
    done = g.load_jsonl(path)
    for u in oc_cells:
        if u in done:
            apply_turn(st, u, "B", rnd, done[u]["res"])
    return len(oc_cells)


# ------------------------------------------------------------------ judge (Opus max) after the debate
def judge_user(report, cells, keys):
    lines = []
    for k in keys:
        c = cells[k]
        lines.append(f"\n## {k} ({g.K2L[k]}) - initial: Reader A {round(c['p5'] * 100)}, Reader B {round(c['p6'] * 100)}")
        for t in c["turns"]:
            lines.append(f"- round {t['round']} Reader {t['who']}: {round(t['p'] * 100)} | quote: \"{t['quote']}\" | "
                         f"argument: {t['argument']} | attack on the other: {t['attack']}")
        lines.append(f"Final: Reader A {round(c['pA'] * 100)}, Reader B {round(c['pB'] * 100)}")
    return g.report_msg(report) + "\n\nDEBATE TRANSCRIPT PER FINDING:" + "\n".join(lines)


def judge(args, st, out):
    cl = g.client(args)
    rep = dict(zip(*[g.reports()[c] for c in ["StudyInstanceUID", "Report"]]))
    oc_cells = open_cells(st)
    path = Path(out) / "debate_judge.jsonl"
    lock = threading.Lock()
    done = g.load_jsonl(path)
    todo = {u: k for u, k in oc_cells.items() if u not in done}
    # budget guard (Anthropic): largest |pA - pB| first
    per = 0.06
    room = max(ANTHROPIC_BUDGET * 0.98 - g.spent(out), 0) / per
    if len(todo) > room:
        order = sorted(todo, key=lambda u: -max(abs(st["studies"][u]["cells"][k]["pA"] - st["studies"][u]["cells"][k]["pB"])
                                                 for k in todo[u]))
        todo = {u: todo[u] for u in order[:int(room)]}
        log(out, f"judge BUDGET: only {len(todo)} studies (largest gaps first)")

    def params(u):
        return dict(model=g.ADJ_MODEL, max_tokens=64000, system=SYS_JUDGE, thinking={"type": "adaptive"},
                    output_config={"effort": "max", "format": {"type": "json_schema", "schema": g.SCHEMA_ADJ}},
                    messages=[{"role": "user", "content": judge_user(rep[u], st["studies"][u]["cells"], todo[u])}])

    def rec(u, msg, batch):
        obj, dec = g.parse_adj(msg, set(todo[u]))
        usage = g.usage_dict(msg.usage)
        g.append_jsonl(path, {"uid": u, "model": msg.model, "dec": dec, "raw": obj, "usage": usage, "batch": batch,
                              "usd": g.usd(g.ADJ_MODEL, usage, batch)}, lock)

    if todo and now() < STOP:
        by_cid = {g.cid_of(u): u for u in todo}
        old = g.BATCH_MAX_WAIT_S
        g.BATCH_MAX_WAIT_S = BATCH_WAIT_S
        try:
            g.run_batch(cl, out, "debate_judge", [(g.cid_of(u), params(u)) for u in todo],
                        lambda cid, m: rec(by_cid[cid], m, True))
        finally:
            g.BATCH_MAX_WAIT_S = old
        done = g.load_jsonl(path)
        rest = [u for u in todo if u not in done]

        def one(u):
            if now() > STOP:
                return
            p = params(u)
            if now() > g.ADJ_HIGH_AFTER:
                p["output_config"]["effort"] = "high"
            with cl.messages.stream(**p) as s_:
                rec(u, s_.get_final_message(), False)
        if rest:
            log(out, f"judge sync fallback {len(rest)}")
            with ThreadPoolExecutor(args.workers) as ex:
                for f in as_completed([ex.submit(one, u) for u in rest]):
                    try:
                        f.result()
                    except Exception as e:
                        print("ERR judge", repr(e)[:300], flush=True)
    done = g.load_jsonl(path)
    for u, keys in oc_cells.items():
        for k in keys:
            c = st["studies"][u]["cells"][k]
            d = done.get(u, {}).get("dec", {}).get(k)
            if d is None:
                c["status"] = "unresolved"
            elif d["verdict"] == "ambiguous":
                c["status"] = "ambiguous"
            else:
                c["status"] = "judged"
                c["p_judge"] = d["p"]


def spent_openai(out):
    return sum(r.get("usd_openai", 0.0) for p in Path(out).glob("debate_gpt_*.jsonl")
               for r in g.load_jsonl(p).values())


# ------------------------------------------------------------------ commands
def cmd_pilot(args):
    out = Path(args.out)
    st = init_state(out)          # not saved: L6 may still be partial during the probe
    ids = sorted(open_cells(st))[:args.n]
    import copy
    tmp = copy.deepcopy(st)
    gpt_turn(args, tmp, 1, out, ids=ids)
    recs = [g.load_jsonl(out / "debate_gpt_r1.jsonl")[u] for u in ids]
    per = float(np.mean([r["usd_openai"] for r in recs]))
    n = len(st["studies"])
    proj = per * n * 2.0      # ~2 GPT turns per study on average over the 3 rounds (assumption)
    r = {"gpt_model": st["gpt_model"], "n_probe": len(recs), "usd_per_turn": per, "n_studies": n,
         "n_cells": sum(len(s["cells"]) for s in st["studies"].values()),
         "tok_mean": {k: float(np.mean([x["usage"][k] for x in recs])) for k in ("in", "out")},
         "proj_openai_usd": proj}
    (out / "debate_pilot.json").write_text(json.dumps(r, indent=1))
    log(out, f"pilot {st['gpt_model']}: ${per:.4f}/turn, {n} studies -> projection OpenAI ${proj:.2f} (2 turns/study)")
    return r


def cmd_run(args):
    out = Path(args.out)
    st = load_state(out) or init_state(out)
    if "start" not in st:
        st["start"] = now().isoformat()
        st["n_cells_start"] = sum(len(s["cells"]) for s in st["studies"].values())
    save_state(out, st)
    log(out, f"{len(st['studies'])} studies, {st['n_cells_start']} cells in dispute; GPT {st['gpt_model']}")
    for rnd in range(1, MAX_ROUNDS + 1):
        if rnd == 3 and now() > LATE:
            log(out, "late: debate limited to 2 rounds")
            break
        for who, fn in (("gpt", gpt_turn), ("claude", claude_turn)):
            ph = f"r{rnd}_{who}"
            if ph in st["phases_done"]:
                continue
            if who == "gpt" and spent_openai(out) > OPENAI_BUDGET * 0.95:
                log(out, "OpenAI budget reached: stop debate")
                break
            if who == "claude" and g.spent(out) > ANTHROPIC_BUDGET * 0.85:
                log(out, "Anthropic budget near: stop debate (judge keeps the rest)")
                break
            n = fn(args, st, rnd, out)
            st["phases_done"].append(ph)
            save_state(out, st)
            oc = open_cells(st)
            log(out, f"{ph}: {n} studies in; open now {len(oc)} studies / {sum(len(v) for v in oc.values())} cells; "
                     f"spent Anthropic ${g.spent(out):.2f} OpenAI ${spent_openai(out):.2f}")
    if "judge" not in st["phases_done"]:
        judge(args, st, out)
        st["phases_done"].append("judge")
        save_state(out, st)
    log(out, f"debate done; spent Anthropic ${g.spent(out):.2f} OpenAI ${spent_openai(out):.2f}")


def cmd_build(args):
    """consenso_v6 with the debate: reuse generar_L6.build for L6 metrics, then override disputed cells."""
    out, dest = Path(args.out), Path(args.dest)
    st = load_state(out)
    df = g.reports()
    uids = list(df.StudyInstanceUID)
    gold = df.dropna(subset=g.LABELS, how="all").set_index("StudyInstanceUID")[g.LABELS].astype(float)
    l5 = g.l5_table().reindex(uids)
    l6 = g.l6_table(out).reindex(uids)
    ok = g.agree_mask(l5, l6)
    tgt = pd.DataFrame(np.nan, index=uids, columns=g.LABELS)
    for lab in g.LABELS:
        a = ok[lab] & l6[lab].notna()
        tgt.loc[a, lab] = np.where(l6.loc[a, lab] > 0.5, g.SHARP_POS, g.SHARP_NEG)
        no6 = l6[lab].isna()
        tgt.loc[no6, lab] = l5.loc[no6, lab]
    pres = (l5 + l6) / 2                      # v6soft: unsharpened resolution probability
    pres = pres.fillna(l5)
    rows, counts = [], {}
    for u, s in (st or {"studies": {}})["studies"].items():
        for k, c in s["cells"].items():
            lab = g.K2L[k]
            stt = c["status"]
            if stt == "agreed":
                v = g.SHARP_POS if (c["pA"] + c["pB"]) / 2 > 0.5 else g.SHARP_NEG
            elif stt == "judged":
                v = c["p_judge"]
            else:                                   # ambiguous / unresolved / still open -> mean of last two
                v = (c["pA"] + c["pB"]) / 2
            tgt.at[u, lab] = v
            pres.at[u, lab] = c["p_judge"] if stt == "judged" else (c["pA"] + c["pB"]) / 2
            movA, movB = abs(c["pA"] - c["p5"]), abs(c["pB"] - c["p6"])
            who = "both" if abs(movA - movB) < 0.05 else ("GPT" if movA > movB else "Claude")
            counts[stt] = counts.get(stt, 0) + 1
            rows.append({"status": stt, "round": c.get("round_agreed"), "conceded": who if stt == "agreed" else None,
                         "label": lab, "n_turns": len(c["turns"])})
    remaining = int(tgt.isna().to_numpy().sum())
    tgt = tgt.fillna((l5 + l6) / 2)     # disputed cells outside the debate state (none expected)
    cons = tgt.copy()
    final = tgt.copy()
    final.loc[gold.index, g.LABELS] = gold[g.LABELS].to_numpy()
    assert final.notna().all().all() and final.stack().between(0, 1).all()
    final.index.name = "StudyInstanceUID"
    dest.mkdir(parents=True, exist_ok=True)
    final.reset_index().to_csv(dest / "consenso_v6.csv", index=False)
    R = pd.DataFrame(rows)
    # informative metrics
    S = {n: g.read_soft(p) for n, p in g.SRC.items()}
    st4 = np.stack([S[n].reindex(uids)[g.LABELS].to_numpy(float) for n in S])
    n4 = np.isfinite(st4).sum(0)
    cons4 = pd.DataFrame(np.where(n4 > 0, np.nansum(st4, 0) / np.maximum(n4, 1), np.nan), index=uids, columns=g.LABELS)
    _s = g.load_jsonl(out / "l6s_sonnet.jsonl")
    l6s = pd.DataFrame({u: r["probs"] for u, r in _s.items()}, index=g.LABELS).T.reindex(uids)
    nongold = [u for u in uids if u not in set(gold.index)]
    c4bin = (cons4 > 0.5).astype(float).where(cons4.notna())
    health, health_per = g.macro_auc_vs(c4bin, l6, nongold)
    coverage = float(l6.notna().all(axis=1).mean())
    # v6soft (orchestrator, 18-oct): 0.5 * consenso4 + 0.5 * p_res, no sharpening; gold-58 hard
    soft = (0.5 * cons4 + 0.5 * pres).where(cons4.notna(), pres)
    soft_final = soft.copy()
    soft_final.loc[gold.index, g.LABELS] = gold[g.LABELS].to_numpy()
    assert soft_final.notna().all().all() and soft_final.stack().between(0, 1).all()
    soft_final.index.name = "StudyInstanceUID"
    gold_auc = g.auc_table(gold, {"L5": l5, "L6_opus": l6, "L6s_sonnet": l6s, "consenso_actual_4src": cons4,
                                  "consenso_v6": cons, "consenso_v6soft": soft})
    a_cost, o_cost = g.spent(out), spent_openai(out)
    agree0 = float((ok & l6.notna()).to_numpy().sum() / max(int(l6.notna().to_numpy().sum()), 1))
    m = {
        "schema_version": 1, "exp": "exp086_etiquetas_l6_claude", "parent": "exp054", "phase": "labels",
        "timestamp_utc": now().isoformat(),
        "models": {"L6": g.L6_MODEL + " effort high", "L6s_informative": g.L6S_MODEL,
                   "debate": {"A": (st or {}).get("gpt_model"), "B": g.ADJ_MODEL + " effort high"},
                   "judge": g.ADJ_MODEL + " effort max"},
        "n_studies": len(uids), "l6_coverage_full_rows": coverage,
        "l6_health": {"macro_auc_vs_cons4_nongold": health, "per_label": health_per,
                      "l6s_sonnet_macro_auc_vs_cons4_nongold": g.macro_auc_vs(c4bin, l6s, nongold)[0]},
        "agreement_initial_cells_frac": agree0,
        "agreement_initial_per_label": {lab: float((ok[lab] & l6[lab].notna()).mean()) for lab in g.LABELS},
        "debate": {"n_studies": len((st or {"studies": {}})["studies"]), "n_cells": len(R),
                   "status_counts": counts,
                   "agreed_by_round": R[R.status == "agreed"]["round"].value_counts().sort_index().to_dict() if len(R) else {},
                   "conceded_counts": R["conceded"].value_counts().to_dict() if len(R) else {},
                   "phases_done": (st or {}).get("phases_done")},
        "cells_filled_outside_debate": remaining,
        "mean_target": {"consenso_v6": float(np.nanmean(cons.to_numpy())), "consenso_v6soft": float(np.nanmean(soft.to_numpy())), "L5": float(np.nanmean(l5.to_numpy())),
                        "L6": float(np.nanmean(l6.to_numpy())), "cons4": float(np.nanmean(cons4.to_numpy()))},
        "gold_auc_informative": gold_auc,
        "cost_usd": {"anthropic": round(a_cost, 4), "openai": round(o_cost, 4), "total": round(a_cost + o_cost, 4)},
    }
    m = json.loads(json.dumps(m, default=lambda o: int(o) if isinstance(o, (np.integer,)) else float(o)))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    healthy = coverage >= 0.99 and health >= 0.93 and min(v for v in health_per.values() if v is not None) >= 0.80
    g6, g4 = gold_auc["consenso_v6"]["macro"], gold_auc["consenso_actual_4src"]["macro"]
    if healthy:
        soft_final.reset_index().to_csv(dest / "consenso_v6soft.csv", index=False)
        (dest / "READY_soft.json").write_text(json.dumps({
            "path": str(dest / "consenso_v6soft.csv"), "created_at": now().isoformat(), "n_studies": int(len(soft_final)),
            "regla": "por celda 0.5*consenso4 (Raptor+L2+L3+L4, media de presentes) + 0.5*p_res; p_res = media(L5, L6-Opus) "
                     "si acordaron de entrada; media de las probs finales si acordaron en el debate; prob del juez si "
                     "juzgado; media de las dos ultimas si ambiguous/sin resolver. Sin afilar. Gold-58 = gold dura 0/1.",
            "gold_macro_auc_informative_before_gold_fix": gold_auc["consenso_v6soft"]["macro"]}, indent=1))
        (dest / "READY.json").write_text(json.dumps({
            "path": str(dest / "consenso_v6.csv"), "n_studies": int(len(final)),
            "n_adjudicated": len(R), "n_studies_adjudicated": m["debate"]["n_studies"],
            "cost_usd": round(a_cost + o_cost, 2), "cost_detail": m["cost_usd"], "created_at": now().isoformat(),
            "gold_rows": "hard gold 0/1 (58)", "format": "StudyInstanceUID + 12 LABELS, prob in [0,1]",
            "debate_status_counts": counts,
            "warning": (f"consenso_v6 gold macro AUC {g6:.4f} < consenso actual {g4:.4f} - 0.02 (informative; "
                        f"sharpening creates ties)") if (g6 is not None and g4 is not None and g6 < g4 - 0.02)
            else None}, indent=1))
    f4 = lambda v: "nan" if v is None else f"{v:.4f}"
    log(out, f"build: healthy={healthy} L6 health {health:.4f} agree0 {agree0:.3f} debate {counts} "
             f"cost A ${a_cost:.2f} O ${o_cost:.2f}; gold macro L5 {f4(gold_auc['L5']['macro'])} "
             f"L6 {f4(gold_auc['L6_opus']['macro'])} L6s {f4(gold_auc['L6s_sonnet']['macro'])} cons4 {f4(g4)} v6 {f4(g6)} "
             f"v6soft {f4(gold_auc['consenso_v6soft']['macro'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "run", "build"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dest", default="/workspace/users/rodrigo/orq/labels_claude")
    ap.add_argument("--secrets", default="/root/.anthropic_env")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    {"pilot": cmd_pilot, "run": cmd_run, "build": cmd_build}[a.cmd](a)
