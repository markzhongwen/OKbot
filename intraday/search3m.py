#!/usr/bin/env python3
"""Continuous 3m search, REPORT ONLY. Never touches paper."""
import sys, json, random, time, gc, os
import numpy as np, pandas as pd
sys.path.insert(0, "C:/Users/matth/okx-bt/intraday")
import week_fit_3m_lean as W, engine as E, evolve as Ev

R = "C:/Users/matth/okx-bt/intraday"
MD = f"{R}/SEARCH_3m.md"
JL = f"{R}/search_3m.jsonl"
ST = f"{R}/SEARCH_3m_STATUS.md"
BASE = dict(
    week=0.1636, month=0.0860, prior=0.0390,
    name="ext_rev short L1 m500 k4 SL3 TP4 ts80 cool30 8%x8 max3",
)
X = E.load("3m")
end = int(X["t"][-1])
tfm = 3
w7 = (end - 7 * 1440, end + 1)
w30 = end - 30 * 1440
pr = (end - 14 * 1440, end - 7 * 1440)
t = X["t"]
o, h, l = X["o"], X["h"], X["l"]
F = {k: X[k].astype(np.float64) for k in "ohlcf"}
bj = lambda m: str(pd.to_datetime(m, unit="m") + pd.Timedelta(hours=8))


def rand_base():
    """Weighted family draw. Bias MR/extreme-rev; add engine families lightly."""
    r = random.random()
    # short bias matches current week regime; keep some both
    d = random.choice(["short", "short", "short", "both"])
    htf = random.choice([0, 0, 0, 240])  # occasional HTF filter (web MR advice)
    he = random.choice([20, 50])

    if r < 0.32:
        return "ext_rev", dict(
            L=random.choice([1, 1, 2, 3]),
            m=random.choice([200, 300, 500, 800, 1000]),
            k=random.choice([3, 3.5, 4, 4.5, 5, 6]),
            dir=d, htf=htf, htf_ema=he,
        )
    if r < 0.48:
        # Connors-style RSI-2 + classic RSI-14 (public MR)
        return "rsi_mr", dict(
            n=random.choice([2, 2, 3, 5, 7, 14, 14]),
            lo=random.choice([5, 10, 10, 15, 20, 25]),
            dir=d, htf=htf, htf_ema=he,
        )
    if r < 0.58:
        return "bb_mr", dict(
            n=random.choice([10, 20, 20, 30, 50]),
            k=random.choice([1.5, 2, 2.5, 2.5, 3]),
            dir=d, htf=htf, htf_ema=he,
        )
    if r < 0.66:
        return "keltner", dict(
            n=random.choice([20, 30, 50, 80]),
            k=random.choice([1.5, 2, 2.5, 3]),
            fade=random.choice([0, 1, 1]),
            dir=d, htf=0, htf_ema=20,
        )
    if r < 0.72:
        return "vol_rev", dict(
            n=random.choice([20, 50, 100]),
            k=random.choice([3, 4, 5, 6]),
            b=random.choice([0.5, 1, 1.5]),
            fade=random.choice([0, 1]),
            dir=d, htf=0, htf_ema=20,
        )
    if r < 0.77:
        return "donchian", dict(
            n=random.choice([20, 55, 96, 160]),
            fade=random.choice([0, 1, 1]),
            dir=d, htf=0, htf_ema=20,
        )
    if r < 0.82:
        return "vwap_mr", dict(
            n=random.choice([20, 48, 96, 288]),
            k=random.choice([1, 1.5, 2, 3]),
            dir=d, htf=htf, htf_ema=he,
        )
    if r < 0.86:
        fade = random.choice([0, 1])
        fam = "squeeze_fade" if fade else "squeeze"
        return fam, dict(
            n=random.choice([10, 20, 30]),
            m=random.choice([200, 500, 1000]),
            q=random.choice([0.05, 0.1, 0.2]),
            fade=fade, dir=d, htf=0, htf_ema=20,
        )
    if r < 0.90:
        return "xs", dict(
            L=random.choice([1, 2, 3, 6]),
            k=random.choice([1, 2, 3]),
            reb=random.choice([1, 2, 4, 8]),
            mode=random.choice(["rev", "rev", "mom"]),
            dir=d, htf=0, htf_ema=20,
        )
    if r < 0.93:
        return "trend_pb", dict(
            n=random.choice([2, 3, 5, 7, 14]),
            lo=random.choice([10, 20, 30, 40]),
            hmin=random.choice([60, 240]),
            hema=random.choice([20, 50, 200]),
            dir=d, htf=0, htf_ema=20,
        )
    if r < 0.95:
        return "macd", dict(
            f=random.choice([6, 12]),
            sl_=random.choice([26, 35]),
            sg=random.choice([5, 9]),
            ema=random.choice([50, 100, 200]),
            dir=d, htf=0, htf_ema=20,
        )
    if r < 0.97:
        return "avwap_gap", dict(
            anchor=random.choice([0, 480, 780, 840]),
            k=random.choice([1.5, 2, 3, 4]),
            fade=random.choice([0, 1]),
            dir=d, htf=0, htf_ema=20,
        )
    if r < 0.985:
        return "liq_casc", dict(
            L=random.choice([1, 2, 4]),
            mv=random.choice([0.5, 1, 2]),
            br=random.choice([0.5, 0.7, 0.9]),
            k=random.choice([1.5, 2, 3]),
            vz=random.choice([1.5, 2, 3]),
            dir=d, htf=0, htf_ema=20,
        )
    # rare: fund_time only (supertrend omitted: pure-Python loop OOMs/stalls on 3m)
    return "fund_time", dict(
        pre=random.choice([-60, -30, -15, 15, 30, 60]),
        thr=random.choice([0, 0.5, 1, 2]),
        side=random.choice([1, -1]),
    )


def rand_stop():
    return dict(
        sl=random.choice([2, 2.5, 3, 3.5, 4]),
        tp=random.choice([0, 2, 3, 4, 5, 6]),
        ts=random.choice([20, 40, 60, 80, 100, 140]),
        cool=random.choice([10, 20, 30, 45, 60]),
    )


SIZES = [
    dict(margin=0.08, lev=8, maxpos=3),
    dict(margin=0.10, lev=8, maxpos=3),
    dict(margin=0.06, lev=10, maxpos=4),
    dict(margin=0.10, lev=5, maxpos=3),
]


def robust_score(r):
    return 0.5 * r["week"]["net"] + 0.5 * r["month"]["net"]


def beats_week(r):
    return r["week"]["net"] > BASE["week"]


def beats_robust(r):
    # claim "better" only if week beats baseline AND 30d at least baseline month
    return r["week"]["net"] > BASE["week"] and r["month"]["net"] >= BASE["month"]


def fmt_row(i, r, tag=""):
    w, m, p = r["week"], r["month"], r["prior"]
    bw = "YES" if beats_week(r) else "no"
    br = "YES" if beats_robust(r) else "no"
    tag = f" [{tag}]" if tag else ""
    return (
        f"{i}. {r['fam']} {json.dumps(r['p'], ensure_ascii=False)} sz {r['sz']}{tag}: "
        f"week **{w['net']:+.2%}** DD {w['dd']:.1%} {w['tr']}tr | "
        f"30d {m['net']:+.2%} DD {m['dd']:.1%} | prior {p['net']:+.2%} | "
        f"robust={robust_score(r):+.2%} | beats week: {bw} | beats robust: {br}"
    )


def load_seed(limit=80):
    """Recover prior strong passers so top board is not empty after restart."""
    rows = []
    if not os.path.exists(JL):
        return rows
    with open(JL) as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            if not isinstance(o, dict) or "week" not in o or "month" not in o:
                continue
            if "fam" not in o:
                # infer
                p = o.get("p", {})
                if "L" in p and "m" in p and "k" in p:
                    o["fam"] = "ext_rev"
                elif "lo" in p:
                    o["fam"] = "rsi_mr"
                else:
                    o["fam"] = o.get("fam") or "?"
            rows.append(o)
    # keep diverse top by week and by robust
    by_w = sorted(rows, key=lambda r: r["week"]["net"], reverse=True)
    by_r = sorted(rows, key=robust_score, reverse=True)
    seen = set()
    out = []
    for pool in (by_r, by_w):
        for r in pool:
            k = json.dumps([r.get("fam"), r.get("p"), r.get("sz")], sort_keys=True)
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
            if len(out) >= limit:
                return out
    return out


best = load_seed(60)
seen = set()
for r in best:
    seen.add(json.dumps([r.get("fam"), r.get("p")], sort_keys=True))
n = 0
n_pass = 0
t0 = time.time()
seed_n = len(best)


def write_boards():
    by_week = sorted(best, key=lambda r: r["week"]["net"], reverse=True)
    by_rob = sorted(best, key=robust_score, reverse=True)
    rob_beaters = [r for r in by_rob if beats_robust(r)]
    week_beaters = [r for r in by_week if beats_week(r)]

    L = [
        "# 3m search (REPORT ONLY, paper not switched)",
        f"Updated {pd.Timestamp.now(tz='Asia/Shanghai'):%F %T} BJ | evaluated {n} configs this run | "
        f"seeded {seed_n} from jsonl | uptime {time.time()-t0:.0f}s",
        f"Window week BJ {bj(w7[0])} .. {bj(end)}; prior {bj(pr[0])}..{bj(pr[1])}; 30d from {bj(w30)}",
        "Cost taker 0.07%/side + funding. Gates: liq=0, SL inside liq, week tr>=10, week DD>=-30%, prior>=0, 30d>=0.",
        f"Baseline (paper3m): {BASE['name']} week +16.36% / 30d +8.60% / prior +3.90%",
        "Ranking: **best week** = max week net; **best robust** = 0.5*week+0.5*30d. "
        "beats robust = week>baseline week AND 30d>=baseline 30d.",
        f"Families: ext_rev,rsi_mr,bb_mr,keltner,vol_rev,donchian,vwap_mr,squeeze(+fade),xs,trend_pb,macd,avwap_gap,liq_casc,fund_time",
        "",
        "## Best week (top 5)",
    ]
    if by_week:
        for i, r in enumerate(by_week[:5]):
            L.append(fmt_row(i + 1, r, "week"))
    else:
        L.append("(no passers yet)")

    L.append("")
    L.append("## Best robust week+30d (top 5)")
    if by_rob:
        for i, r in enumerate(by_rob[:5]):
            L.append(fmt_row(i + 1, r, "robust"))
    else:
        L.append("(no passers yet)")

    L.append("")
    L.append(f"## Robust beaters vs baseline ({len(rob_beaters)})")
    if rob_beaters:
        for i, r in enumerate(rob_beaters[:8]):
            L.append(fmt_row(i + 1, r, "BEATS"))
    else:
        L.append("(none yet that beat week AND keep 30d >= baseline)")

    L.append("")
    L.append(f"Week-only beaters (30d may be weaker): {len(week_beaters)}")
    L.append("\nCaveat: one-week fit, heavy multiple testing -> overfit risk. Paper NOT switched.")
    open(MD, "w").write("\n".join(L) + "\n")

    # compact STATUS for parent/routine
    top_w = by_week[0] if by_week else None
    top_r = by_rob[0] if by_rob else None
    top_b = rob_beaters[0] if rob_beaters else None
    S = [
        "# SEARCH_3m_STATUS",
        f"Updated {pd.Timestamp.now(tz='Asia/Shanghai'):%F %T} BJ",
        f"Service: okx-search3m (REPORT ONLY). Paper3m/paper untouched.",
        f"Evaluated this run: {n} | jsonl seed on start: {seed_n} | passers held in memory: {len(best)} | new passes this run: {n_pass}",
        f"Baseline: week +16.36% / 30d +8.60% / prior +3.90%",
        f"Robust beaters (week>16.36% AND 30d>=8.60%): **{len(rob_beaters)}**",
        f"Week-only beaters: {len(week_beaters)}",
        "",
    ]
    if top_b:
        S.append("## Strongest robust beater")
        S.append(fmt_row(1, top_b, "BEATS"))
        S.append("")
    else:
        S.append("## Strongest robust beater")
        S.append("(none)")
        S.append("")
    if top_w:
        S.append("## Best week")
        S.append(fmt_row(1, top_w, "week"))
        S.append("")
    if top_r:
        S.append("## Best robust score")
        S.append(fmt_row(1, top_r, "robust"))
        S.append("")
    S.append("Families active: ext_rev rsi_mr bb_mr keltner vol_rev donchian vwap_mr squeeze xs trend_pb macd avwap_gap liq_casc fund_time")
    S.append("Public scan note: RSI-2/Connors + BB z-score MR + HTF filter ideas folded into rsi_mr/bb_mr grids and occasional htf=240.")
    open(ST, "w").write("\n".join(S) + "\n")


write_boards()

while True:
    fam, bp = rand_base()
    st = rand_stop()
    # fund_time has no dir/sl in signal the same way — still needs stops
    p = dict(bp, **{k: st[k] for k in st if k not in bp})
    if "sl" not in p:
        p.update(st)
    k = json.dumps([fam, p], sort_keys=True)
    if k in seen:
        continue
    seen.add(k)
    try:
        s, A = E.signal(X, Ev.efam(fam), p, tfm)
        A = np.nan_to_num(A, nan=0.0)
        e, x, sd, g, fu, prr = E.sim(
            F["o"], F["h"], F["l"], F["c"], F["f"], s, A,
            float(p["sl"]), float(p["tp"]), int(p["ts"]), int(p["cool"]), 0,
        )
        e = np.atleast_1d(np.asarray(e))
        if e.size == 0:
            del s, A
            continue
    except Exception:
        continue
    for sz in SIZES:
        n += 1
        try:
            w = W.size_from_trades(t, e, x, sd, g, fu, prr, A, o, h, l, p, sz, *w7)
            if w["liq"] or w["sl_beyond"] or w["tr"] < 10 or w["dd"] < -0.30 or w["net"] <= 0:
                continue
            pw = W.size_from_trades(t, e, x, sd, g, fu, prr, A, o, h, l, p, sz, *pr)
            m = W.size_from_trades(t, e, x, sd, g, fu, prr, A, o, h, l, p, sz, w30, w7[1])
            if pw["net"] < 0 or m["net"] < 0 or m["liq"] or pw["liq"]:
                continue
        except Exception:
            continue
        r = dict(fam=fam, p=p, sz=sz, week=w, month=m, prior=pw)
        open(JL, "a").write(json.dumps(r) + "\n")
        n_pass += 1
        best.append(r)
        by_r = sorted(best, key=robust_score, reverse=True)[:30]
        by_w = sorted(best, key=lambda x: x["week"]["net"], reverse=True)[:20]
        keep = {}
        for xx in by_r + by_w:
            kk = json.dumps([xx.get("fam"), xx.get("p"), xx.get("sz")], sort_keys=True)
            keep[kk] = xx
        best = list(keep.values())
    try:
        del s, A
    except Exception:
        pass
    if n % 80 < len(SIZES) or n < 20:
        write_boards()
        gc.collect()
