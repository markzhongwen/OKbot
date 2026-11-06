#!/usr/bin/env python3
"""Week-regime fit for 5m and 3m. BACKTEST ONLY. Fee-aware. REPORT ONLY — does not touch paper."""
import os,sys,json,math,itertools,time
import numpy as np,pandas as pd
sys.path.insert(0,"C:/Users/matth/okx-bt/intraday")
import engine as E
import evolve as Ev

R="C:/Users/matth/okx-bt/intraday"
EQ0=25.0; COST=0.0007; MMR=0.005
OUT=f"{R}/week_fit_5m3m_out.json"
MD=f"{R}/WEEK_FIT_5m3m.md"
# Match WEEK_REGIME.md window: last 7d ending at latest bar of each TF load
# (same relative window as week_fit.py)

def size_from_trades(t,e,x,sd,g,fu,pr,A,o,h,l,p,sz,t0,t1,mk=0):
    CI=0.0002 if mk else COST
    if len(e)==0:
        return dict(final=EQ0,net=0.0,dd=0.0,sh=0.0,tr=0,liq=0,sl_beyond=0,worst_day=0.0)
    keep=(t[e]>=t0)&(t[e]<t1)
    e,x,sd,g,fu,pr=[np.asarray(v)[keep] for v in (e,x,sd,g,fu,pr)]
    if len(e)==0:
        return dict(final=EQ0,net=0.0,dd=0.0,sh=0.0,tr=0,liq=0,sl_beyond=0,worst_day=0.0)
    A=np.nan_to_num(A)
    sl_d=np.array([p["sl"]*A[max(e[i]-1,0),pr[i]]/max(o[e[i],pr[i]],1e-12) for i in range(len(e))])
    mae=np.array([(o[e[i],pr[i]]-l[e[i]:x[i]+1,pr[i]].min())/o[e[i],pr[i]] if sd[i]>0
                  else (h[e[i]:x[i]+1,pr[i]].max()-o[e[i],pr[i]])/o[e[i],pr[i]] for i in range(len(e))])
    margin,lev,maxpos=float(sz["margin"]),int(sz["lev"]),int(sz["maxpos"]); liqd=1.0/lev-MMR
    idx=np.argsort(e,kind="stable")
    eq=EQ0; opn=[]; curve=[]; liq=0; slb=0; peak=EQ0; mdd=0.0
    for i in idx:
        for j in [k for k in opn if k[0]<=e[i]]:
            eq+=j[1]; opn.remove(j); peak=max(peak,eq); mdd=min(mdd,eq/peak-1 if peak>0 else 0); curve.append((t[j[0]],eq))
        if len(opn)>=maxpos or eq<=0.5: continue
        if sl_d[i]>=liqd: slb+=1; continue
        notional=eq*margin*lev
        if mae[i]>=liqd: pnl=-eq*margin-COST*notional; liq+=1
        else: pnl=notional*(g[i]-CI-COST-fu[i])
        opn.append((x[i],pnl))
    for j in sorted(opn, key=lambda z:z[0]):
        eq+=j[1]; peak=max(peak,eq); mdd=min(mdd,eq/peak-1 if peak>0 else 0); curve.append((t[j[0]],eq))
    if not curve:
        return dict(final=EQ0,net=0.0,dd=0.0,sh=0.0,tr=0,liq=0,sl_beyond=0,worst_day=0.0)
    ser=pd.Series([v for _,v in curve],index=pd.to_datetime([k for k,_ in curve],unit="m")).groupby(level=0).last()
    d=ser.resample("D").last().ffill(); rt=d.pct_change().dropna()
    sh=float(rt.mean()/rt.std()*math.sqrt(365)) if len(rt)>2 and rt.std()>0 else 0.0
    return dict(final=float(round(eq,2)),net=float(round(eq/EQ0-1,4)),dd=float(round(mdd,4)),
                sh=float(round(sh,3)),tr=int(len(curve)),liq=int(liq),sl_beyond=int(slb),
                worst_day=float(round(float(rt.min()) if len(rt) else 0,4)))

def signal_bases(tfm):
    """Short-biased families for mild risk-off week. Params in minutes where applicable."""
    bases=[]
    # ORB — open_min/range/win in minutes (engine uses minutes)
    for open_min,range_min,win_min,fade,d,htf in itertools.product(
        [0,420,480,810,840,870],[30,60],[60,120],[0,1],["short","both"],[0,240]):
        bases.append(("orb", dict(open_min=open_min,range_min=range_min,win_min=win_min,fade=fade,dir=d,htf=htf,htf_ema=20)))
    for n,k,d in itertools.product([10,20,30],[1.5,2.0,2.5],["short","both"]):
        bases.append(("bb_mr", dict(n=n,k=k,dir=d,htf=0,htf_ema=20)))
    for n,lo,d in itertools.product([5,7,14],[15,20,25],["short","both"]):
        bases.append(("rsi_mr", dict(n=n,lo=lo,dir=d,htf=0,htf_ema=20)))
    for n,k,fade,d in itertools.product([20,50],[2,2.5],[0,1],["short","both"]):
        bases.append(("keltner", dict(n=n,k=k,fade=fade,dir=d,htf=0,htf_ema=20)))
    for n,fade,d in itertools.product([20,55,96],[0,1],["short","both"]):
        bases.append(("donchian", dict(n=n,fade=fade,dir=d,htf=0,htf_ema=20)))
    for n,k,b,d in itertools.product([20,50],[3,4],[1],["short","both"]):
        bases.append(("vol_rev", dict(n=n,k=k,b=b,fade=1,dir=d,htf=0,htf_ema=20)))
    for L,m,k,d in itertools.product([1,2],[200,500],[3,4,5],["short","both"]):
        bases.append(("ext_rev", dict(L=L,m=m,k=k,dir=d,htf=0,htf_ema=20)))
    for ema,d in itertools.product([50,200],["short","both"]):
        bases.append(("macd", dict(f=12,sl_=26,sg=9,ema=ema,dir=d,htf=0,htf_ema=20)))
    for n,k,d in itertools.product([10,14],[2,3],["short","both"]):
        bases.append(("supertrend", dict(n=n,k=k,dir=d,htf=0,htf_ema=20)))
    for n,d in itertools.product([25,49],["short","both"]):
        bases.append(("hma", dict(n=n,dir=d,htf=0,htf_ema=20)))
    for fade,d in itertools.product([0,1],["short","both"]):
        bases.append(("squeeze", dict(n=20,m=500,q=0.1,fade=fade,dir=d,htf=0,htf_ema=20)))
    seen=set(); out=[]
    for fam,bp in bases:
        k=json.dumps([fam,bp],sort_keys=True)
        if k in seen: continue
        seen.add(k); out.append((fam,bp))
    return out

def stops_for(tfm):
    """Wall-clock-scaled stops vs 15m reference, plus a couple shorter exits."""
    # 15m refs: ts 8/12/16 bars = 120/180/240 min; cool 3/6 = 45/90 min
    def b(mins): return max(2, int(round(mins/tfm)))
    out=[]
    for sl,tp,ts_m,cool_m in [
        (2,0,120,45),(2,0,180,90),(3,0,180,90),
        (2,3,180,45),(1.5,2,120,45),(3,4,240,90),
        # faster HF exits
        (2,3,60,30),(3,4,90,45),(2,0,90,30),
    ]:
        out.append(dict(sl=sl,tp=tp,ts=b(ts_m),cool=b(cool_m)))
    # dedupe
    seen=set(); uniq=[]
    for s in out:
        k=json.dumps(s,sort_keys=True)
        if k in seen: continue
        seen.add(k); uniq.append(s)
    return uniq

SIZES=[
    dict(margin=0.08,lev=8,maxpos=3),  # match current paper
    dict(margin=0.10,lev=5,maxpos=3),
    dict(margin=0.10,lev=3,maxpos=5),
    dict(margin=0.12,lev=5,maxpos=2),
]

def search_tf(tf):
    t0=time.time()
    print(f"=== LOAD {tf} ===", flush=True)
    X=E.load(tf)
    end_m=int(X["t"][-1])
    w7_0,w7_1=end_m-7*1440, end_m+1
    w30_0=end_m-30*1440
    prior_0,prior_1=end_m-14*1440, end_m-7*1440
    t=X["t"]; o,h,l=X["o"],X["h"],X["l"]; tfm=E.TF_MIN[tf]
    win={
        "tf":tf,"tfm":tfm,
        "week_m":[w7_0,w7_1],"month_m":[w30_0,w7_1],"prior_m":[prior_0,prior_1],
        "last_bar_utc":str(pd.to_datetime(X["t"][-1],unit="m")),
        "last_bar_bj":str(pd.to_datetime(X["t"][-1],unit="m")+pd.Timedelta(hours=8)),
        "week_bj":[
            str(pd.to_datetime(w7_0,unit="m")+pd.Timedelta(hours=8)),
            str(pd.to_datetime(end_m,unit="m")+pd.Timedelta(hours=8)),
        ],
    }
    print("window",json.dumps(win), flush=True)

    # baseline: current 15m keltner short params ported (bar stops scaled)
    base_p={"n":20,"k":2,"fade":0,"dir":"short","htf":0,"htf_ema":20,
            "sl":3,"tp":4,"ts":max(2,int(round(240/tfm))),"cool":max(2,int(round(90/tfm)))}
    base_sz={"margin":0.08,"lev":8,"maxpos":3}
    def eval_full(fam,p,sz):
        s,A=E.signal(X,Ev.efam(fam),p,tfm)
        e,x,sd,g,fu,pr=E.sim(X["o"].astype(np.float64),X["h"].astype(np.float64),X["l"].astype(np.float64),
            X["c"].astype(np.float64),X["f"].astype(np.float64),s,np.nan_to_num(A,nan=0.0),
            float(p["sl"]),float(p.get("tp",0)),int(p["ts"]),int(p.get("cool",1)),0)
        return (size_from_trades(t,e,x,sd,g,fu,pr,A,o,h,l,p,sz,w7_0,w7_1),
                size_from_trades(t,e,x,sd,g,fu,pr,A,o,h,l,p,sz,w30_0,w7_1),
                size_from_trades(t,e,x,sd,g,fu,pr,A,o,h,l,p,sz,prior_0,prior_1))

    bw,bm,bp=eval_full("keltner",base_p,base_sz)
    print("PORTED_KELTNER week",bw,"month",bm,"prior",bp, flush=True)

    bases=signal_bases(tfm)
    STOPS=stops_for(tfm)
    print(f"bases={len(bases)} stops={len(STOPS)} sizes={len(SIZES)} total~{len(bases)*len(STOPS)*len(SIZES)}", flush=True)
    rows=[]
    for bi,(fam,bp0) in enumerate(bases):
        try:
            s,A=E.signal(X,Ev.efam(fam),dict(bp0,sl=2,tp=0,ts=12,cool=3),tfm)
        except Exception as e:
            print("sig fail",fam,bp0,type(e).__name__,e, flush=True); continue
        A=np.nan_to_num(A,nan=0.0)
        for st in STOPS:
            p=dict(bp0); p.update(st)
            try:
                e,x,sd,g,fu,pr=E.sim(X["o"].astype(np.float64),X["h"].astype(np.float64),X["l"].astype(np.float64),
                    X["c"].astype(np.float64),X["f"].astype(np.float64),s,A,
                    float(p["sl"]),float(p.get("tp",0)),int(p["ts"]),int(p.get("cool",1)),0)
            except Exception:
                continue
            for sz in SIZES:
                w=size_from_trades(t,e,x,sd,g,fu,pr,A,o,h,l,p,sz,w7_0,w7_1)
                # gates: liq=0, prefer trades>=10, week DD>=-30%, positive week
                if w["liq"] or w["sl_beyond"] or w["net"]<=0 or w["dd"]<-0.30:
                    continue
                if w["tr"]<5:  # keep some for ranking but soft; hard filter later for report
                    continue
                rows.append(dict(fam=fam,p=p,sz=sz,week=w,_pack=(e,x,sd,g,fu,pr,A)))
        if (bi+1)%20==0:
            print(f"  {tf} base {bi+1}/{len(bases)} surv={len(rows)} {time.time()-t0:.0f}s", flush=True)

    rows.sort(key=lambda r: (r["week"]["net"], r["week"]["sh"]), reverse=True)
    print(f"{tf} week-pos survivors {len(rows)} in {time.time()-t0:.0f}s", flush=True)

    top=[]
    for r in rows[:120]:
        e,x,sd,g,fu,pr,A=r.pop("_pack")
        m=size_from_trades(t,e,x,sd,g,fu,pr,A,o,h,l,r["p"],r["sz"],w30_0,w7_1)
        pw=size_from_trades(t,e,x,sd,g,fu,pr,A,o,h,l,r["p"],r["sz"],prior_0,prior_1)
        r["month"]=m; r["prior"]=pw
        sc=r["week"]["net"]*100
        if r["week"]["tr"]>=10: sc+=4
        if r["week"]["tr"]>=20: sc+=2
        if m["net"]>0: sc+=5
        if m["dd"]>=-0.30: sc+=2
        if pw["net"]>0: sc+=3
        if r["week"]["dd"]>=-0.15: sc+=1
        if m["liq"] or pw["liq"]: sc-=20
        sc += min(4.0, r["week"]["tr"]/5.0)
        r["score"]=round(sc,3)
        top.append(r)
    top.sort(key=lambda r: r["score"], reverse=True)
    # also strict week-net rank among tr>=10
    strict=[r for r in top if r["week"]["tr"]>=10]
    strict.sort(key=lambda r: (r["week"]["net"], r["week"]["sh"]), reverse=True)

    # free big arrays
    del X
    return {
        "window":win,
        "ported_keltner":{"p":base_p,"sz":base_sz,"week":bw,"month":bm,"prior":bp},
        "n_bases":len(bases),"n_stops":len(STOPS),"n_week_pos":len(rows),
        "top_score":[{k:r[k] for k in ("fam","p","sz","week","month","prior","score")} for r in top[:15]],
        "top_week_net_tr10":[{k:r[k] for k in ("fam","p","sz","week","month","prior","score")} for r in strict[:10]],
        "elapsed_s":round(time.time()-t0,1),
    }

def write_md(res):
    ref_week=0.1864; ref_dd=-0.0393; ref_tr=73
    lines=["# Week-fit 5m / 3m vs 15m keltner short",
           f"Updated {pd.Timestamp.now(tz='Asia/Shanghai'):%Y-%m-%d %H:%M} Asia/Shanghai",
           "Window: same relative last-7d as WEEK_REGIME.md (BJ ~2026-10-27 23:45 .. 2026-11-03 23:45).",
           "Cost: taker 0.07%/side + funding. No liquidations allowed. REPORT ONLY — paper not changed.",
           "",
           "## Reference (current paper)",
           f"- 15m keltner short n20 k2 fade0: week **+{ref_week*100:.2f}%** DD {ref_dd*100:.2f}% {ref_tr}tr; "
           "30d +16.84% DD −15.81%; prior week +2.31%",
           ""]
    for tf in ("5m","3m"):
        if tf not in res: continue
        d=res[tf]
        lines.append(f"## {tf}")
        lines.append(f"- Window BJ: {d['window']['week_bj'][0]} .. {d['window']['week_bj'][1]}")
        lines.append(f"- Survivors week>0 (tr≥5, DD≥−30%, liq=0): {d['n_week_pos']}  elapsed {d['elapsed_s']}s")
        pk=d["ported_keltner"]
        lines.append(f"- Ported 15m keltner params: week {pk['week']['net']*100:+.2f}% DD {pk['week']['dd']*100:.2f}% "
                     f"{pk['week']['tr']}tr; 30d {pk['month']['net']*100:+.2f}%; prior {pk['prior']['net']*100:+.2f}%")
        best=None
        pool=d.get("top_week_net_tr10") or d.get("top_score") or []
        if pool: best=pool[0]
        if best:
            w,m,p=best["week"],best["month"],best["prior"]
            beat="YES" if w["net"]>ref_week else "NO"
            lines.append(f"- **Best (week net, tr≥10 preferred):** {best['fam']} {json.dumps(best['p'],ensure_ascii=False)}")
            lines.append(f"  size {best['sz']}: week **{w['net']*100:+.2f}%** DD {w['dd']*100:.2f}% {w['tr']}tr sh {w['sh']} | "
                         f"30d {m['net']*100:+.2f}% DD {m['dd']*100:.2f}% {m['tr']}tr | prior {p['net']*100:+.2f}% DD {p['dd']*100:.2f}% {p['tr']}tr")
            lines.append(f"  beats 15m week? **{beat}** (Δ {(w['net']-ref_week)*100:+.2f}pp)")
            if len(pool)>1:
                b2=pool[1]; w2=b2["week"]
                lines.append(f"- Runner-up: {b2['fam']} week {w2['net']*100:+.2f}% DD {w2['dd']*100:.2f}% {w2['tr']}tr size {b2['sz']}")
        else:
            lines.append("- No survivor passed gates.")
        lines.append("")
    # recommendation
    b5=(res.get("5m") or {}).get("top_week_net_tr10") or (res.get("5m") or {}).get("top_score") or []
    b3=(res.get("3m") or {}).get("top_week_net_tr10") or (res.get("3m") or {}).get("top_score") or []
    n5=b5[0]["week"]["net"] if b5 else -1
    n3=b3[0]["week"]["net"] if b3 else -1
    caveats=[]
    if b5:
        if b5[0]["month"]["net"]<=0: caveats.append("5m 30d≤0")
        if b5[0]["prior"]["net"]<=0: caveats.append("5m prior≤0")
    if b3:
        if b3[0]["month"]["net"]<=0: caveats.append("3m 30d≤0")
        if b3[0]["prior"]["net"]<=0: caveats.append("3m prior≤0")
    if n5>ref_week and n5>=n3:
        rec="switch 5m" if (b5 and b5[0]["month"]["net"]>0 and b5[0]["prior"]["net"]>-0.05) else "stay 15m"
        if rec=="switch 5m" and caveats: rec="stay 15m"
    elif n3>ref_week:
        rec="switch 3m" if (b3 and b3[0]["month"]["net"]>0 and b3[0]["prior"]["net"]>-0.05) else "stay 15m"
        if rec=="switch 3m" and caveats: rec="stay 15m"
    elif max(n5,n3)>0:
        rec="stay 15m"
    else:
        rec="none / stay 15m"
    # softer: never switch from this script; recommend stay unless clear beat + stable
    if max(n5,n3) <= ref_week:
        rec="stay 15m"
    elif (n5>ref_week and b5 and b5[0]["month"]["net"]>0 and b5[0]["prior"]["net"]>=0) or \
         (n3>ref_week and b3 and b3[0]["month"]["net"]>0 and b3[0]["prior"]["net"]>=0):
        # still prefer report-only; Mark did not ask to switch
        winner="5m" if n5>=n3 else "3m"
        rec=f"stay 15m (optional candidate {winner}; overfit risk — do not auto-switch)"
    else:
        rec="stay 15m"
    lines.append("## Verdict")
    lines.append(f"- Recommendation: **{rec}**")
    lines.append(f"- Caveats: {', '.join(caveats) if caveats else 'week-fit search is multiple-testing heavy; short TF fees eat edge.'}")
    lines.append("- Paper STATUS / week_fit_champ.json **not** modified.")
    open(MD,"w").write("\n".join(lines)+"\n")
    print("WROTE",MD, flush=True)
    return rec

def main():
    t0=time.time()
    # free memory: search one TF at a time
    res={}
    for tf in ("5m","3m"):
        try:
            res[tf]=search_tf(tf)
            # checkpoint
            json.dump(res, open(OUT,"w"), indent=1)
            print(f"checkpoint {tf} -> {OUT}", flush=True)
        except Exception as e:
            print("FAIL",tf,type(e).__name__,e, flush=True)
            res[tf]={"error":f"{type(e).__name__}: {e}"}
            json.dump(res, open(OUT,"w"), indent=1)
    rec=write_md(res)
    print("REC",rec, flush=True)
    print("DONE",round(time.time()-t0,1), flush=True)

if __name__=="__main__":
    main()
