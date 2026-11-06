#!/usr/bin/env python3
"""Week-regime fit for 15m paper. BACKTEST ONLY. Fee-aware. Signal-cached grid."""
import os,sys,json,math,itertools,time
import numpy as np,pandas as pd
sys.path.insert(0,"C:/Users/matth/okx-bt/intraday")
import engine as E
import evolve as Ev

R="C:/Users/matth/okx-bt/intraday"
TF="15m"; EQ0=25.0; COST=0.0007; MMR=0.005

def regime_note(X):
    t=pd.to_datetime(X["t"],unit="m")
    i=list(X["pairs"]).index("BTC_USDT_USDT")
    c=X["c"][:,i]; end=t[-1]; a=end-pd.Timedelta(days=7)
    m=t>=a; cc=c[m]
    btc=(cc[-1]/cc[0]-1)*100
    dvol=float(pd.Series(cc,index=t[m]).resample("1D").last().pct_change().dropna().std()*100)
    m2=(t>=end-pd.Timedelta(days=14))&(t<end-pd.Timedelta(days=7))
    dvol2=float(pd.Series(c[m2],index=t[m2]).resample("1D").last().pct_change().dropna().std()*100) if m2.sum()>100 else float("nan")
    alts=[]
    for j,p in enumerate(X["pairs"]):
        if "BTC" in str(p): continue
        vv=X["c"][m,j]
        if vv[0]==0 or np.isnan(vv[0]): continue
        alts.append((str(p).split("_")[0], float(vv[-1]/vv[0]-1)))
    alts.sort(key=lambda x:x[1])
    alt_med=float(np.median([x[1] for x in alts]))*100 if alts else 0
    h4=pd.Series(cc,index=t[m]).resample("4h").last().dropna()
    ema=h4.ewm(span=20).mean()
    trend="up" if h4.iloc[-1]>ema.iloc[-1] else "down"
    now=pd.Timestamp.now(tz="Asia/Shanghai")
    lines=[
        "# Week regime note",
        f"Updated {now:%Y-%m-%d %H:%M} Asia/Shanghai",
        f"Window (last 7d to latest candle): {(a+pd.Timedelta(hours=8)):%Y-%m-%d %H:%M} .. {(end+pd.Timedelta(hours=8)):%Y-%m-%d %H:%M} Asia/Shanghai",
        f"(UTC {a} .. {end})",
        f"- BTC week return: {btc:.2f}% | daily realized vol {dvol:.2f}% (prior week {dvol2:.2f}%)",
        f"- 4h vs EMA20 at end: {trend}",
        f"- Alt median week: {alt_med:.2f}% | worst {alts[:3]} | best {alts[-3:]}",
        "- Character: mild down / risk-off, vol up vs prior week, alts weaker than BTC.",
        "- Fit bias: short-biased MR / fade-long ORB / failed breakout shorts; not aggressive long-only ORB.",
    ]
    open(f"{R}/WEEK_REGIME.md","w").write("\n".join(lines)+"\n")
    return "\n".join(lines)

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

def signal_bases():
    bases=[]
    for open_min,range_min,win_min,fade,d,htf in itertools.product(
        [0,60,420,480,810,840,870],[30,60],[60,120],[0,1],["short","both","long"],[0,240]):
        bases.append(("orb", dict(open_min=open_min,range_min=range_min,win_min=win_min,fade=fade,dir=d,htf=htf,htf_ema=20)))
    for n,k,d in itertools.product([10,20,30],[1.5,2.0,2.5],["short","both","long"]):
        bases.append(("bb_mr", dict(n=n,k=k,dir=d,htf=0,htf_ema=20)))
    for n,lo,d in itertools.product([5,7,14],[15,20,25],["short","both"]):
        bases.append(("rsi_mr", dict(n=n,lo=lo,dir=d,htf=0,htf_ema=20)))
    for n,k,d in itertools.product([48,96],[2,3],["short","both"]):
        bases.append(("vwap_mr", dict(n=n,k=k,dir=d,htf=0,htf_ema=20)))
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

STOPS=[
    dict(sl=2,tp=0,ts=8,cool=3),
    dict(sl=2,tp=0,ts=12,cool=6),
    dict(sl=3,tp=0,ts=12,cool=6),
    dict(sl=2,tp=3,ts=12,cool=3),
    dict(sl=1.5,tp=2,ts=8,cool=3),
    dict(sl=3,tp=4,ts=16,cool=6),
]
SIZES=[
    dict(margin=0.10,lev=3,maxpos=5),
    dict(margin=0.10,lev=5,maxpos=3),
    dict(margin=0.15,lev=3,maxpos=3),
    dict(margin=0.08,lev=8,maxpos=3),
    dict(margin=0.12,lev=5,maxpos=2),
]

def main():
    t0=time.time()
    X=E.load(TF)
    print(regime_note(X), flush=True)
    end_m=int(X["t"][-1])
    w7_0,w7_1=end_m-7*1440, end_m+1
    w30_0=end_m-30*1440
    prior_0,prior_1=end_m-14*1440, end_m-7*1440
    t=X["t"]; o,h,l=X["o"],X["h"],X["l"]; tfm=E.TF_MIN[TF]

    baseline=dict(fam="orb",
        p={"open_min":840,"range_min":30,"win_min":60,"fade":1,"sl":3,"tp":0,"ts":12,"cool":6,"dir":"long","htf":240,"htf_ema":20},
        sz={"margin":0.1,"lev":3,"maxpos":5})
    def eval_full(fam,p,sz):
        s,A=E.signal(X,Ev.efam(fam),p,tfm)
        e,x,sd,g,fu,pr=E.sim(X["o"].astype(np.float64),X["h"].astype(np.float64),X["l"].astype(np.float64),
            X["c"].astype(np.float64),X["f"].astype(np.float64),s,np.nan_to_num(A,nan=0.0),
            float(p["sl"]),float(p.get("tp",0)),int(p["ts"]),int(p.get("cool",1)),0)
        return (size_from_trades(t,e,x,sd,g,fu,pr,A,o,h,l,p,sz,w7_0,w7_1),
                size_from_trades(t,e,x,sd,g,fu,pr,A,o,h,l,p,sz,w30_0,w7_1),
                size_from_trades(t,e,x,sd,g,fu,pr,A,o,h,l,p,sz,prior_0,prior_1))

    bw,bm,bp=eval_full(baseline["fam"],baseline["p"],baseline["sz"])
    print("BASELINE week",bw,"month",bm,"prior",bp, flush=True)

    bases=signal_bases()
    print(f"signal bases {len(bases)}; stop*size={len(STOPS)*len(SIZES)}; total cfg~{len(bases)*len(STOPS)*len(SIZES)}", flush=True)
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
                if w["tr"]<3 or w["liq"] or w["sl_beyond"] or w["net"]<=0 or w["dd"]<-0.30:
                    continue
                rows.append(dict(fam=fam,p=p,sz=sz,week=w,_pack=(e,x,sd,g,fu,pr,A)))
        if (bi+1)%25==0:
            print(f"  base {bi+1}/{len(bases)} survivors_so_far={len(rows)} elapsed={time.time()-t0:.0f}s", flush=True)

    rows.sort(key=lambda r: (r["week"]["net"], r["week"]["sh"]), reverse=True)
    print(f"week-positive survivors {len(rows)} in {time.time()-t0:.0f}s", flush=True)

    top=[]
    for r in rows[:100]:
        e,x,sd,g,fu,pr,A=r.pop("_pack")
        m=size_from_trades(t,e,x,sd,g,fu,pr,A,o,h,l,r["p"],r["sz"],w30_0,w7_1)
        pw=size_from_trades(t,e,x,sd,g,fu,pr,A,o,h,l,r["p"],r["sz"],prior_0,prior_1)
        r["month"]=m; r["prior"]=pw
        sc=r["week"]["net"]*100
        if m["net"]>0: sc+=5
        if m["dd"]>=-0.30: sc+=2
        if pw["net"]>0: sc+=3
        if r["week"]["tr"]>=5: sc+=2
        if r["week"]["tr"]>=8: sc+=2
        if r["week"]["dd"]>=-0.15: sc+=1
        sc += min(3.0, r["week"]["tr"]/3.0)
        r["score"]=round(sc,3)
        top.append(r)
    top.sort(key=lambda r: r["score"], reverse=True)

    out={"baseline":{"cfg":baseline,"week":bw,"month":bm,"prior":bp},
         "window":{"week_m":[w7_0,w7_1],"month_m":[w30_0,w7_1],"prior_m":[prior_0,prior_1],
                   "last_bar_utc":str(pd.to_datetime(X["t"][-1],unit="m")),
                   "last_bar_bj":str(pd.to_datetime(X["t"][-1],unit="m")+pd.Timedelta(hours=8))},
         "n_bases":len(bases),"n_week_pos":len(rows),
         "top":top[:30]}
    json.dump(out, open(f"{R}/week_fit_out.json","w"), indent=1)
    print("TOP8:", flush=True)
    for r in top[:8]:
        print(json.dumps({k:r[k] for k in ("fam","p","sz","week","month","prior","score")}), flush=True)
    print("DONE", round(time.time()-t0,1), flush=True)

if __name__=="__main__":
    main()
