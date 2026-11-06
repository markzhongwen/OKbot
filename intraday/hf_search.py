#!/usr/bin/env python3
"""Low-mem HF search 3m all + 1m top8. Modest lev≤5x margin≤10%. BACKTEST ONLY."""
import os,sys,json,math,random,time,gc
import numpy as np,pandas as pd
sys.path.insert(0,'C:/Users/matth/okx-bt/intraday'); import engine as E
OUT='C:/Users/matth/okx-bt/intraday/hf_runs.jsonl'; START='2026-05-03'; SPLIT='2026-09-01'
COST=0.0007; MMR=0.005; EQ0=25.0
TOP8=['BTC_USDT_USDT','ETH_USDT_USDT','SOL_USDT_USDT','XRP_USDT_USDT','DOGE_USDT_USDT','BNB_USDT_USDT','LINK_USDT_USDT','AVAX_USDT_USDT']
ch=random.choice
FAM={
 'ext_rev':lambda:dict(L=ch([1,2,3]),m=ch([300,500,1000]),k=ch([3.5,4,5,6])),
 'vol_rev':lambda:dict(n=ch([20,50,100]),k=ch([3,4,6,8]),b=ch([0.8,1,1.5,2]),fade=1),
 'avwap_gap':lambda:dict(anchor=ch([0,480,780,840]),k=ch([2,3,4]),fade=1),
 'vwap_mr':lambda:dict(n=ch([48,96,192]),k=ch([1.5,2,3])),
 'rsi_mr':lambda:dict(n=ch([2,3,5]),lo=ch([5,8,10,12])),
 'bb_mr':lambda:dict(n=ch([20,30,50]),k=ch([2,2.5,3])),
 'orb_asia':lambda:dict(open_min=ch([0,30,60]),range_min=ch([15,30,45]),win_min=ch([30,60,90]),fade=ch([0,1])),
 'orb_eu':lambda:dict(open_min=ch([420,450,480]),range_min=ch([15,30]),win_min=ch([30,60]),fade=ch([0,1])),
 'orb_us':lambda:dict(open_min=ch([810,840,870]),range_min=ch([15,30,60]),win_min=ch([30,60,120]),fade=ch([0,1])),
 'fund_time':lambda:dict(pre=ch([-30,-15,15,30]),thr=ch([0.5,1,2]),side=ch([1,-1])),
}
W=['orb_us']*3+['orb_asia']*2+['orb_eu']*2+['ext_rev']*3+['vol_rev']*3+['avwap_gap']*2+['vwap_mr']*2+['rsi_mr']*2+['bb_mr']*2+['fund_time']*3
EFAM={'orb_asia':'orb','orb_eu':'orb','orb_us':'orb'}
def efam(f): return EFAM.get(f,f)
STOPS=lambda:dict(sl=ch([0.75,1,1.5,2]),tp=ch([0,1,1.5,2,3]),ts=ch([2,3,4,6,8]),cool=ch([2,3,6,12]),
                  dir=ch(['both','long','short','long']),htf=ch([0,0,60,240]),htf_ema=ch([20,50]))
SIZES=lambda:dict(margin=ch([0.05,0.08,0.1]),lev=ch([3,5]),maxpos=ch([1,2,3,5]))
def load_sub(tf,pairs=None):
    X=E.load(tf)
    if pairs is None: return X
    idx=[list(X['pairs']).index(p) for p in pairs if p in list(X['pairs'])]
    out={k:(X[k][:,idx] if isinstance(X[k],np.ndarray) and X[k].ndim==2 else X[k]) for k in X}
    out['pairs']=np.array([list(X['pairs'])[i] for i in idx]); return out
def size_run(X,fam,p,sz,tf,mk=0):
    tfm=E.TF_MIN[tf]; s,A=E.signal(X,fam,p,tfm)
    e,x,sd,g,fu,pr=E.sim(*(X[k].astype(np.float64) for k in 'ohlcf'),s,np.nan_to_num(A),
        float(p['sl']),float(p.get('tp',0)),int(p['ts']),int(p.get('cool',1)),mk)
    t=X['t']; T0=int(pd.Timestamp(START).value//60_000_000_000); SPL=int(pd.Timestamp(SPLIT).value//60_000_000_000)
    keep=t[e]>=T0; e,x,sd,g,fu,pr=[np.asarray(v)[keep] for v in (e,x,sd,g,fu,pr)]
    if len(e)==0: return None
    A=np.nan_to_num(A); o,h,l=X['o'],X['h'],X['l']; CI=0.0002 if mk else COST
    sld=np.array([p['sl']*A[max(e[i]-1,0),pr[i]]/max(o[e[i],pr[i]],1e-12) for i in range(len(e))])
    mae=np.array([(o[e[i],pr[i]]-l[e[i]:x[i]+1,pr[i]].min())/o[e[i],pr[i]] if sd[i]>0
                  else (h[e[i]:x[i]+1,pr[i]].max()-o[e[i],pr[i]])/o[e[i],pr[i]] for i in range(len(e))])
    margin,lev,maxpos=float(sz['margin']),int(sz['lev']),int(sz['maxpos']); liqd=1/lev-MMR
    def run(mask):
        idx=np.where(mask)[0]; idx=idx[np.argsort(e[idx],kind='stable')]
        eq=EQ0; opn=[]; curve=[]; liq=slb=0; peak=EQ0; mdd=0.0
        for i in idx:
            for j in [k for k in opn if k[0]<=e[i]]:
                eq+=j[1]; opn.remove(j); peak=max(peak,eq); mdd=min(mdd,eq/peak-1 if peak>0 else 0); curve.append((t[j[0]],eq))
            if len(opn)>=maxpos or eq<=0.5: continue
            if sld[i]>=liqd: slb+=1; continue
            notional=eq*margin*lev
            if mae[i]>=liqd: pnl=-eq*margin-COST*notional; liq+=1
            else: pnl=notional*(g[i]-CI-COST-fu[i])
            opn.append((x[i],pnl))
        for j in sorted(opn,key=lambda z:z[0]):
            eq+=j[1]; peak=max(peak,eq); mdd=min(mdd,eq/peak-1); curve.append((t[j[0]],eq))
        if not curve: return dict(final=EQ0,net=0,dd=0,sh=0,tr=0,liq=0,sl_beyond=0,worst=0,mo={})
        ser=pd.Series([v for _,v in curve],index=pd.to_datetime([k for k,_ in curve],unit='m')).groupby(level=0).last()
        d=ser.resample('D').last().ffill(); d=pd.concat([pd.Series([EQ0],[d.index[0]-pd.Timedelta('1D')]),d]); rt=d.pct_change().dropna()
        sh=float(rt.mean()/rt.std()*math.sqrt(365)) if len(rt)>5 and rt.std()>0 else 0.0
        mo=d.resample('ME').last(); mo=(mo/mo.shift(1).fillna(EQ0)-1).round(3)
        return dict(final=round(eq,2),net=round(eq/EQ0-1,4),dd=round(mdd,4),sh=round(sh,3),tr=len(curve),
                    liq=int(liq),sl_beyond=int(slb),worst=round(float(rt.min()) if len(rt) else 0,4),
                    mo={str(k)[:7]:float(v) for k,v in mo.items()})
    tr=t[e]<SPL; all_=run(np.ones(len(e),bool)); train=run(tr); hold=run(~tr)
    return dict(all=all_,train=train,hold=hold,tr_sh=train['sh'],ho_sh=hold['sh'],tr_tr=train['tr'],ho_tr=hold['tr'],
                dd=all_['dd'],net=all_['net'],liq=all_['liq'],sl_beyond=all_['sl_beyond'],final=all_['final'],worst=all_['worst'],mo=all_['mo'])
def pass_sz(s,ho=1.5):
    return s and s['liq']==0 and s['sl_beyond']==0 and s['dd']>=-0.30 and s['ho_tr']>=100 and s['tr_tr']>=200 and s['ho_sh']>=ho and s['tr_sh']>=1.5
def main():
    N=int(os.environ.get('GEN','120')); tf=os.environ.get('TF','3m')
    pairs=TOP8 if tf=='1m' else None
    print('load',tf,pairs and len(pairs)); X=load_sub(tf,pairs); print('shape',X['c'].shape)
    # seed known-ish ORB variants
    seeds=[]
    for open_min,fade,sl,tp,ts,cool,m,L,mp in [
        (840,1,2,0,8,3,0.1,3,5),(840,1,2,4,8,3,0.1,3,5),(840,1,1.5,2,6,3,0.08,5,3),
        (810,1,2,0,6,2,0.1,3,5),(870,1,2,0,8,3,0.1,5,3),(0,1,2,0,8,3,0.1,3,5),
        (420,1,2,0,6,3,0.1,3,5),(480,1,1.5,2,4,2,0.08,5,5)]:
        fam='orb_us' if open_min>=780 else ('orb_eu' if open_min>=400 else 'orb_asia')
        seeds.append((fam,dict(open_min=open_min,range_min=30,win_min=60,fade=fade,sl=sl,tp=tp,ts=ts,cool=cool,dir='long',htf=240,htf_ema=50),dict(margin=m,lev=L,maxpos=mp)))
    done=set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: done.add(json.loads(l)['key'])
            except: pass
    n=0; hits=[]
    queue=list(seeds)+[None]*N
    for item in queue:
        if item is None:
            fam=ch(W); p=FAM[fam](); p.update(STOPS()); sz=SIZES()
        else: fam,p,sz=item
        if sz['margin']*sz['lev']>1.0: sz['margin']=round(1.0/sz['lev'],3)
        k=json.dumps([fam,tf,p,sz,bool(pairs)],sort_keys=True)
        if k in done: continue
        done.add(k)
        try:
            bt=E.backtest(X,efam(fam),p,tf)
            unit=E.evaluate(bt,COST,COST,START,SPLIT)
            r=dict(key=k,fam=fam,tf=tf,p=p,sz=sz,pairs=len(X['pairs']),taker=unit,ts=time.time())
            # size if enough trades and not catastrophic unit sharpe
            if unit['hold']['tr']>=80 and unit['train']['tr']>=150 and unit['hold']['sh']>-5:
                r['sized']=size_run(X,efam(fam),p,sz,tf,0)
                try: r['sized_mk']=size_run(X,efam(fam),p,sz,tf,1)
                except: r['sized_mk']=None
            else: r['sized']=None; r['sized_mk']=None
        except Exception as e:
            print('err',fam,e); continue
        open(OUT,'a').write(json.dumps(r)+'\n'); n+=1
        s=r.get('sized')
        if s and pass_sz(s,1.5):
            hits.append(r); print('HIT',fam,tf,sz,{k:s[k] for k in ['net','dd','ho_sh','tr_sh','tr_tr','ho_tr']})
        if n%25==0: print('prog',n,'hits',len(hits)); gc.collect()
        if n>=N+len(seeds): break
    rows=[]
    for l in open(OUT):
        try:
            r=json.loads(l)
            if r.get('sized') and r['tf']==tf and r['sized']['liq']==0 and r['sized']['sl_beyond']==0:
                rows.append(r)
        except: pass
    rows=sorted(rows,key=lambda r:-(r['sized']['ho_sh']+0.25*min(r['sized']['tr_sh'],3)))[:12]
    print('===TOP',tf,'===')
    for r in rows:
        s=r['sized']; mk=r.get('sized_mk') or {}
        print(json.dumps(dict(fam=r['fam'],tf=r['tf'],sz=r['sz'],p=r['p'],
            taker={k:s[k] for k in ['final','net','dd','ho_sh','tr_sh','tr_tr','ho_tr','liq','worst']},
            maker={k:mk.get(k) for k in ['net','dd','ho_sh','tr_sh']},mo=s.get('mo')),ensure_ascii=False))
    print('gen done',tf,n,'hits',len(hits),'sized_ok',len(rows))
if __name__=='__main__': main()
