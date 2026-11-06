#!/usr/bin/env python3
"""Bounded fee-aware random search on OKX USDT-SWAP data (freqtrade feather files). BACKTEST ONLY.
Costs: 0.05% taker + 0.02% slippage per side on every unit of turnover; funding: real OKX history where
available (only ~3 months), before that 0.01%/8h charged to whichever side is held (conservative).
Stops (sl/tp/ts) are evaluated on bar closes and exit at the next bar (approximation)."""
import os,json,time,random,glob
import numpy as np,pandas as pd
B='C:/Users/matth/okx-bt'; D=f'{B}/ud/data/okx/futures'; R=f'{B}/results'; os.makedirs(R,exist_ok=True)
COST=0.0007; START=pd.Timestamp('2024-01-01',tz='UTC'); SPLIT=pd.Timestamp('2025-07-01',tz='UTC')
FSPLIT=pd.Timestamp('2026-09-15',tz='UTC')   # funding-signal families: judged only on the real-funding window
N=int(os.environ.get('N_ITER','300')); LB_ONLY=bool(os.environ.get('LB_ONLY'))
GATE=dict(ho_sh=2.0,tr_sh=1.5,ho_tr=30,ho_net=0.10)
def load(tf):
    C={};H={};L={};F={};M={}
    for f in glob.glob(f'{D}/*-{tf}-futures.feather'):
        p=os.path.basename(f).split('-')[0]; df=pd.read_feather(f).set_index('date'); df=df[df.index>=START]
        if len(df)<500: continue
        C[p]=df.close;H[p]=df.high;L[p]=df.low
        fr=f'{D}/{p}-1h-funding_rate.feather'
        if os.path.exists(fr): F[p]=pd.read_feather(fr).set_index('date')['funding_rate']
        mk=f'{D}/{p}-1h-mark.feather'
        if os.path.exists(mk): M[p]=pd.read_feather(mk).set_index('date')['close']
    C=pd.DataFrame(C);idx=C.index
    Fr=pd.DataFrame(F).reindex(idx).reindex(columns=C.columns)
    is8=pd.Series((idx.hour%8==0)&(idx.minute==0),index=idx)
    miss=Fr.isna().mul(is8,axis=0)
    Mk=pd.DataFrame(M).reindex(columns=C.columns); Mk=Mk.reindex(idx) if tf=='1h' else Mk.resample(tf).last().reindex(idx)
    # last known funding (forward-filled, known at the bar) for signals
    Fk=pd.DataFrame(F).reindex(columns=C.columns).reindex(idx.union(pd.DataFrame(F).index)).ffill().reindex(idx)
    return dict(C=C,H=pd.DataFrame(H),L=pd.DataFrame(L),Fr=Fr.fillna(0),miss=miss.astype(float),Mk=Mk,Fk=Fk)
def ema(x,n): return x.ewm(span=n,adjust=False).mean()
def rsi(c,n):
    d=c.diff(); u=d.clip(lower=0).ewm(alpha=1/n).mean(); v=(-d.clip(upper=0)).ewm(alpha=1/n).mean(); return 100-100/(1+u/v)
def latch(c,ent_l,ent_s,flat):
    s=pd.Series(np.nan,index=c.index); s[flat]=0; s[ent_l]=1; s[ent_s]=-1; return s.ffill().fillna(0)
def raw_signal(c,h,l,fam,p):
    if fam=='ema_x': return np.sign(ema(c,p['f'])-ema(c,p['s']))
    if fam=='tsmom': return np.sign(c.pct_change(p['n']))
    if fam=='donchian':
        hi=h.rolling(p['n']).max().shift(1);lo=l.rolling(p['n']).min().shift(1);ex=max(2,p['n']//p['x'])
        ehi=h.rolling(ex).max().shift(1);elo=l.rolling(ex).min().shift(1)
        s=latch(c,c>hi,c<lo,pd.Series(False,index=c.index)).values.copy()
        cv=c.values;el=elo.values;eh=ehi.values;hv=hi.values;lv=lo.values;cur=0
        for i in range(len(cv)):
            if cv[i]>hv[i]: cur=1
            elif cv[i]<lv[i]: cur=-1
            elif cur==1 and cv[i]<el[i]: cur=0
            elif cur==-1 and cv[i]>eh[i]: cur=0
            s[i]=cur
        return pd.Series(s,index=c.index)
    if fam=='rsi_mr':
        r=rsi(c,p['n']); return latch(c,r<p['lo'],r>100-p['lo'],(r>50-p['b'])&(r<50+p['b']))
    if fam=='bb_mr':
        m=c.rolling(p['n']).mean(); z=(c-m)/c.rolling(p['n']).std(); return latch(c,z<-p['k'],z>p['k'],z.abs()<0.2)
    raise ValueError(fam)
def apply_stops(s,c,h,l,p):
    sl,tp,ts=p.get('sl',0),p.get('tp',0),p.get('ts',0)
    if not(sl or tp or ts): return s
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); atr=tr.rolling(14).mean().values
    sv=s.values;cv=c.values;out=np.zeros(len(sv));pos=0;ep=0;ea=0;age=0;blocked=0
    for i in range(len(sv)):
        sig=sv[i]
        if blocked and sig!=blocked: blocked=0
        if pos!=0:
            age+=1; mv=(cv[i]-ep)*pos
            hit=(sl and mv<=-sl*ea) or (tp and mv>=tp*ea) or (ts and age>=ts)
            if hit: blocked=pos; pos=0
            elif sig!=pos: pos=0
        if pos==0 and sig!=0 and sig!=blocked and not np.isnan(atr[i]):
            pos=int(sig);ep=cv[i];ea=atr[i];age=0
        out[i]=pos
    return pd.Series(out,index=s.index)
def weights(X,fam,p):
    C,H,L=X['C'],X['H'],X['L']; n=C.shape[1]
    if fam=='xsmom':
        sc=C.pct_change(p['n'])
        if p.get('skip'): sc=C.shift(p['skip']).pct_change(p['n'])
        if p['mode']=='rev': sc=-sc
    elif fam in('fund_carry','fund_contra'):
        sc=-X['Fk'] if fam=='fund_carry' else X['Fk']
    elif fam=='prem_mr':
        prem=C/X['Mk']-1; z=(prem-prem.rolling(p['n']).mean())/prem.rolling(p['n']).std(); sc=-z
    else: sc=None
    if sc is not None:
        k=p['k']; rk=sc.rank(axis=1,ascending=False); cnt=sc.notna().sum(axis=1)
        W=(rk<=k).astype(float)/ (k if p['dir']=='long' else 2*k)
        if p['dir']!='long': W=W-(rk>cnt.values[:,None]-k).astype(float)/(2*k)
        W=W.where(sc.notna(),0)
        reb=p.get('reb',1); m=np.arange(len(W))%reb==0; W=W.where(pd.Series(m,index=W.index),np.nan).ffill().fillna(0)
    else:
        cols={}
        for k_ in C.columns:
            c=C[k_].dropna(); h=H[k_].reindex(c.index); l=L[k_].reindex(c.index)
            s=pd.Series(raw_signal(c,h,l,fam,p),index=c.index).fillna(0)
            if p.get('dir')=='long': s=s.clip(lower=0)
            if p.get('dir')=='short': s=s.clip(upper=0)
            if p.get('trend'): s=s.where(np.sign(s)==np.sign(c-ema(c,p['trend'])),0)
            s=apply_stops(s,c,h,l,p)
            if p.get('vt'): s=s*(p['vt']/c.pct_change().rolling(100).std()).clip(upper=2).fillna(0)
            cols[k_]=s
        W=pd.DataFrame(cols).reindex(C.index).fillna(0)/n
    if p.get('btc'):  # BTC regime filter
        b=C['BTC_USDT_USDT']; reg=np.sign(b-ema(b,p['btc']))
        W=W.where(np.sign(W).eq(reg,axis=0)|(W==0),0)
    return W
def run(X,fam,p):
    W=weights(X,fam,p).shift(1).fillna(0)   # act next bar
    r=X['C'].pct_change().fillna(0)
    turn=W.diff().abs().fillna(0)
    net=(W*r).sum(axis=1)-COST*turn.sum(axis=1)-(W*X['Fr']).sum(axis=1)-(W.abs()*X['miss']*0.0001).sum(axis=1)
    ent=((W!=0)&(np.sign(W)!=np.sign(W.shift(1).fillna(0)))).sum(axis=1)
    return net.resample('1D').sum(),ent.resample('1D').sum()
def stats(d,e):
    if len(d)<20 or d.std()==0: return dict(sh=0,net=0,dd=0,pf=0,tr=int(e.sum()))
    eq=(1+d).cumprod(); g=d[d>0].sum(); b=-d[d<0].sum()
    return dict(sh=round(d.mean()/d.std()*np.sqrt(365),2),net=round(eq.iloc[-1]-1,3),dd=round((eq/eq.cummax()-1).min(),3),pf=round(g/b,2) if b>0 else 99,tr=int(e.sum()))
ch=random.choice
def stops(p):
    if random.random()<0.5: p.update(sl=ch([1.5,2,3]),tp=ch([0,2,3,5]),ts=ch([0,12,24,48]))
    p['btc']=ch([0,0,50,200]); return p
SP={'ema_x':lambda:stops(dict(f=ch([5,8,10,12,20,30]),s=ch([20,30,50,80,100,150,200]))),
 'donchian':lambda:stops(dict(n=ch([20,30,48,55,72,100,150]),x=ch([2,3,4]))),
 'tsmom':lambda:stops(dict(n=ch([12,24,48,72,120,168,240,336]))),
 'rsi_mr':lambda:stops(dict(n=ch([2,3,5,7,14]),lo=ch([10,15,20,25,30]),b=ch([0,5,10]))),
 'bb_mr':lambda:stops(dict(n=ch([20,30,50,100]),k=ch([1.5,2,2.5,3]))),
 'xsmom':lambda:dict(n=ch([6,12,24,48,72,120,168,336]),k=ch([2,3,4,5]),reb=ch([1,4,6,12,24,48]),mode=ch(['mom','rev']),skip=ch([0,0,1,6]),dir=ch(['both','long']),btc=ch([0,0,50,200])),
 'fund_carry':lambda:dict(k=ch([2,3,4,5]),reb=ch([8,24,48]),dir=ch(['both','long']),btc=0),
 'fund_contra':lambda:dict(k=ch([2,3,4,5]),reb=ch([8,24,48]),dir=ch(['both','long']),btc=0),
 'prem_mr':lambda:dict(n=ch([24,48,96,168]),k=ch([2,3,4,5]),reb=ch([1,4,8,24]),dir=ch(['both','long']),btc=ch([0,0,200]))}
WT={'xsmom':4,'fund_carry':1,'fund_contra':1,'prem_mr':2}
def passed(r): return r['hold']['sh']>=GATE['ho_sh'] and r['train']['sh']>=GATE['tr_sh'] and r['hold'].get('tr',0)>=GATE['ho_tr'] and r['hold']['net']>=GATE['ho_net']
def report(t_start,n_new):
    rows=[json.loads(l) for l in open(f'{R}/all_runs.jsonl')]
    df=pd.DataFrame([dict(fam=r['fam'],tf=r['tf'],params=json.dumps(r['p']),split=r.get('split','2025-07-01'),
        tr_sh=r['train']['sh'],tr_pf=r['train']['pf'],tr_dd=r['train']['dd'],tr_net=r['train']['net'],
        ho_sh=r['hold']['sh'],ho_pf=r['hold']['pf'],ho_dd=r['hold']['dd'],ho_net=r['hold']['net'],ho_tr=r['hold'].get('tr',-1),PASS=passed(r)) for r in rows])
    df['score']=df[['tr_sh','ho_sh']].min(axis=1)
    df=df.sort_values('score',ascending=False); df.to_csv(f'{R}/leaderboard.csv',index=False)
    npass=int(df.PASS.sum()); fams=df.fam.value_counts().to_dict()
    with open(f'{R}/leaderboard.md','w') as f:
        f.write(f'# OKX fee-aware search — {len(df)} configs tried (deflated-Sharpe caution!)\nRanked by min(train,holdout) Sharpe. Gate: {GATE}. Passers: {npass}\n\n')
        f.write(df.drop(columns='score').head(30).to_markdown(index=False))
    with open(f'{R}/STATUS.md','w') as f:
        f.write(f'# okx-bt search status\nUpdated: {pd.Timestamp.now(tz="Asia/Shanghai"):%Y-%m-%d %H:%M} (UTC+8)\n\nBACKTEST ONLY — no orders, no API keys.\n\n')
        f.write(f'- Configs tried total: {len(df)}; this run: {n_new} in {round((time.time()-t_start)/60)} min\n- By family: {fams}\n- Gate {GATE}: **{npass} passers**\n')
        f.write(f'- Best by min(train,holdout) Sharpe:\n\n{df.drop(columns="score").head(5).to_markdown(index=False)}\n\n')
        cx=f'{R}/CROSSCHECK.md'
        if os.path.exists(cx): f.write(open(cx).read()+'\n')
        f.write('Notes: funding families judged only on real-funding window (2026-07-27..), split 2026-09-15 — tiny sample.\nService: okx-bt-search.timer (stop: systemctl disable --now okx-bt-search.timer)\n')
    if npass: df[df.PASS].to_csv(f'{R}/PASSERS.csv',index=False)
def main():
    t0=time.time(); lb=f'{R}/all_runs.jsonl'; done=set()
    if os.path.exists(lb):
        for line in open(lb): done.add(json.loads(line)['key'])
    n=0
    if not LB_ONLY:
        cache={tf:load(tf) for tf in ['1h','4h']}; fams=[f for f in SP for _ in range(WT.get(f,1))]
        tries=0
        while n<N and tries<N*5:
            tries+=1; fam=ch(fams); tf=ch(['1h','4h']); p=SP[fam]()
            if fam not in WT: p['dir']=ch(['both','long','short']); p['trend']=ch([0,0,200,400]); p['vt']=ch([0,0.01])
            if fam=='ema_x' and p['f']>=p['s']: continue
            key=json.dumps([fam,tf,p],sort_keys=True)
            if key in done: continue
            done.add(key); X=cache[tf]
            d,e=run(X,fam,p)
            if fam.startswith('fund'):
                m=d.index>=pd.Timestamp('2026-07-28',tz='UTC'); d,e=d[m],e[m]; sp=FSPLIT
            else: sp=SPLIT
            r=dict(key=key,fam=fam,tf=tf,p=p,split=str(sp.date()),train=stats(d[d.index<sp],e[e.index<sp]),hold=stats(d[d.index>=sp],e[e.index>=sp]))
            with open(lb,'a') as f: f.write(json.dumps(r)+'\n')
            n+=1
            if n%25==0: report(t0,n)
    report(t0,n); print('done',n,'in',round(time.time()-t0),'s')
main()
