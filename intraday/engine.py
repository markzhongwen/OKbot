"""Intraday fee-aware backtest engine (BACKTEST ONLY). Per-pair single position, entry at next bar open after signal,
ATR SL/TP checked intrabar on high/low (SL assumed first if both hit), time stop, funding charged per settlement crossed.
Each pair gets 1/N of equity at 1x; daily P&L summed over pairs."""
import os,glob,json,numpy as np,pandas as pd
from numba import njit
B='C:/Users/matth/okx-bt'; D=f'{B}/ud/data/okx/futures'; FD=f'{B}/ud/data_proxy/okx/futures'; CACHE=f'{B}/intraday/cache'
TF_MIN={'1m':1,'3m':3,'5m':5,'15m':15}
def pairs(): return sorted(os.path.basename(f).split('-')[0] for f in glob.glob(f'{D}/*-1m-futures.feather'))
def load(tf):
    os.makedirs(CACHE,exist_ok=True); fn=f'{CACHE}/{tf}.npz'; src=f'{D}/BTC_USDT_USDT-1m-futures.feather'
    if os.path.exists(fn) and os.path.getmtime(fn)>os.path.getmtime(src):
        z=np.load(fn,allow_pickle=True); return {k:z[k] for k in z.files}
    P=pairs(); cols={k:{} for k in 'ohlcv'}
    for p in P:
        df=pd.read_feather(f'{D}/{p}-1m-futures.feather').set_index('date')
        if tf!='1m': df=df.resample(f'{TF_MIN[tf]}min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
        for k,c in zip('ohlcv',['open','high','low','close','volume']): cols[k][p]=df[c]
    out={k:pd.DataFrame(v).ffill().astype('float32') for k,v in cols.items()}
    idx=out['c'].index; fund={}
    for p in P:
        f=pd.read_feather(f'{FD}/{p}-1h-funding_rate.feather').set_index('date')['funding_rate']
        fund[p]=f.reindex(idx).fillna(0)          # rate applied on the bar starting at settlement time
    X=dict(o=out['o'].values,h=out['h'].values,l=out['l'].values,c=out['c'].values,v=out['v'].values,
           f=pd.DataFrame(fund).values.astype('float32'),t=idx.values.astype('datetime64[m]').astype('int64'),pairs=np.array(P))
    np.savez(fn,**X); return X
# ---------- indicators (column-wise numpy/pandas) ----------
def P(a): return pd.DataFrame(a)
def atr(X,n=14):
    h,l,c=P(X['h']),P(X['l']),P(X['c']); pc=c.shift()
    tr=np.maximum(h-l,np.maximum((h-pc).abs(),(l-pc).abs())); return tr.rolling(n).mean().values
def rsi(c,n):
    d=c.diff(); u=d.clip(lower=0).ewm(alpha=1/n).mean(); v=(-d.clip(upper=0)).ewm(alpha=1/n).mean(); return (100-100/(1+u/v))
def htf_trend(X,tfm,hmin,n):
    """sign(close - EMA_n) computed on a higher TF of hmin minutes, known only after that bar closed"""
    c=P(X['c']); t=pd.to_datetime(X['t'],unit='m'); c.index=t
    hc=c.resample(f'{hmin}min').last(); tr=np.sign(hc-hc.ewm(span=n,adjust=False).mean()).shift(1)
    return tr.reindex(t,method='ffill').values
def minute_of_day(X): return (X['t']%1440).astype(np.int64)
def signal(X,fam,p,tfm):
    c=P(X['c']); h=P(X['h']); l=P(X['l']); A=atr(X); n=p.get('n',20)
    if fam=='bb_mr':
        z=(c-c.rolling(n).mean())/c.rolling(n).std(); s=np.where(z<-p['k'],1,np.where(z>p['k'],-1,0))
    elif fam=='rsi_mr':
        r=rsi(c,n).values; s=np.where(r<p['lo'],1,np.where(r>100-p['lo'],-1,0))
    elif fam=='vwap_mr':
        v=P(X['v']); vw=(c*v).rolling(n).sum()/v.rolling(n).sum(); d=(c-vw)/A; s=np.where(d<-p['k'],1,np.where(d>p['k'],-1,0))
    elif fam=='squeeze':
        bw=c.rolling(n).std()/c.rolling(n).mean(); q=bw.rolling(p['m']).quantile(p['q']); sq=(bw<=q).shift(1).fillna(False)
        hi=h.rolling(n).max().shift(1); lo=l.rolling(n).min().shift(1); s=np.where(sq&(c>hi),1,np.where(sq&(c<lo),-1,0))
        if p.get('fade'): s=-s
    elif fam=='orb':
        mod=minute_of_day(X); H0=p['open_min']; R=p['range_min']; W=p['win_min']
        t=pd.to_datetime(X['t'],unit='m'); day=(X['t']-H0)//1440
        inr=((mod-H0)%1440<R); g=pd.Series(day)
        m=np.broadcast_to(inr[:,None],h.shape)
        rh=h.where(m).groupby(g.values).transform('max').values; rl=l.where(m).groupby(g.values).transform('min').values
        rh=pd.DataFrame(rh).ffill().values; rl=pd.DataFrame(rl).ffill().values
        act=(((mod-H0)%1440>=R)&((mod-H0)%1440<R+W))[:,None]
        cv=c.values
        up=(cv>rh)&act; dn=(cv<rl)&act; s=np.where(up,1,np.where(dn,-1,0))
        if p.get('fade'): s=-s
    elif fam=='trend_pb':
        tr=htf_trend(X,tfm,p['hmin'],p['hema']); r=rsi(c,p['n']).values
        s=np.where((tr>0)&(r<p['lo']),1,np.where((tr<0)&(r>100-p['lo']),-1,0))
    elif fam=='xs':
        L=p['L']; ret=c.pct_change(L); rk=ret.rank(axis=1); N=c.shape[1]; k=p['k']
        reb=(np.arange(len(c))%p['reb']==0)[:,None]
        bot=(rk<=k).values; top=(rk>N-k).values
        s=np.where(bot,1,np.where(top,-1,0))*(1 if p['mode']=='rev' else -1); s=np.where(reb,s,0)
    elif fam=='btc_lead':
        L=p['L']; r=c.pct_change(L); bi=list(X['pairs']).index('BTC_USDT_USDT'); br=r.iloc[:,bi]
        bz=br/br.rolling(500).std(); lag=r.sub(br,axis=0)
        s=np.where(((bz.values>p["z"])[:,None]&(lag.values<-p['g']*np.abs(br.values)[:,None])),1,
                   np.where(((bz.values<-p["z"])[:,None]&(lag.values>p['g']*np.abs(br.values)[:,None])),-1,0)); s[:,bi]=0
    elif fam=='supertrend':
        A2=pd.DataFrame(atr(X,p['n'])); mid=(h+l)/2; ub=(mid+p['k']*A2).values; lb=(mid-p['k']*A2).values; cv=c.values
        T,N=cv.shape; d=np.ones((T,N),np.int8); fu=ub.copy(); fl=lb.copy()
        for i in range(1,T):
            fu[i]=np.where(np.isnan(fu[i-1])|(ub[i]<fu[i-1])|(cv[i-1]>fu[i-1]),ub[i],fu[i-1]); fl[i]=np.where(np.isnan(fl[i-1])|(lb[i]>fl[i-1])|(cv[i-1]<fl[i-1]),lb[i],fl[i-1])
            d[i]=np.where(cv[i]>fu[i-1],1,np.where(cv[i]<fl[i-1],-1,d[i-1]))
        pd_=np.roll(d,1,axis=0); s=np.where((d==1)&(pd_==-1),1,np.where((d==-1)&(pd_==1),-1,0)); s[0]=0
    elif fam=='macd':
        m=c.ewm(span=p['f'],adjust=False).mean()-c.ewm(span=p['sl_'],adjust=False).mean(); sg=m.ewm(span=p['sg'],adjust=False).mean()
        x=np.sign(m-sg); px=x.shift(1); tr=np.sign(c-c.ewm(span=p['ema'],adjust=False).mean())
        s=np.where((x>0)&(px<=0)&(tr>0),1,np.where((x<0)&(px>=0)&(tr<0),-1,0))
    elif fam=='keltner':
        e=c.ewm(span=n,adjust=False).mean(); A2=pd.DataFrame(A); up=e+p['k']*A2; dn=e-p['k']*A2
        s=np.where((c>up)&(c.shift(1)<=up.shift(1)),1,np.where((c<dn)&(c.shift(1)>=dn.shift(1)),-1,0))
        if p.get('fade'): s=-s
    elif fam=='hma':
        def wma(x,m): w=np.arange(1,m+1); return x.rolling(m).apply(lambda a:(a*w).sum()/w.sum(),raw=True)
        hm=wma(2*wma(c,n//2)-wma(c,n),max(2,int(np.sqrt(n)))); sl_=np.sign(hm.diff()); ps=sl_.shift(1)
        s=np.where((sl_>0)&(ps<=0),1,np.where((sl_<0)&(ps>=0),-1,0))
    elif fam=='idmom':
        mod=minute_of_day(X); H0=p['open_min']; R=p['range_min']; day=(X['t']-H0)//1440
        k0=((mod-H0)%1440==0); k1=((mod-H0)%1440==R-tfm)
        o0=pd.DataFrame(np.where(k0[:,None],X['o'],np.nan)).groupby(day).transform('first').values
        r=c.values/o0-1; s=np.where(k1[:,None],np.sign(r)*(np.abs(r)>p['thr']/100),0)
        if p.get('fade'): s=-s
    elif fam=='fund_time':
        mod=minute_of_day(X); f=pd.DataFrame(np.where(X['f']!=0,X['f'],np.nan)).ffill().fillna(0).values
        at=((mod+p['pre'])%480==0); s=np.where(at[:,None]&(np.abs(f)>p['thr']*1e-4),-np.sign(f)*p['side'],0)
    elif fam=='vol_rev':
        v=P(X['v']); vz=v/v.rolling(n).mean(); body=(c-P(X['o']))/pd.DataFrame(A)
        s=np.where((vz>p['k'])&(body<-p['b']),1,np.where((vz>p['k'])&(body>p['b']),-1,0))
        if p.get('fade')==0: s=-s
    elif fam=='donchian':
        hi=h.rolling(n).max().shift(1); lo=l.rolling(n).min().shift(1)
        s=np.where(c>hi,1,np.where(c<lo,-1,0))
        if p.get('fade'): s=-s
    elif fam=='ext_rev':
        r=c.pct_change(p['L']); z=r/r.rolling(p['m']).std()
        s=np.where(z<-p['k'],1,np.where(z>p['k'],-1,0))
    elif fam=='liq_casc':
        L=p['L']; r=c.pct_change(L); dA=(c-c.shift(L))/pd.DataFrame(A); v=P(X['v']); vz=v/v.rolling(96).mean()
        brd=(r<-p['mv']/100).mean(axis=1).values[:,None]; brdu=(r>p['mv']/100).mean(axis=1).values[:,None]
        s=np.where((brd>=p['br'])&(dA<-p['k'])&(vz>p['vz']),1,np.where((brdu>=p['br'])&(dA>p['k'])&(vz>p['vz']),-1,0))
    elif fam=='avwap_gap':
        v=P(X['v']); day=(X['t']-p['anchor'])//1440; pv=(c*v).groupby(day).cumsum(); vv=v.groupby(day).cumsum()
        d=(c-pv/vv)/pd.DataFrame(A); s=np.where(d<-p['k'],1,np.where(d>p['k'],-1,0))
        if p.get('fade')==0: s=-s
    elif fam=='wkend':
        dow=((X['t']//1440)+3)%7; mod=minute_of_day(X)   # 0=Mon
        at=(dow==p['dow'])&(mod==p['at_min']); s=np.zeros(c.shape); s[at,:]=p['side']
    elif fam=='tod':
        mod=minute_of_day(X); s=np.zeros(c.shape); s[(mod==p['at_min']),:]=p['side']
    else: raise ValueError(fam)
    s=np.nan_to_num(np.asarray(s,dtype=np.float64))
    if p.get('htf'):  # optional higher-TF trend agreement filter
        tr=htf_trend(X,tfm,p['htf'],p.get('htf_ema',50)); s=np.where(np.sign(s)==tr,s,0)
    if fam not in('tod','fund_time','wkend') and p.get('dir')=='long': s=np.clip(s,0,1)
    if fam not in('tod','fund_time','wkend') and p.get('dir')=='short': s=np.clip(s,-1,0)
    return s.astype(np.int8),A
@njit(cache=True)
def sim(o,h,l,c,f,sig,A,sl,tp,ts,cool,mk=0,tick=1e-4):
    T,N=c.shape; maxtr=T*N//4+10
    ent_i=np.empty(maxtr,np.int64); ex_i=np.empty(maxtr,np.int64); side=np.empty(maxtr,np.int8); gross=np.empty(maxtr); fund=np.empty(maxtr); pr=np.empty(maxtr,np.int64)
    k=0
    for j in range(N):
        pos=0; ep=0.0; ea=0.0; ei=0; fsum=0.0; last=-10**9
        for i in range(1,T-1):
            if pos!=0:
                fsum+=pos*f[i,j]
                stop=ep-pos*sl*ea; take=ep+pos*tp*ea; xp=0.0; hit=False
                if pos==1:
                    if l[i,j]<=stop: xp=min(o[i,j],stop); hit=True
                    elif tp>0 and h[i,j]>=take: xp=max(o[i,j],take); hit=True
                else:
                    if h[i,j]>=stop: xp=max(o[i,j],stop); hit=True
                    elif tp>0 and l[i,j]<=take: xp=min(o[i,j],take); hit=True
                if not hit and i-ei>=ts: xp=c[i,j]; hit=True
                if hit:
                    ent_i[k]=ei; ex_i[k]=i; side[k]=pos; gross[k]=pos*(xp/ep-1); fund[k]=fsum; pr[k]=j; k+=1; pos=0; last=i
                    if k>=maxtr: break
                    continue
            if pos==0 and sig[i-1,j]!=0 and i-last>=cool and A[i-1,j]>0 and not np.isnan(A[i-1,j]):
                sg=sig[i-1,j]
                if mk==1:
                    lim=c[i-1,j]
                    if (sg==1 and l[i,j]<lim*(1-tick)) or (sg==-1 and h[i,j]>lim*(1+tick)):
                        pos=sg; ep=min(lim,o[i,j]) if sg==1 else max(lim,o[i,j]); ea=A[i-1,j]; ei=i; fsum=0.0
                else:
                    pos=sg; ep=o[i,j]; ea=A[i-1,j]; ei=i; fsum=0.0
    return ent_i[:k],ex_i[:k],side[:k],gross[:k],fund[:k],pr[:k]
def backtest(X,fam,p,tf,mk=0):
    tfm=TF_MIN[tf]; s,A=signal(X,fam,p,tfm)
    e,x,sd,g,fu,pr=sim(X['o'].astype(np.float64),X['h'].astype(np.float64),X['l'].astype(np.float64),X['c'].astype(np.float64),
                       X['f'].astype(np.float64),s,np.nan_to_num(A,nan=0.0),float(p['sl']),float(p.get('tp',0.0)),int(p['ts']),int(p.get('cool',1)),mk)
    return dict(ent=e,ex=x,side=sd,gross=g,fund=fu,t=X['t'],N=X['c'].shape[1])
def evaluate(r,cost_in,cost_out,start,split,end=None):
    t=r['t']; N=r['N']; net=r['gross']-cost_in-cost_out-r['fund']
    day=pd.to_datetime(t[r['ex']],unit='m').floor('D')
    d=pd.Series(net/N).groupby(day).sum()
    full=pd.date_range(pd.Timestamp(start),pd.to_datetime(t[-1],unit='m').floor('D'),freq='D'); d=d.reindex(full).fillna(0)
    ent=pd.to_datetime(t[r['ent']],unit='m')
    def st(m,tm):
        x=d[m]; 
        if len(x)<10 or x.std()==0: return dict(sh=0.0,net=0.0,dd=0.0,pf=0.0,tr=int(tm.sum()),pos_m=0,n_m=0)
        eq=(1+x).cumprod(); g_=net[tm.values]; pf=g_[g_>0].sum()/max(1e-12,-g_[g_<0].sum())
        mo=x.groupby(x.index.to_period('M')).sum()
        return dict(sh=round(float(x.mean()/x.std()*np.sqrt(365)),2),net=round(float(eq.iloc[-1]-1),3),dd=round(float((eq/eq.cummax()-1).min()),3),
                    pf=round(float(pf),2),tr=int(tm.sum()),pos_m=int((mo>0).sum()),n_m=int(len(mo)))
    sp=pd.Timestamp(split); s0=pd.Timestamp(start)
    em=pd.Series(ent>=s0)
    return dict(train=st((d.index>=s0)&(d.index<sp),em&pd.Series(ent<sp)),hold=st(d.index>=sp,pd.Series(ent>=sp)),all=st(d.index>=s0,em))
 
