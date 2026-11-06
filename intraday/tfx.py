import sys,json,math,gc;sys.path.insert(0,'C:/Users/matth/okx-bt/intraday')
import numpy as np,pandas as pd,engine as E
START='2026-05-03';SPLIT='2026-09-01';C=0.0007;EQ0=25.0
def atr15(X):
    t=X['t'];g=t//15;idx=pd.Index(g)
    H=pd.DataFrame(X['h']).groupby(g).max();L=pd.DataFrame(X['l']).groupby(g).min();Cc=pd.DataFrame(X['c']).groupby(g).last()
    pc=Cc.shift();tr=np.maximum(H-L,np.maximum((H-pc).abs(),(L-pc).abs()));a=tr.rolling(14).mean().shift(1) # known after prior 15m bar closed
    return a.reindex(g).values
def trades(X,p,tfm,mode):
    s,A=E.signal(X,'orb',p,tfm)
    if mode=='atr15': A=atr15(X)
    A=np.nan_to_num(A)
    e,x,sd,g,fu,pr=E.sim(*(X[k].astype(np.float64) for k in 'ohlcf'),s,A,float(p['sl']),0.0,int(p['ts']),int(p['cool']),0)
    em,xm,sdm,gm,fum,prm=E.sim(*(X[k].astype(np.float64) for k in 'ohlcf'),s,A,float(p['sl']),0.0,int(p['ts']),int(p['cool']),1)
    o,h,l=X['o'],X['h'],X['l']
    def extra(e,x,sd,pr):
        sld=np.array([p['sl']*A[e[i]-1,pr[i]]/o[e[i],pr[i]] for i in range(len(e))])
        mae=np.array([(o[e[i],pr[i]]-l[e[i]:x[i]+1,pr[i]].min())/o[e[i],pr[i]] if sd[i]>0 else (h[e[i]:x[i]+1,pr[i]].max()-o[e[i],pr[i]])/o[e[i],pr[i]] for i in range(len(e))])
        return sld,mae
    T=X['t']
    a=(T[e],T[x],g,fu,*extra(e,x,sd,pr)); b=(T[em],T[xm],gm,fum,*extra(em,xm,sdm,prm))
    return a,b
def size(tr,m,L,mp,mk):
    te,tx,g,fu,sld,mae=tr;T0=int(pd.Timestamp(START).value//6e10);SPL=int(pd.Timestamp(SPLIT).value//6e10)
    k=te>=T0;te,tx,g,fu,sld,mae=[v[k] for v in (te,tx,g,fu,sld,mae)];CI=0.0002 if mk else C;ld=1/L-0.005
    def run(mask):
        idx=np.where(mask)[0];idx=idx[np.argsort(te[idx],kind='stable')];eq=EQ0;opn=[];cur=[];liq=slb=0;pk=EQ0;dd=0
        for i in idx:
            for j in [z for z in opn if z[0]<=te[i]]:eq+=j[1];opn.remove(j);pk=max(pk,eq);dd=min(dd,eq/pk-1);cur.append((j[0],eq))
            if len(opn)>=mp or eq<=0.5:continue
            if sld[i]>=ld:slb+=1;continue
            no=eq*m*L
            pnl=-eq*m-C*no if mae[i]>=ld else no*(g[i]-CI-C-fu[i])
            if mae[i]>=ld:liq+=1
            opn.append((tx[i],pnl))
        for j in sorted(opn):eq+=j[1];pk=max(pk,eq);dd=min(dd,eq/pk-1);cur.append((j[0],eq))
        if not cur:return dict(net=0,dd=0,sh=0,tr=0,liq=0,slb=0,mo={})
        s=pd.Series([v for _,v in cur],index=pd.to_datetime([k for k,_ in cur],unit='m')).groupby(level=0).last()
        d=s.resample('D').last().ffill();d=pd.concat([pd.Series([EQ0],[d.index[0]-pd.Timedelta('1D')]),d]);rt=d.pct_change().dropna()
        mo=d.resample('ME').last();mo=(mo/mo.shift(1).fillna(EQ0)-1).round(3)
        return dict(final=round(eq,2),net=round(eq/EQ0-1,3),dd=round(dd,3),sh=round(float(rt.mean()/rt.std()*math.sqrt(365)),2) if rt.std()>0 else 0,tr=len(cur),liq=liq,slb=slb,worst=round(float(rt.min()),3),mo={str(k)[:7]:float(v) for k,v in mo.items()})
    a=run(np.ones(len(te),bool));t=run(te<SPL);h=run(te>=SPL)
    return dict(final=a.get('final'),net=a['net'],dd=a['dd'],tr_sh=t['sh'],ho_sh=h['sh'],tr_n=t['tr'],ho_n=h['tr'],liq=a['liq'],slb=a['slb'],worst=a.get('worst'),mo=a['mo'])
base=dict(open_min=840,range_min=30,win_min=60,fade=1,sl=2,tp=0,dir='long',htf=240,htf_ema=50)
SZ=[(0.08,8,5),(0.1,3,5)]
tf=sys.argv[1];tfm=E.TF_MIN[tf]
Xall=E.load(tf);N=Xall['c'].shape[1];print('loaded',tf,Xall['c'].shape,flush=True)
cases=[]
if tf=='15m': cases.append(('baseline',dict(base,ts=8,cool=3),'native'))
else:
    k=15//tfm
    for mode in ('native','atr15'):
        cases.append((f'a_wall_{mode}',dict(base,ts=120//tfm,cool=45//tfm),mode))
    cases.append(('b_bars',dict(base,range_min=2*tfm,win_min=4*tfm,ts=8,cool=3),'native'))
    for mode in ('native','atr15'):
        for sl in (1.5,2,3):
            for tsm in (60,120,180):
                if sl==2 and tsm==120: continue
                cases.append((f'grid_{mode}_sl{sl}_ts{tsm}',dict(base,sl=sl,ts=tsm//tfm,cool=45//tfm),mode))
CH=5 if tf=='1m' else N
out=[]
for name,p,mode in cases:
    acc=[[] for _ in range(6)];accm=[[] for _ in range(6)]
    for c0 in range(0,N,CH):
        X={k:(v[:,c0:c0+CH] if isinstance(v,np.ndarray) and v.ndim==2 else v) for k,v in Xall.items()};X['pairs']=Xall['pairs'][c0:c0+CH]
        a,b=trades(X,p,tfm,mode)
        for i in range(6):acc[i].append(a[i]);accm[i].append(b[i])
        del X;gc.collect()
    tr=[np.concatenate(v) for v in acc];trm=[np.concatenate(v) for v in accm]
    for m,L,mp in SZ:
        r=dict(tf=tf,case=name,sz=f'{int(m*100)}%x{L}x max{mp}',taker=size(tr,m,L,mp,0),maker=size(trm,m,L,mp,1))
        out.append(r);print(json.dumps(r),flush=True)
json.dump(out,open(f'C:/Users/matth/okx-bt/intraday/tfx_{tf}.json','w'))
print('done',tf,flush=True)
