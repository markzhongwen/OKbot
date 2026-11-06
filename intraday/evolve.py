#!/usr/bin/env python3
"""Evolving intraday search: strategy + sizing jointly. BACKTEST ONLY. No orders."""
import math,os,sys,json,random,time,subprocess,urllib.request,urllib.parse
import numpy as np,pandas as pd
sys.path.insert(0,'C:/Users/matth/okx-bt/intraday'); import engine as E
R='C:/Users/matth/okx-bt/intraday'; LOG=f'{R}/runs.jsonl'; CJ=f'{R}/champion.json'
START='2026-05-03'; SPLIT='2026-09-01'
TAKER=(0.0007,0.0007); MAKER=(0.0002,0.0007); COST=0.0007; MMR=0.005; EQ0=25.0
GATE=dict(ho_sh=2.0,tr_sh=1.5,ho_tr=100,tr_tr=150,dd=-0.30,pos_m_frac=0.5)
GEN=int(os.environ.get('GEN_SIZE','80')); ch=random.choice
STOPS=lambda:dict(sl=ch([0.75,1,1.5,2,3]),tp=ch([0,1,1.5,2,3,4]),ts=ch([4,8,12,24,48,96]),cool=ch([1,3,6]),
                  dir=ch(['both','both','long','short']),htf=ch([0,0,60,240]),htf_ema=ch([20,50,200]))
SIZES=lambda:dict(margin=ch([0.05,0.1,0.15,0.25,0.5]),lev=ch([3,5,8,10,15,20]),maxpos=ch([1,2,3,5]))
FAM={
 'bb_mr':lambda:dict(n=ch([10,20,30,50]),k=ch([1.5,2,2.5,3])),
 'rsi_mr':lambda:dict(n=ch([2,3,5,7,14]),lo=ch([5,10,15,20,25])),
 'vwap_mr':lambda:dict(n=ch([20,48,96,288]),k=ch([1,1.5,2,3])),
 'squeeze':lambda:dict(n=ch([10,20,30]),m=ch([200,500,1000]),q=ch([0.05,0.1,0.2])),
 'squeeze_fade':lambda:dict(n=ch([10,20,30]),m=ch([200,500,1000]),q=ch([0.05,0.1,0.2]),fade=1),
 'orb':lambda:dict(open_min=ch([0,60,420,480,810,840,870]),range_min=ch([15,30,60]),win_min=ch([60,120,240]),fade=ch([0,1])),
 'orb_asia':lambda:dict(open_min=ch([0,30,60]),range_min=ch([15,30,60]),win_min=ch([60,120]),fade=ch([0,1])),
 'orb_eu':lambda:dict(open_min=ch([420,450,480]),range_min=ch([15,30,60]),win_min=ch([60,120]),fade=ch([0,1])),
 'orb_us':lambda:dict(open_min=ch([810,840,870]),range_min=ch([15,30,60]),win_min=ch([60,120,240]),fade=ch([0,1])),
 'trend_pb':lambda:dict(n=ch([2,3,5,7,14]),lo=ch([10,20,30,40]),hmin=ch([60,240]),hema=ch([20,50,200])),
 'xs':lambda:dict(L=ch([1,2,3,6,12]),k=ch([1,2,3]),reb=ch([1,2,4,8]),mode=ch(['rev','mom'])),
 'btc_lead':lambda:dict(L=ch([1,2,3,5]),z=ch([1.5,2,2.5,3]),g=ch([0.3,0.5,0.8])),
 'supertrend':lambda:dict(n=ch([7,10,14]),k=ch([2,3,4])),
 'macd':lambda:dict(f=ch([6,12]),sl_=ch([26,35]),sg=ch([5,9]),ema=ch([50,100,200])),
 'keltner':lambda:dict(n=ch([20,50]),k=ch([1.5,2,2.5,3]),fade=ch([0,1])),
 'hma':lambda:dict(n=ch([16,25,49,81])),
 'idmom':lambda:dict(open_min=ch([0,420,780,810,840,960]),range_min=ch([30,60,120,240,360]),thr=ch([0,0.3,0.6,1.0]),fade=ch([0,1])),
 'fund_time':lambda:dict(pre=ch([-60,-30,-15,15,30,60,120]),thr=ch([0,0.5,1,2]),side=ch([1,-1])),
 'vol_rev':lambda:dict(n=ch([20,50,100]),k=ch([3,4,6]),b=ch([0.5,1,1.5]),fade=ch([0,1])),
 'donchian':lambda:dict(n=ch([20,55,96]),fade=ch([0,1])),
 'ext_rev':lambda:dict(L=ch([1,2,3]),m=ch([200,500,1000]),k=ch([3,4,5,6])),
 'liq_casc':lambda:dict(L=ch([1,2,4]),mv=ch([0.5,1,2]),br=ch([0.5,0.7,0.9]),k=ch([1.5,2,3]),vz=ch([1.5,2,3])),
 'avwap_gap':lambda:dict(anchor=ch([0,480,780,840]),k=ch([1.5,2,3,4]),fade=ch([0,1])),
 'wkend':lambda:dict(dow=ch([4,5,6,0]),at_min=ch(list(range(0,1440,120))),side=ch([1,-1])),
 'tod':lambda:dict(at_min=ch(list(range(0,1440,30))),side=ch([1,-1])),
}
EFAM={'orb_asia':'orb','orb_eu':'orb','orb_us':'orb','squeeze_fade':'squeeze'}
TFW=['15m']*4+['5m']*4+['3m']*2
def efam(f): return EFAM.get(f,f)
def key(fam,tf,p,sz): return json.dumps([fam,tf,p,sz],sort_keys=True)
def rand_cfg(tf):
    fam=ch(os.environ['FAMS'].split(',') if os.environ.get('FAMS') else list(FAM)); p=FAM[fam](); p.update(STOPS()); sz=SIZES()
    if sz['margin']*sz['lev']>20: sz['margin']=round(20/sz['lev'],3)
    return fam,p,sz
def mutate(fam,p,sz):
    q=dict(p); s=dict(sz); base=FAM[fam](); base.update(STOPS()); bz=SIZES()
    for k in random.sample(list(base),k=min(len(base),ch([1,2,3]))): q[k]=base[k]
    if random.random()<0.55:
        for k in random.sample(list(bz),k=ch([1,2])): s[k]=bz[k]
    if s['margin']*s['lev']>20: s['margin']=round(20/s['lev'],3)
    return q,s
def notify(msg,title='BT策略找到候选'):
    url=None
    for line in open('/etc/okx-alert.env'):
        if line.startswith('NTFY_URL='): url=line.split('=',1)[1].strip().strip('"')
    if not url: return
    try: urllib.request.urlopen(urllib.request.Request(url+'?'+urllib.parse.urlencode({'title':title}),data=msg.encode(),method='POST'),timeout=15)
    except Exception as e: print('ntfy fail',e)

def size_eval(X,fam,p,sz,tf,start=START,split=SPLIT,mk=0):
    """Trade-level compounding from 25U; fees on notional; concurrent maxpos; isolated liq if MAE>=1/lev-MMR."""
    tfm=E.TF_MIN[tf]; s,A=E.signal(X,efam(fam),p,tfm)
    e,x,sd,g,fu,pr=E.sim(X['o'].astype(np.float64),X['h'].astype(np.float64),X['l'].astype(np.float64),
        X['c'].astype(np.float64),X['f'].astype(np.float64),s,np.nan_to_num(A,nan=0.0),
        float(p['sl']),float(p.get('tp',0)),int(p['ts']),int(p.get('cool',1)),mk)
    t=X['t']; CI=0.0002 if mk else COST; T0=int(pd.Timestamp(start).value//60_000_000_000); SPL=int(pd.Timestamp(split).value//60_000_000_000)
    keep=t[e]>=T0
    e,x,sd,g,fu,pr=[np.asarray(v)[keep] for v in (e,x,sd,g,fu,pr)]
    if len(e)==0: return None
    A=np.nan_to_num(A); o,h,l=X['o'],X['h'],X['l']
    sl_d=np.array([p['sl']*A[max(e[i]-1,0),pr[i]]/max(o[e[i],pr[i]],1e-12) for i in range(len(e))])
    mae=np.array([(o[e[i],pr[i]]-l[e[i]:x[i]+1,pr[i]].min())/o[e[i],pr[i]] if sd[i]>0
                  else (h[e[i]:x[i]+1,pr[i]].max()-o[e[i],pr[i]])/o[e[i],pr[i]] for i in range(len(e))])
    margin,lev,maxpos=float(sz['margin']),int(sz['lev']),int(sz['maxpos']); liqd=1/lev-MMR
    def run(mask):
        idx=np.where(mask)[0]; idx=idx[np.argsort(e[idx],kind='stable')]
        eq=EQ0; opn=[]; curve=[]; liq=0; slb=0; peak=EQ0; mdd=0.0
        for i in idx:
            for j in [k for k in opn if k[0]<=e[i]]:
                eq+=j[1]; opn.remove(j); peak=max(peak,eq); mdd=min(mdd,eq/peak-1 if peak>0 else 0); curve.append((t[j[0]],eq))
            if len(opn)>=maxpos or eq<=0.5: continue
            if sl_d[i]>=liqd: slb+=1; continue   # skip: SL beyond isolated liq distance
            notional=eq*margin*lev
            if mae[i]>=liqd: pnl=-eq*margin-COST*notional; liq+=1
            else: pnl=notional*(g[i]-CI-COST-fu[i])
            opn.append((x[i],pnl))
        for j in sorted(opn, key=lambda z:z[0]):
            eq+=j[1]; peak=max(peak,eq); mdd=min(mdd,eq/peak-1 if peak>0 else 0); curve.append((t[j[0]],eq))
        if not curve: return dict(final=EQ0,net=0,dd=0,sh=0,tr=0,liq=0,sl_beyond=0,worst_day=0)
        ser=pd.Series([v for _,v in curve],index=pd.to_datetime([k for k,_ in curve],unit='m')).groupby(level=0).last()
        d=ser.resample('D').last().ffill(); rt=d.pct_change().dropna()
        sh=float(rt.mean()/rt.std()*math.sqrt(365)) if len(rt)>5 and rt.std()>0 else 0.0
        return dict(final=float(round(eq,2)),net=float(round(eq/EQ0-1,4)),dd=float(round(mdd,4)),sh=float(round(sh,3)),tr=int(len(curve)),
                    liq=int(liq),sl_beyond=int(slb),worst_day=float(round(float(rt.min()) if len(rt) else 0,4)))
    tr=t[e]<SPL; all_=run(np.ones(len(e),bool)); train=run(tr); hold=run(~tr)
    return dict(all=all_,train=train,hold=hold,tr_sh=train['sh'],ho_sh=hold['sh'],tr_tr=train['tr'],ho_tr=hold['tr'],
                dd=all_['dd'],net=all_['net'],liq=all_['liq'],sl_beyond=all_['sl_beyond'],worst_day=all_['worst_day'],
                final=all_['final'])

def rscore(r):
    sz=r.get('sized')
    if not sz: return -99
    tt,ht=sz['tr_tr'],sz['ho_tr']
    sc=min(sz['tr_sh'],sz['ho_sh'])*min(1.0,math.sqrt(min(max(tt,1)/150,max(ht,1)/100)))
    sc+=0.3*max(-1.0,min(2.0,sz.get('hold',{}).get('net',sz['net'])))
    if tt<150 or ht<100: sc-=10
    if sz['dd']<-0.30 or sz['liq'] or sz['sl_beyond']: sc-=5
    elif sz['dd']<-0.15: sc-=0.5
    return round(sc,3)

def sized_pass(sz,dd_lim=-0.30):
    return (sz and sz['liq']==0 and sz['sl_beyond']==0 and sz['dd']>=dd_lim
            and sz['ho_tr']>=100 and sz['tr_tr']>=150 and sz['ho_sh']>=2.0 and sz['tr_sh']>=1.5)

def fast_pass(sz):
    return (sz and sz['liq']==0 and sz['sl_beyond']==0 and sz['dd']>=-0.30 and sz['ho_tr']>=100 and sz['tr_tr']>=150
            and sz['ho_sh']>=1.5 and sz['tr_sh']>=1.5 and sz['net']>0)
def ft_sh(ft):
    try: return float(str(ft).split('Sharpe')[-1].strip())
    except Exception: return -9
def posm(r):
    a=r.get('taker',{}).get('all',{}); return a.get('pos_m',0)/max(a.get('n_m',1),1)
def soft_pass(r):
    t,h,a=r['taker']['train'],r['taker']['hold'],r['taker']['all']
    return h['tr']>=GATE['ho_tr'] and t['tr']>=GATE['tr_tr'] and a['n_m']>0 and a['pos_m']/max(a['n_m'],1)>=GATE['pos_m_frac']

def crosscheck(fam,p,tf):
    cfg=f'{R}/ft_params.json'; json.dump(dict(fam=efam(fam),p=p,tf=tf),open(cfg,'w'))
    cmd=['C:/Users/matth/okx-bt/venv/bin/freqtrade','backtesting','-c','C:/Users/matth/okx-bt/config.json','--userdir','C:/Users/matth/okx-bt/ud',
         '--datadir','C:/Users/matth/okx-bt/ud/data_proxy/okx','-s','IntradayGeneric','-i',tf,
         '--timerange',START.replace('-','')+'-','--max-open-trades','20','--stake-amount','unlimited',
         '--dry-run-wallet','1000','--export','none']
    try: out=subprocess.run(cmd,capture_output=True,text=True,timeout=5400,env=dict(os.environ,INTRADAY_PARAMS=cfg)).stdout
    except Exception as e: return f'FAIL {e}'
    g=lambda lab:next((ln.split('│')[2].strip() for ln in out.splitlines() if lab in ln and ln.count('│')>=3),'?')
    sh=g('Sharpe (daily wallet balance)'); res=f"trades {g('Total/Daily Avg Trades')}, profit {g('Total profit %')}, PF {g('Profit factor')}, DD {g('Max % of account underwater')}, Sharpe {sh}"
    try: ok=float(sh)>=2.0
    except: ok=False
    return ('OK ' if ok else 'WEAK ')+res

def pick_champions(rows):
    # merge verified passers file so a generation that finds nothing cannot wipe the champ
    pool=list(rows)
    if os.path.exists(f'{R}/passers.jsonl'):
        for ln in open(f'{R}/passers.jsonl'):
            try: pool.append(json.loads(ln))
            except Exception: pass
    c30=c15=None
    for r in pool:
        if not str(r.get('ft','')).startswith('OK'): continue
        sz=r.get('sized')
        if not sized_pass(sz,-0.30): continue
        if c30 is None or rscore(r)>rscore(c30): c30=r
        if sized_pass(sz,-0.15) and (c15 is None or rscore(r)>rscore(c15)): c15=r
    cf=None
    for r in pool:
        if ft_sh(r.get('ft',''))<1.5 or not fast_pass(r.get('sized')) or posm(r)<0.5: continue
        if cf is None or r['sized']['net']>cf['sized']['net']: cf=r
    def pack(r):
        if not r: return None
        return dict(fam=r['fam'],tf=r['tf'],p=r['p'],sz=r['sz'],sized=r['sized'],taker=r['taker'],ft=r['ft'],score=rscore(r),ts=r.get('ts',time.time()))
    pc=c30
    if cf is not None and (c30 is None or (cf['sized']['net']>c30['sized']['net'] and cf['sized']['ho_sh']>c30['sized']['ho_sh'])): pc=cf
    out=dict(dd30=pack(c30),dd15=pack(c15),fast=pack(cf),paper=pack(pc),updated=pd.Timestamp.now(tz='Asia/Shanghai').isoformat())
    json.dump(out,open(CJ,'w'),indent=1)
    return out

def report(rows,t0,gen_tf,n_new):
    chs=pick_champions(rows)
    rec=[]
    for r in rows:
        sz=r.get('sized') or {}
        rec.append(dict(fam=r['fam'],tf=r['tf'],params=json.dumps(r['p']),sz=json.dumps(r.get('sz',{})),
            tr_sh=sz.get('tr_sh',r['taker']['train']['sh']),ho_sh=sz.get('ho_sh',r['taker']['hold']['sh']),
            tr_tr=sz.get('tr_tr',r['taker']['train']['tr']),ho_tr=sz.get('ho_tr',r['taker']['hold']['tr']),
            score=rscore(r),dd=sz.get('dd',r['taker']['all']['dd']),net=sz.get('net',r['taker']['all']['net']),
            final=sz.get('final',''),liq=sz.get('liq',''),PASS=bool(r.get('ft','').startswith('OK') and sized_pass(sz)),
            PASS15=bool(r.get('ft','').startswith('OK') and sized_pass(sz,-0.15)),ft=r.get('ft','')))
    df=pd.DataFrame(rec).sort_values('score',ascending=False); df.to_csv(f'{R}/leaderboard.csv',index=False)
    with open(f'{R}/STATUS.md','w') as f:
        f.write(f"# intraday evolving search — BACKTEST ONLY (strategy+sizing)\nUpdated {pd.Timestamp.now(tz='Asia/Shanghai'):%Y-%m-%d %H:%M} UTC+8. Last gen TF={gen_tf}, {n_new} new in {round((time.time()-t0)/60,1)} min.\n")
        f.write(f"Total configs tried: {len(df)} (multiple-testing risk rises with this). Window {START}.., train<{SPLIT}. Costs 0.07%/side + funding on notional.\n")
        f.write(f"Gates: soft unit-BT {GATE}; sized DD≤30% (hard champion), preferred DD≤15%; liq=0; SL inside liq; ho_sh≥2 tr_sh≥1.5; trades≥150/100.\n")
        f.write(f"Verified DD≤30 passers: {int(df.PASS.sum())}; DD≤15: {int(df.PASS15.sum())}. By family: {df.fam.value_counts().to_dict()}\n\n")
        for lab,c in [('DD≤30 champion',chs.get('dd30')),('FAST growth (max net, DD≤30)',chs.get('fast')),('PAPER follows',chs.get('paper')),('DD≤15 best',chs.get('dd15'))]:
            f.write(f"## {lab}\n"); f.write(json.dumps(c,indent=1) if c else 'None'); f.write('\n\n')
        f.write(df.head(25).to_markdown(index=False))
    return df,chs

def main():
    t0=time.time(); rows=[json.loads(l) for l in open(LOG)] if os.path.exists(LOG) else []; done={r['key'] for r in rows}
    # migrate old rows without sz: leave them; they won't score well without sized
    tf=os.environ.get('TF') or ch(TFW); X=E.load(tf)
    pool=sorted([r for r in rows if r['tf']==tf and r.get('sized')],key=lambda r:-rscore(r))[:15]
    n=0; tries=0
    while n<GEN and tries<GEN*12:
        tries+=1
        if pool and random.random()<(0.6 if not os.environ.get('FAMS') else 0.0):
            par=ch(pool); fam=par['fam']; p,sz=mutate(fam,par['p'],par.get('sz',SIZES()))
        else: fam,p,sz=rand_cfg(tf)
        k=key(fam,tf,p,sz)
        if k in done: continue
        done.add(k)
        try:
            bt=E.backtest(X,efam(fam),p,tf)
            r=dict(key=k,fam=fam,tf=tf,p=p,sz=sz,taker=E.evaluate(bt,*TAKER,START,SPLIT),
                   maker=E.evaluate(E.backtest(X,efam(fam),p,tf,mk=1),*MAKER,START,SPLIT),ts=time.time())
            if soft_pass(r) and r['taker']['all']['dd']>=-0.35:
                r['sized']=size_eval(X,fam,p,sz,tf)
                try: r['sized_mk']=size_eval(X,fam,p,sz,tf,mk=1)
                except Exception: r['sized_mk']=None
            else:
                r['sized']=None
        except Exception as e:
            print('err',fam,p,sz,e); continue
        if r.get('sized') and (sized_pass(r['sized'],-0.30) or fast_pass(r['sized'])) and r['taker']['train']['sh']>=1.5 and r['taker']['hold']['sh']>=1.5 and (sized_pass(r['sized'],-0.30) or fast_pass(r['sized'])):
            r['ft']=crosscheck(fam,p,tf)
            if r['ft'].startswith('OK') or ft_sh(r['ft'])>=1.5:
                with open(f'{R}/passers.jsonl','a') as f: f.write(json.dumps(dict(fam=fam,tf=tf,p=p,sz=sz,sized=r['sized'],taker=r['taker'],ft=r['ft'],ts=time.time()))+'\n')
                with open(f'{R}/FOUND.md','a') as f: f.write(f"\n## {pd.Timestamp.now(tz='Asia/Shanghai'):%F %H:%M} {fam} {tf} sz={sz} {json.dumps(p)}\nsized={json.dumps(r['sized'])}\nfreqtrade: {r['ft']}\n")
                notify(f"回测候选(仅回测,未下单): {fam} {tf} 持出夏普{r['sized']['ho_sh']} 净{r['sized']['net']} DD{r['sized']['dd']} 杠杆{sz['lev']}x 保证金{sz['margin']}")
        rows.append(r); open(LOG,'a').write(json.dumps(r)+'\n'); n+=1
    report(rows,t0,tf,n); print('gen done',tf,n,round(time.time()-t0),'s')
if __name__=='__main__': main()
