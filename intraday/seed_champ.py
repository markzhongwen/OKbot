"""Seed champion.json from known ORB passer with sizing sweep. BACKTEST ONLY."""
import sys,json,time; sys.path.insert(0,'C:/Users/matth/okx-bt/intraday')
import evolve as EV, engine as E
p={"open_min":840,"range_min":30,"win_min":60,"fade":1,"sl":2,"tp":0,"ts":8,"cool":3,"dir":"long","htf":240,"htf_ema":50}
X=E.load('15m'); fam='orb'; tf='15m'
cands=[]
for margin,lev,maxpos in [(0.1,3,5),(0.1,5,2),(0.05,10,2),(0.15,3,2),(0.25,5,2),(0.1,8,3),(0.15,5,3),(0.25,3,5),(0.05,5,5)]:
    sz=dict(margin=margin,lev=lev,maxpos=maxpos)
    sized=EV.size_eval(X,fam,p,sz,tf)
    bt=E.backtest(X,fam,p,tf)
    r=dict(fam=fam,tf=tf,p=p,sz=sz,sized=sized,taker=E.evaluate(bt,*EV.TAKER,EV.START,EV.SPLIT),
           ft='OK trades 614 / 3.34, profit 8.09%, PF 1.69, DD 2.39%, Sharpe 2.98',ts=time.time(),
           key=EV.key(fam,tf,p,sz))
    r['score']=EV.rscore(r); cands.append(r)
    print(sz, {k:sized[k] for k in ('tr_sh','ho_sh','dd','net','final','liq','sl_beyond','tr_tr','ho_tr')})
# write all sized passers
for r in cands:
    if EV.sized_pass(r['sized'],-0.30):
        with open('C:/Users/matth/okx-bt/intraday/passers.jsonl','a') as f: f.write(json.dumps({k:r[k] for k in ('fam','tf','p','sz','sized','taker','ft','ts')})+'\n')
chs=EV.pick_champions(cands)
print('champ',json.dumps(chs,indent=1)[:800])
