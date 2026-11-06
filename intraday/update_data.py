# Daily: append new days of 1m candles from OKX public archive; refresh real funding (public API) into data_proxy. No keys.
import os,glob,io,zipfile,json,time,urllib.request,pandas as pd
D='C:/Users/matth/okx-bt/ud/data/okx/futures'; FD='C:/Users/matth/okx-bt/ud/data_proxy/okx/futures'; UA={'User-Agent':'Mozilla/5.0'}
def get(u):
    for a in range(4):
        try: return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=20).read()
        except Exception as e:
            if '404' in str(e): return None
            time.sleep(3+a*3)
for f in sorted(glob.glob(f'{D}/*-1m-futures.feather')):
    p=os.path.basename(f).split('-')[0]; inst=p.replace('_USDT_USDT','-USDT-SWAP'); df=pd.read_feather(f)
    last=df.date.max().normalize(); new=[]
    for d in pd.date_range(last+pd.Timedelta('1D'),pd.Timestamp.now(tz='UTC').normalize()-pd.Timedelta('1D')):
        b=get(f'https://static.okx.com/cdn/okex/traderecords/candlesticks/daily/{d:%Y%m%d}/{inst}-candlesticks-{d:%Y-%m-%d}.zip')
        if not b: continue
        z=zipfile.ZipFile(io.BytesIO(b)); x=pd.read_csv(z.open(z.namelist()[0]),usecols=['open','high','low','close','vol','open_time'])
        x['date']=pd.to_datetime(x.open_time,unit='ms',utc=True); new.append(x.rename(columns={'vol':'volume'})[['date','open','high','low','close','volume']])
    if new:
        df=pd.concat([df]+new).drop_duplicates('date').sort_values('date').reset_index(drop=True); df.to_feather(f)
        r=df.set_index('date').resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index(); r.to_feather(f.replace('-1m-','-5m-'))
        r3=df.set_index('date').resample('3min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index(); r3.to_feather(f.replace('-1m-','-3m-'))
    b=get(f'https://www.okx.com/api/v5/public/funding-rate-history?instId={inst}&limit=100')
    if b:
        rows=json.loads(b).get('data',[]); s=pd.Series({pd.Timestamp(int(x['fundingTime']),unit='ms',tz='UTC'):float(x['realizedRate'] or x['fundingRate']) for x in rows})
        ff=f'{FD}/{p}-1h-funding_rate.feather'; o=pd.read_feather(ff).set_index('date')['funding_rate']
        m=pd.concat([o,s]); m=m[~m.index.duplicated(keep='last')].sort_index(); m.rename('funding_rate').rename_axis('date').reset_index().to_feather(ff)
    print(p,'+',len(new),'days',flush=True); time.sleep(0.3)
