import io,json,math,requests
from pathlib import Path
import numpy as np,pandas as pd
PAIRS=['eurjpy','gbpjpy','audjpy'];PIP={p:.01 for p in PAIRS};COST={'eurjpy':1.4,'gbpjpy':2.4,'audjpy':1.3};SPLITS={'train':(2016,2020),'val':(2021,2022),'test':(2023,2024),'holdout':(2025,2026)}
def fred(series):
 u=f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}';r=requests.get(u,timeout=60);r.raise_for_status();d=pd.read_csv(io.StringIO(r.text));d.columns=['date',series];d['date']=pd.to_datetime(d.date,utc=True);d[series]=pd.to_numeric(d[series],errors='coerce');return d.set_index('date')
def load(p):
 d=pd.read_csv(f'/tmp/{p}_trend_h1.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 d=d.dropna();prev=d.close.shift(1);tr=pd.concat([(d.high-d.low)/.01,(d.high-prev).abs()/.01,(d.low-prev).abs()/.01],axis=1).max(axis=1);d['atr']=tr.rolling(14,min_periods=10).mean().shift(1);d['mom24']=d.close/d.close.shift(24)-1;return d
def macro():
 s=fred('SP500');v=fred('VIXCLS');x=s.join(v,how='outer').sort_index().ffill();x['sp1']=x.SP500.pct_change(1);x['sp5']=x.SP500.pct_change(5);x['v1']=x.VIXCLS.diff(1);x['v5']=x.VIXCLS.diff(5)
 # Conservative availability: use each close observation only from the next UTC day.
 x.index=x.index+pd.Timedelta(days=1);return x
def metric(a):
 a=np.asarray(a,float);n=len(a)
 if not n:return {'n':0}
 gp=a[a>0].sum();gl=-a[a<0].sum();sd=a.std(ddof=1) if n>1 else np.nan;return {'n':n,'mean_R':round(float(a.mean()),4),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((a>0).mean()*100),1),'t':round(float(a.mean()/(sd/math.sqrt(n))),2) if n>1 and sd>0 else None}
def sm(dates,s):lo,hi=SPLITS[s];y=pd.DatetimeIndex(dates).year;return (y>=lo)&(y<=hi)
def run(d,p,x,spec,costmult=1):
 idx=d.index;rows=[];dates=[];vals=[]
 for day in pd.date_range(max(idx.min().date(),pd.Timestamp('2016-01-01').date()),idx.max().date(),freq='D',tz='UTC'):
  if day.weekday()>=5:continue
  i=idx.searchsorted(day)
  if i>=len(idx) or (idx[i]-day)>pd.Timedelta(hours=2):continue
  macroday=day.normalize();j=x.index.searchsorted(macroday,side='right')-1
  if j<0:continue
  row=x.iloc[j];h=spec['h'];sp=row[f'sp{h}'];vv=row[f'v{h}']
  if not(np.isfinite(sp) and np.isfinite(vv)):continue
  dire=1 if sp>0 and vv<0 else (-1 if sp<0 and vv>0 else 0)
  if dire==0:continue
  if spec['confirm'] and np.sign(float(d.iloc[i].mom24))!=dire:continue
  ei=i;xi=i+spec['hold'];
  if xi>=len(d):continue
  entry=float(d.open.iloc[ei]);atr=float(d.atr.iloc[ei]);stop=max(5,atr*spec['stop']);spx=entry-dire*stop*.01;st=False
  for k in range(ei,xi+1):
   if dire>0 and float(d.low.iloc[k])<=spx:st=True;break
   if dire<0 and float(d.high.iloc[k])>=spx:st=True;break
  cost=COST[p]*costmult;R=(-1-cost/stop) if st else dire*(float(d.open.iloc[xi])-entry)/.01/stop-cost/stop
  vals.append(R);dates.append(idx[ei])
 return np.array(vals),pd.DatetimeIndex(dates)
def ev(d,p,x,spec):
 vals,dates=run(d,p,x,spec);o=dict(spec);o['pair']=p
 for s in SPLITS:o[s]=metric(vals[sm(dates,s)])
 return o
def score(o):
 t=o['train'];v=o['val']
 if t.get('n',0)<100 or v.get('n',0)<50 or t.get('mean_R',-9)<=0 or v.get('mean_R',-9)<=0:return -999
 return min(t['mean_R'],v['mean_R'])
def main():
 x=macro();data={p:load(p) for p in PAIRS};c=[]
 for p,d in data.items():
  for h in [1,5]:
   for conf in [False,True]:
    for hold in [6,12,18]:
     for stop in [1.5,2.0]:c.append(ev(d,p,x,{'h':h,'confirm':conf,'hold':hold,'stop':stop}))
 winners=[]
 for p in PAIRS:
  q=sorted([z for z in c if z['pair']==p],key=score,reverse=True)
  if q and score(q[0])>-900:winners.append(q[0])
 passing=[z for z in winners if z['test'].get('mean_R',-9)>0 and (z['test'].get('pf') or 0)>1]
 out={'status':'FX_JPY_RISK_PHASE5','source':'Dukascopy H1 + FRED SP500/VIX, macro lagged one day','candidate_count':len(c),'winners':winners,'passing_test':passing,'top':sorted(c,key=score,reverse=True)[:20]};Path('tmp/fx_jpy_risk_phase5_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'winners':[(z['pair'],z['h'],z['confirm'],z['hold'],z['stop'],z['train']['mean_R'],z['val']['mean_R'],z['test']['mean_R'],z['holdout']['mean_R']) for z in winners]},indent=2))
if __name__=='__main__':main()
