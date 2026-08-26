import json,math
from pathlib import Path
import numpy as np,pandas as pd
PIP=.01;BASE_COST=1.0;SPLITS={'train':(2014,2018),'val':(2019,2021),'test':(2022,2024),'holdout':(2025,2026)}
def load():
 d=pd.read_csv('/tmp/usdjpy_trend_h1.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 d=d.dropna();prev=d.close.shift(1);tr=pd.concat([(d.high-d.low)/PIP,(d.high-prev).abs()/PIP,(d.low-prev).abs()/PIP],axis=1).max(axis=1);d['atr']=tr.rolling(24,min_periods=18).mean().shift(1);r=d.close.diff()/PIP;d['sig']=r.rolling(480,min_periods=240).std().shift(1);d['z6']=((d.close-d.close.shift(6))/PIP)/(d.sig*np.sqrt(6));return d
def met(a,dates=None):
 a=np.asarray(a,float);n=len(a)
 if not n:return {'n':0}
 gp=a[a>0].sum();gl=-a[a<0].sum();sd=a.std(ddof=1) if n>1 else np.nan;o={'n':n,'mean_R':round(float(a.mean()),4),'median_R':round(float(np.median(a)),4),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((a>0).mean()*100),1),'t':round(float(a.mean()/(sd/math.sqrt(n))),2) if n>1 and sd>0 else None}
 if dates is not None:
  s=pd.Series(a,index=pd.DatetimeIndex(dates));m=s.groupby([s.index.year,s.index.month]).sum();o['positive_month_pct']=round(float((m>0).mean()*100),1)
 return o
def run(d,th=2.0,hold=12,stopm=1.5,costm=1.0):
 idx=d.index;vals=[];dates=[];rows=[];i=500;N=len(d)
 while i<N-hold-2:
  if idx[i].hour%3!=0 or idx[i].weekday()>=5 or (idx[i].weekday()==4 and idx[i].hour>=12):i+=1;continue
  z=float(d.iloc[i].z6)
  if not np.isfinite(z) or abs(z)<th:i+=1;continue
  dire=1 if z>0 else -1;ei=i+1;xi=ei+hold;entry=float(d.open.iloc[ei]);atr=float(d.atr.iloc[ei])
  if not np.isfinite(atr) or atr<=0:i+=1;continue
  stop=max(5,atr*stopm);sp=entry-dire*stop*PIP;st=False
  for k in range(ei,xi+1):
   if dire>0 and float(d.low.iloc[k])<=sp:st=True;break
   if dire<0 and float(d.high.iloc[k])>=sp:st=True;break
  cost=BASE_COST*costm;R=(-1-cost/stop) if st else dire*(float(d.open.iloc[xi])-entry)/PIP/stop-cost/stop
  vals.append(R);dates.append(idx[ei]);rows.append((idx[ei],idx[xi],float(R)));i=xi+1
 return np.array(vals),pd.DatetimeIndex(dates),rows
def ev(d,th,stopm,costm):
 vals,dates,_=run(d,th,12,stopm,costm);o={'threshold':th,'stop_mult':stopm,'cost_mult':costm}
 for s,(lo,hi) in SPLITS.items():m=(dates.year>=lo)&(dates.year<=hi);o[s]=met(vals[m],dates[m])
 return o
def port(rows,risk,cap=500000):
 q=[z for z in rows if z[0].year>=2025]
 if not q:return {'n':0}
 df=pd.DataFrame(q,columns=['dt','end','R']);m=df.set_index('dt').R.resample('MS').sum();m=m.reindex(pd.date_range(m.index.min(),m.index.max(),freq='MS',tz='UTC'),fill_value=0);pnl=m*cap*risk;tradep=df.R*cap*risk;eq=cap+tradep.cumsum();peak=np.maximum.accumulate(eq);dd=(eq-peak)/peak;gp=df.loc[df.R>0,'R'].sum();gl=-df.loc[df.R<0,'R'].sum()
 return {'n':len(df),'months':len(m),'trades_per_month':round(len(df)/len(m),1),'avg_month_R':round(float(m.mean()),3),'avg_monthly_jpy':round(float(pnl.mean())),'median_monthly_jpy':round(float(pnl.median())),'positive_month_pct':round(float((pnl>0).mean()*100),1),'p10_jpy':round(float(pnl.quantile(.1))),'p90_jpy':round(float(pnl.quantile(.9))),'max_dd_pct':round(float(-dd.min()*100),2),'avg_R':round(float(df.R.mean()),4),'pf_R':round(float(gp/gl),3) if gl>0 else None}
def main():
 d=load();primary={str(cm):ev(d,2.0,1.5,cm) for cm in [1,1.5,2]};neigh=[ev(d,th,sm,1) for th in [1.8,2.0,2.2] for sm in [1.25,1.5,1.75]];ports={}
 for cm in [1,1.5,2]:
  vals,dates,rows=run(d,2.0,12,1.5,cm)
  for r in [.004,.006,.008]:ports[f'cost{cm}_risk{r}']=port(rows,r)
 out={'status':'FX_USDJPY_SHOCK_PHASE6B_STRICT','protocol':'Primary rule frozen from broad phase6 before strict test: completed 6h move >=2 rolling sigma, continuation, enter next H1 open, 12h hold, no overlapping trades, 1.5x ATR stop','primary':primary,'neighborhood_diagnostic':neigh,'holdout_portfolio':ports};Path('tmp/fx_usdjpy_shock_phase6b_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'primary':{k:[v['train']['mean_R'],v['val']['mean_R'],v['test']['mean_R'],v['holdout']['mean_R'],v['holdout']['pf'],v['holdout']['n']] for k,v in primary.items()},'portfolio':ports},indent=2))
if __name__=='__main__':main()
