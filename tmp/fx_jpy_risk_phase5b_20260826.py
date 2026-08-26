import io,json,math,requests
from pathlib import Path
import numpy as np,pandas as pd
PAIRS=['eurjpy','audjpy'];COST={'eurjpy':1.4,'audjpy':1.3};SPLITS={'train':(2016,2020),'val':(2021,2022),'test':(2023,2024),'holdout':(2025,2026)}
PRIMARY={'h':5,'confirm':True,'hold':18,'stop':2.0}
def fred(s):
 r=requests.get(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}',timeout=60);r.raise_for_status();d=pd.read_csv(io.StringIO(r.text));d.columns=['date',s];d.date=pd.to_datetime(d.date,utc=True);d[s]=pd.to_numeric(d[s],errors='coerce');return d.set_index('date')
def macro():
 x=fred('SP500').join(fred('VIXCLS'),how='outer').sort_index().ffill();x['sp5']=x.SP500.pct_change(5);x['v5']=x.VIXCLS.diff(5);x.index=x.index+pd.Timedelta(days=1);return x
def load(p):
 d=pd.read_csv(f'/tmp/{p}_trend_h1.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 d=d.dropna();prev=d.close.shift(1);tr=pd.concat([(d.high-d.low)/.01,(d.high-prev).abs()/.01,(d.low-prev).abs()/.01],axis=1).max(axis=1);d['atr']=tr.rolling(14,min_periods=10).mean().shift(1)
 # At an entry timestamp t, only the previous completed H1 bar may be used.
 d['mom24_known']=(d.close.shift(1)/d.close.shift(25)-1)
 return d
def met(a):
 a=np.array(a,float);n=len(a)
 if not n:return {'n':0}
 gp=a[a>0].sum();gl=-a[a<0].sum();sd=a.std(ddof=1) if n>1 else np.nan;return {'n':n,'mean_R':round(float(a.mean()),4),'median_R':round(float(np.median(a)),4),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((a>0).mean()*100),1),'t':round(float(a.mean()/(sd/math.sqrt(n))),2) if n>1 and sd>0 else None}
def run(d,p,x,costmult=1.0,hold=18,stopm=2.0):
 vals=[];dates=[];details=[];idx=d.index
 start=max(idx.min().date(),pd.Timestamp('2016-01-01').date())
 for day in pd.date_range(start,idx.max().date(),freq='D',tz='UTC'):
  if day.weekday()>=5:continue
  i=idx.searchsorted(day)
  if i>=len(idx) or idx[i]-day>pd.Timedelta(hours=2):continue
  j=x.index.searchsorted(day.normalize(),side='right')-1
  if j<0:continue
  m=x.iloc[j];
  if not(np.isfinite(m.sp5) and np.isfinite(m.v5)):continue
  dire=1 if m.sp5>0 and m.v5<0 else (-1 if m.sp5<0 and m.v5>0 else 0)
  if dire==0:continue
  mom=float(d.iloc[i].mom24_known)
  if not np.isfinite(mom) or np.sign(mom)!=dire:continue
  xi=i+hold
  if xi>=len(d):continue
  entry=float(d.open.iloc[i]);atr=float(d.atr.iloc[i]);
  if not np.isfinite(atr) or atr<=0:continue
  stop=max(5,atr*stopm);sp=entry-dire*stop*.01;st=False
  for k in range(i,xi+1):
   if dire>0 and float(d.low.iloc[k])<=sp:st=True;break
   if dire<0 and float(d.high.iloc[k])>=sp:st=True;break
  cost=COST[p]*costmult;R=(-1-cost/stop) if st else dire*(float(d.open.iloc[xi])-entry)/.01/stop-cost/stop
  vals.append(R);dates.append(idx[i]);details.append((idx[i],idx[xi],p,float(R)))
 return np.array(vals),pd.DatetimeIndex(dates),details
def evaluate(d,p,x,costmult,hold,stopm):
 vals,dates,_=run(d,p,x,costmult,hold,stopm);o={'pair':p,'costmult':costmult,'hold':hold,'stop':stopm}
 for s,(lo,hi) in SPLITS.items():m=(dates.year>=lo)&(dates.year<=hi);o[s]=met(vals[m])
 return o
def portfolio(rows,risk=0.008,cap=500000):
 # One JPY-risk position at a time. AUDJPY gets priority if simultaneous, fixed ex ante from stronger pre-2025 validation/test evidence.
 pri={'audjpy':2,'eurjpy':1};rows=sorted(rows,key=lambda z:(z[0],-pri[z[2]]));active_end=None;kept=[]
 for st,en,p,R in rows:
  if active_end is not None and active_end>st:continue
  active_end=en;kept.append((st,p,R))
 if not kept:return {'n':0}
 df=pd.DataFrame(kept,columns=['dt','pair','R']);m=df.set_index('dt').R.resample('MS').sum();m=m.reindex(pd.date_range(m.index.min(),m.index.max(),freq='MS',tz='UTC'),fill_value=0);pnl=m*cap*risk;tradep=df.R*cap*risk;eq=cap+tradep.cumsum();peak=np.maximum.accumulate(eq);dd=(eq-peak)/peak;gp=df.loc[df.R>0,'R'].sum();gl=-df.loc[df.R<0,'R'].sum()
 return {'n':len(df),'months':len(m),'trades_per_month':round(len(df)/len(m),1),'avg_month_R':round(float(m.mean()),3),'avg_monthly_jpy':round(float(pnl.mean())),'median_monthly_jpy':round(float(pnl.median())),'positive_month_pct':round(float((pnl>0).mean()*100),1),'p10_jpy':round(float(pnl.quantile(.1))),'p90_jpy':round(float(pnl.quantile(.9))),'max_dd_pct':round(float(-dd.min()*100),2),'avg_R':round(float(df.R.mean()),4),'pf_R':round(float(gp/gl),3) if gl>0 else None}
def main():
 x=macro();data={p:load(p) for p in PAIRS};primary={};neighborhood={};holdrows={cm:[] for cm in [1,1.5,2]}
 for p,d in data.items():
  primary[p]={str(cm):evaluate(d,p,x,cm,18,2.0) for cm in [1,1.5,2]}
  # Neighborhood is diagnostic only; primary parameters remain frozen.
  neighborhood[p]=[evaluate(d,p,x,1,h,s) for h in [12,18,24] for s in [1.5,2.0,2.5]]
  for cm in [1,1.5,2]:
   vals,dates,rows=run(d,p,x,cm,18,2.0);holdrows[cm]+=[z for z in rows if z[0].year>=2025]
 ports={}
 for cm in [1,1.5,2]:
  for r in [.004,.006,.008]:ports[f'cost{cm}_risk{r}']=portfolio(holdrows[cm],r)
 out={'status':'FX_JPY_RISK_PHASE5B_NO_LOOKAHEAD','protocol':'Primary rule frozen from phase5 train/validation selection; momentum confirmation corrected to previous completed H1 bar; FRED daily data usable next UTC day; 2025-26 reported without retuning','primary':primary,'neighborhood_diagnostic':neighborhood,'holdout_portfolio':ports};Path('tmp/fx_jpy_risk_phase5b_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'primary':{p:{cm:[z['train']['mean_R'],z['val']['mean_R'],z['test']['mean_R'],z['holdout']['mean_R'],z['holdout']['pf']] for cm,z in v.items()} for p,v in primary.items()},'portfolio':ports},indent=2))
if __name__=='__main__':main()
