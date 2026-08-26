import json,math
from pathlib import Path
import numpy as np,pandas as pd
PAIRS=['usdjpy','eurjpy','gbpjpy','eurusd','gbpusd']
PIP={p:(0.01 if p.endswith('jpy') else 0.0001) for p in PAIRS}
COST={'usdjpy':1.0,'eurjpy':1.4,'gbpjpy':2.4,'eurusd':0.8,'gbpusd':1.3}
SPLITS={'train':(2014,2018),'val':(2019,2021),'test':(2022,2024),'holdout':(2025,2026)}

def load(p):
 d=pd.read_csv(f'/tmp/{p}_londonmom_m5.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 return d.dropna(subset=['open','close'])

def at(d,t,maxmin=6):
 i=d.index.searchsorted(t)
 if i>=len(d) or (d.index[i]-t).total_seconds()>maxmin*60:return None
 return i

def build(p,d,costmult=1.0):
 rows=[];pip=PIP[p]
 for day in pd.date_range('2014-01-06','2026-07-31',freq='B'):
  # Standard intraday-momentum replication: first 30m of London session predicts final 30m.
  base=pd.Timestamp(day.date()).tz_localize('Europe/London')
  s0=(base+pd.Timedelta(hours=8)).tz_convert('UTC');s1=(base+pd.Timedelta(hours=8,minutes=30)).tz_convert('UTC')
  e0=(base+pd.Timedelta(hours=16,minutes=30)).tz_convert('UTC');e1=(base+pd.Timedelta(hours=17)).tz_convert('UTC')
  a,b,c,e=[at(d,t) for t in [s0,s1,e0,e1]]
  if None in [a,b,c,e]:continue
  impulse=(float(d.open.iloc[b])-float(d.open.iloc[a]))/pip
  if impulse==0:continue
  signal=np.sign(impulse)
  # 2026 paper reports structurally reversed GBPUSD signal; freeze that rule ex ante.
  if p=='gbpusd':signal*=-1
  gross=signal*(float(d.open.iloc[e])-float(d.open.iloc[c]))/pip;net=gross-COST[p]*costmult
  rows.append({'dt':d.index[c],'pair':p,'impulse_pips':float(impulse),'signal':int(signal),'gross_pips':float(gross),'net_pips':float(net)})
 return pd.DataFrame(rows)

def metric(df):
 if df is None or len(df)==0:return {'n':0}
 a=df.net_pips.values;gp=a[a>0].sum();gl=-a[a<0].sum();sd=a.std(ddof=1) if len(a)>1 else np.nan;m=df.set_index('dt').net_pips.resample('MS').sum();y=df.set_index('dt').net_pips.resample('YS').sum()
 return {'n':len(df),'trades_per_month':round(len(df)/max(1,len(m)),1),'mean_net_pips':round(float(a.mean()),3),'median_net_pips':round(float(np.median(a)),3),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((a>0).mean()*100),1),'t':round(float(a.mean()/(sd/math.sqrt(len(a)))),2) if len(a)>1 and sd>0 else None,'positive_month_pct':round(float((m>0).mean()*100),1),'positive_year_pct':round(float((y>0).mean()*100),1),'mean_abs_impulse':round(float(df.impulse_pips.abs().mean()),2)}

def split(df,a,b):return df[(df.dt.dt.year>=a)&(df.dt.dt.year<=b)].copy()

def main():
 data={p:load(p) for p in PAIRS};out={'status':'FX_LONDON_MOM_PHASE14','source':'Dukascopy free M5','rule':'08:00-08:30 Europe/London sign predicts 16:30-17:00; GBPUSD reversed ex ante per 2026 study','no_parameter_search':True,'cost_scenarios':{}}
 for cm in [1.0,1.5,2.0]:
  res={};allrows=[]
  for p,d in data.items():
   q=build(p,d,cm);allrows.append(q);res[p]={s:metric(split(q,*ab)) for s,ab in SPLITS.items()}
  pooled=pd.concat(allrows,ignore_index=True);res['pooled']={s:metric(split(pooled,*ab)) for s,ab in SPLITS.items()};out['cost_scenarios'][str(cm)]=res
 Path('tmp/fx_london_mom_phase14_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps(out,indent=2,default=str))
if __name__=='__main__':main()
