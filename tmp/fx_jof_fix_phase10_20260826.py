import json,math
from pathlib import Path
import numpy as np,pandas as pd
PAIRS=['eurusd','gbpusd','audusd','nzdusd','usdjpy','usdcad','usdchf']
ORIENT={'eurusd':1,'gbpusd':1,'audusd':1,'nzdusd':1,'usdjpy':-1,'usdcad':-1,'usdchf':-1}
PIP={p:(0.01 if p.endswith('jpy') else 0.0001) for p in PAIRS}
COST={'eurusd':0.8,'gbpusd':1.3,'audusd':1.1,'nzdusd':1.5,'usdjpy':1.0,'usdcad':1.5,'usdchf':1.5}
SPLITS={'train':(2014,2019),'val':(2020,2022),'test':(2023,2024),'holdout':(2025,2026)}

def load(p):
 d=pd.read_csv(f'/tmp/{p}_jof_m5.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 return d.dropna(subset=['open','close'])

def at(d,t,maxmin=10):
 i=d.index.searchsorted(t)
 if i>=len(d):return None
 if (d.index[i]-t).total_seconds()>maxmin*60:return None
 return i

def prev_ny_17(t):
 x=t.tz_convert('America/New_York');day=pd.Timestamp(x.date()).tz_localize('America/New_York')+pd.Timedelta(hours=17)
 if day>=t.tz_convert('America/New_York'):day-=pd.Timedelta(days=1)
 return day.tz_convert('UTC')

def next_ny(t,h):
 x=t.tz_convert('America/New_York');day=pd.Timestamp(x.date()).tz_localize('America/New_York')+pd.Timedelta(hours=h)
 if day<=x:day+=pd.Timedelta(days=1)
 return day.tz_convert('UTC')

def segment_return(data,t0,t1,sign,costmult):
 vals=[];details=[]
 for p,d in data.items():
  i=at(d,t0);j=at(d,t1)
  if i is None or j is None or j<=i:return None
  e=float(d.open.iloc[i]);x=float(d.open.iloc[j]);fc_bps=ORIENT[p]*math.log(x/e)*10000.0
  # roundtrip cost converted from pip quote to bps of notional
  cbps=COST[p]*costmult*PIP[p]/e*10000.0
  r=sign*fc_bps-cbps;vals.append(r);details.append((p,r,fc_bps,cbps))
 return float(np.mean(vals)),details

def metric(a):
 a=np.asarray(a,float);n=len(a)
 if n==0:return {'n':0}
 gp=a[a>0].sum();gl=-a[a<0].sum();sd=a.std(ddof=1) if n>1 else np.nan
 return {'n':n,'mean_bps':round(float(a.mean()),4),'median_bps':round(float(np.median(a)),4),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((a>0).mean()*100),1),'t':round(float(a.mean()/(sd/math.sqrt(n))),2) if n>1 and sd>0 else None,'sum_bps':round(float(a.sum()),2)}

def split_metric(df,col):
 out={}
 for s,(a,b) in SPLITS.items():
  x=df[(df.date.dt.year>=a)&(df.date.dt.year<=b)][col].dropna();o=metric(x.values)
  if len(x):
   z=pd.Series(x.values,index=df.loc[x.index,'date']);m=z.groupby([z.index.year,z.index.month]).sum();y=z.groupby(z.index.year).sum();o['positive_month_pct']=round(float((m>0).mean()*100),1);o['positive_year_pct']=round(float((y>0).mean()*100),1)
  out[s]=o
 return out

def equity(df,yr0,yr1,leverage,cap=500000):
 x=df[(df.date.dt.year>=yr0)&(df.date.dt.year<=yr1)].copy();
 if len(x)==0:return {'n':0}
 # all four legs are sequential; sum net bps per day then apply total notional/equity leverage.
 x['ret']=x['daily_bps']/10000.0*leverage;eq=cap;peak=cap;md=0;vals=[]
 for _,r in x.iterrows():
  eq*=max(.01,1+float(r.ret));peak=max(peak,eq);md=max(md,(peak-eq)/peak);vals.append((r.date,eq))
 q=pd.DataFrame(vals,columns=['date','eq']);q['pnl']=q.eq.diff().fillna(q.eq.iloc[0]-cap);m=q.set_index('date').pnl.resample('MS').sum();m=m.reindex(pd.date_range(f'{yr0}-01-01',f'{yr1}-12-01',freq='MS',tz='UTC'),fill_value=0)
 return {'days':len(x),'leverage':leverage,'final_equity':round(float(eq)),'total_return_pct':round(float((eq/cap-1)*100),2),'avg_monthly_jpy':round(float(m.mean())),'median_monthly_jpy':round(float(m.median())),'positive_month_pct':round(float((m>0).mean()*100),1),'p10_month_jpy':round(float(m.quantile(.1))),'p90_month_jpy':round(float(m.quantile(.9))),'max_dd_pct':round(float(md*100),2)}

def build(data,costmult):
 rows=[]
 # Tokyo business date anchors. Require weekdays only; missing-market holidays drop automatically.
 for day in pd.date_range('2014-01-06','2026-07-31',freq='B'):
  # 09:55 Tokyo local
  tfix=pd.Timestamp(day.date()).tz_localize('Asia/Tokyo')+pd.Timedelta(hours=9,minutes=55);tfix=tfix.tz_convert('UTC')
  tpre=prev_ny_17(tfix);teu=next_ny(tfix,2)
  # European anchors based on local date at London/Frankfurt after Europe open
  ld=teu.tz_convert('Europe/London').date();bd=teu.tz_convert('Europe/Berlin').date()
  tecb=(pd.Timestamp(bd).tz_localize('Europe/Berlin')+pd.Timedelta(hours=14,minutes=15)).tz_convert('UTC')
  tlon=(pd.Timestamp(ld).tz_localize('Europe/London')+pd.Timedelta(hours=16)).tz_convert('UTC')
  tend=next_ny(tlon,17)
  segs=[('preT',tpre,tfix,-1),('postT',tfix,teu,1),('preE',teu,tecb,-1),('postL',tlon,tend,1)]
  rr={'date':tfix}
  ok=True
  for name,a,b,sgn in segs:
   z=segment_return(data,a,b,sgn,costmult)
   if z is None:ok=False;break
   rr[name]=z[0]
  if ok:
   rr['daily_bps']=rr['preT']+rr['postT']+rr['preE']+rr['postL'];rows.append(rr)
 return pd.DataFrame(rows)

def main():
 data={p:load(p) for p in PAIRS};out={'status':'FX_JOF_FIX_PHASE10','source':'Dukascopy M5 free','paper_replication':'Krohn Mueller Whelan JF 2024 W-shaped USD fix reversal; fixed windows, no parameter search','pairs':PAIRS,'cost_pips':COST,'splits':SPLITS,'cost_scenarios':{}}
 for cm in [1.0,1.5,2.0]:
  df=build(data,cm);sc={'n_days':len(df),'segments':{c:split_metric(df,c) for c in ['preT','postT','preE','postL']},'daily':split_metric(df,'daily_bps'),'holdout_equity':{}}
  for lev in [1,2,4,6,8]:sc['holdout_equity'][str(lev)]=equity(df,2025,2026,lev)
  out['cost_scenarios'][str(cm)]=sc
 Path('tmp/fx_jof_fix_phase10_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({k:{'n':v['n_days'],'daily':v['daily'],'equity8':v['holdout_equity']['8']} for k,v in out['cost_scenarios'].items()},indent=2))
if __name__=='__main__':main()
