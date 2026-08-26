import json,math
from pathlib import Path
import numpy as np,pandas as pd
PAIRS=['eurusd','gbpusd','audusd','usdcad','usdchf','usdjpy']
PIP={p:(0.01 if p=='usdjpy' else 0.0001) for p in PAIRS};COST={'eurusd':0.8,'gbpusd':1.3,'audusd':1.1,'usdcad':1.5,'usdchf':1.5,'usdjpy':1.0}
SPLITS={'train':(2014,2018),'val':(2019,2021),'test':(2022,2024),'holdout':(2025,2026)}

def load(p):
 d=pd.read_csv(f'/tmp/{p}_wmr_m15.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 return d.dropna(subset=['open','close'])

def at(d,t,maxmin=16):
 i=d.index.searchsorted(t)
 if i>=len(d) or (d.index[i]-t).total_seconds()>maxmin*60:return None
 return i

def available_days(d):
 li=d.index.tz_convert('Europe/London');dates=pd.Series(li.date,index=d.index);return sorted(set(dates))

def monthends(d):
 days=available_days(d);z=pd.DataFrame({'date':pd.to_datetime(days)});z['ym']=z.date.dt.to_period('M');return set(z.groupby('ym').date.max().dt.date)

def build(p,d,holdmin,costmult,only_monthend=True):
 pip=PIP[p];mes=monthends(d);rows=[]
 for date in available_days(d):
  if (date in mes)!=only_monthend:continue
  base=pd.Timestamp(date).tz_localize('Europe/London')
  t0=(base+pd.Timedelta(hours=15,minutes=30)).tz_convert('UTC');t1=(base+pd.Timedelta(hours=16)).tz_convert('UTC')
  # Avoid trading exactly inside the WMR window; enter 15m after 4pm, then hold 30/60m.
  e0=(base+pd.Timedelta(hours=16,minutes=15)).tz_convert('UTC');e1=e0+pd.Timedelta(minutes=holdmin)
  a,b,c,e=[at(d,t) for t in [t0,t1,e0,e1]]
  if None in [a,b,c,e]:continue
  pre=(float(d.open.iloc[b])-float(d.open.iloc[a]))/pip
  if pre==0:continue
  direction=-np.sign(pre);gross=direction*(float(d.open.iloc[e])-float(d.open.iloc[c]))/pip;net=gross-COST[p]*costmult
  rows.append({'dt':d.index[c],'pair':p,'pre_pips':float(pre),'direction':int(direction),'net_pips':float(net)})
 return pd.DataFrame(rows)

def metric(df):
 if df is None or len(df)==0:return {'n':0}
 x=df.net_pips.values;gp=x[x>0].sum();gl=-x[x<0].sum();sd=x.std(ddof=1) if len(x)>1 else np.nan;m=df.set_index('dt').net_pips.resample('MS').sum();y=df.set_index('dt').net_pips.resample('YS').sum();corr=np.corrcoef(df.pre_pips.values,df.net_pips.values)[0,1] if len(df)>2 else np.nan
 return {'n':len(df),'trades_per_month':round(len(df)/max(1,len(m)),1),'mean_net_pips':round(float(x.mean()),3),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((x>0).mean()*100),1),'t':round(float(x.mean()/(sd/math.sqrt(len(x)))),2) if len(x)>1 and sd>0 else None,'positive_month_pct':round(float((m>0).mean()*100),1),'positive_year_pct':round(float((y>0).mean()*100),1),'mean_abs_pre_pips':round(float(df.pre_pips.abs().mean()),2),'pre_vs_strategy_corr':round(float(corr),3) if np.isfinite(corr) else None}

def split(df,a,b):return df[(df.dt.dt.year>=a)&(df.dt.dt.year<=b)].copy()

def main():
 data={p:load(p) for p in PAIRS};out={'status':'FX_WMR_MONTHEND_PHASE16','source':'Dukascopy free M15','rule':'last available London trading day each month; observe 15:30-16:00 pre-fix move; enter opposite at 16:15; hold 30/60m; no threshold optimization','results':{}}
 for cm in [1.0,1.5,2.0]:
  out['results'][str(cm)]={}
  for hold in [30,60]:
   me=[];ctrl=[];bp={}
   for p,d in data.items():
    q=build(p,d,hold,cm,True);c=build(p,d,hold,cm,False);me.append(q);ctrl.append(c);bp[p]={s:metric(split(q,*ab)) for s,ab in SPLITS.items()}
   mdf=pd.concat(me,ignore_index=True);cdf=pd.concat(ctrl,ignore_index=True);out['results'][str(cm)][str(hold)]={'monthend':{s:metric(split(mdf,*ab)) for s,ab in SPLITS.items()},'non_monthend_control':{s:metric(split(cdf,*ab)) for s,ab in SPLITS.items()},'by_pair':bp}
 Path('tmp/fx_wmr_monthend_phase16_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({cm:{h:z['monthend'] for h,z in d.items()} for cm,d in out['results'].items()},indent=2))
if __name__=='__main__':main()
