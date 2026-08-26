import json,math
from pathlib import Path
import numpy as np,pandas as pd
PAIRS=['eurusd','gbpusd','audusd','usdcad','usdchf','usdjpy']
PIP={p:(0.01 if p=='usdjpy' else 0.0001) for p in PAIRS}
SPLITS={'val':(2019,2021),'test':(2022,2024),'holdout':(2025,2026)}

def read(p,side):
 d=pd.read_csv(f'/tmp/{p}_reopen_{side}_h1.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 return d[['open','high','low','close']].dropna()

def load(p):
 b=read(p,'bid').add_prefix('b_');a=read(p,'ask').add_prefix('a_');return b.join(a,how='inner').dropna()

def build(p,d,threshold,slip_pips=0.0):
 pip=PIP[p];li=d.index.tz_convert('America/New_York');poss=np.flatnonzero((li.hour==17)&(li.minute==0)&(li.weekday<5));rows=[]
 for i in poss:
  if i<=0 or (d.index[i]-d.index[i-1]).total_seconds()>7200:continue
  prev_mid=(float(d.b_close.iloc[i-1])+float(d.a_close.iloc[i-1]))/2;opmid=(float(d.b_open.iloc[i])+float(d.a_open.iloc[i]))/2;gap=(opmid-prev_mid)/pip
  if abs(gap)<threshold:continue
  direction=-np.sign(gap)
  if direction>0:
   entry=float(d.a_open.iloc[i])+slip_pips*pip;exitp=float(d.b_close.iloc[i])-slip_pips*pip;net=(exitp-entry)/pip
  else:
   entry=float(d.b_open.iloc[i])-slip_pips*pip;exitp=float(d.a_close.iloc[i])+slip_pips*pip;net=(entry-exitp)/pip
  spread_open=(float(d.a_open.iloc[i])-float(d.b_open.iloc[i]))/pip;spread_close=(float(d.a_close.iloc[i])-float(d.b_close.iloc[i]))/pip
  rows.append({'dt':d.index[i],'pair':p,'gap_pips':float(gap),'direction':int(direction),'net_pips':float(net),'spread_open':float(spread_open),'spread_close':float(spread_close)})
 return pd.DataFrame(rows)

def metric(df):
 if df is None or len(df)==0:return {'n':0}
 x=df.net_pips.values;gp=x[x>0].sum();gl=-x[x<0].sum();sd=x.std(ddof=1) if len(x)>1 else np.nan;m=df.set_index('dt').net_pips.resample('MS').sum();y=df.set_index('dt').net_pips.resample('YS').sum()
 return {'n':len(df),'trades_per_month':round(len(df)/max(1,len(m)),1),'mean_net_pips':round(float(x.mean()),3),'median_net_pips':round(float(np.median(x)),3),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((x>0).mean()*100),1),'t':round(float(x.mean()/(sd/math.sqrt(len(x)))),2) if len(x)>1 and sd>0 else None,'positive_month_pct':round(float((m>0).mean()*100),1),'positive_year_pct':round(float((y>0).mean()*100),1),'avg_gap_pips':round(float(df.gap_pips.abs().mean()),2),'avg_open_spread':round(float(df.spread_open.mean()),2),'p90_open_spread':round(float(df.spread_open.quantile(.9)),2),'avg_close_spread':round(float(df.spread_close.mean()),2)}

def split(df,a,b):return df[(df.dt.dt.year>=a)&(df.dt.dt.year<=b)].copy()

def main():
 data={p:load(p) for p in PAIRS};out={'status':'FX_REOPEN_BIDASK_PHASE15','source':'Dukascopy free BID+ASK H1','execution':'signal on mid gap; long ASK-open -> BID-close, short BID-open -> ASK-close; no synthetic spread; optional extra slippage each side','thresholds':[1,2,5],'slippage_each_side_pips':[0,0.25,0.5],'results':{}}
 for slip in [0,0.25,0.5]:
  out['results'][str(slip)]={}
  for th in [1,2,5]:
   allrows=[];bp={}
   for p,d in data.items():
    q=build(p,d,th,slip);allrows.append(q);bp[p]={s:metric(split(q,*ab)) for s,ab in SPLITS.items()}
   pooled=pd.concat(allrows,ignore_index=True);out['results'][str(slip)][str(th)]={'pooled':{s:metric(split(pooled,*ab)) for s,ab in SPLITS.items()},'by_pair':bp}
 Path('tmp/fx_reopen_ba_phase15_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({sl:{th:z['pooled'] for th,z in d.items()} for sl,d in out['results'].items()},indent=2))
if __name__=='__main__':main()
