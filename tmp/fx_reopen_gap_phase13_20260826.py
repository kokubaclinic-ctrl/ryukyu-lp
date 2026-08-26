import json,math
from pathlib import Path
import numpy as np,pandas as pd
PAIRS=['eurusd','gbpusd','audusd','usdcad','usdchf','usdjpy']
PIP={p:(0.01 if p=='usdjpy' else 0.0001) for p in PAIRS}
# Conservative normal-market round-trip friction. Rollover spread risk is stressed at 1.5x/2x and must later be verified on bid/ask tick data.
COST={'eurusd':0.8,'gbpusd':1.3,'audusd':1.1,'usdcad':1.5,'usdchf':1.5,'usdjpy':1.0}
SPLITS={'train':(2014,2018),'val':(2019,2021),'test':(2022,2024),'holdout':(2025,2026)}

def load(p):
 d=pd.read_csv(f'/tmp/{p}_reopen_h1.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 return d.dropna(subset=['open','close'])

def trades_for(p,d,threshold,costmult):
 pip=PIP[p];rows=[];li=d.index.tz_convert('America/New_York')
 # FX trading day convention: 17:00 New York rollover / new-day boundary, DST-aware.
 poss=np.flatnonzero((li.hour==17)&(li.minute==0)&(li.weekday<5))
 for i in poss:
  if i<=0:continue
  # Require prior bar to end immediately before reopen bar (1h spacing); skip missing/holiday discontinuities > 2h.
  if (d.index[i]-d.index[i-1]).total_seconds()>7200:continue
  prev_close=float(d.close.iloc[i-1]);op=float(d.open.iloc[i]);cl=float(d.close.iloc[i]);gap=(op-prev_close)/pip
  if abs(gap)<threshold:continue
  direction=-np.sign(gap);gross=direction*(cl-op)/pip;net=gross-COST[p]*costmult
  rows.append({'dt':d.index[i],'pair':p,'gap_pips':float(gap),'direction':int(direction),'gross_pips':float(gross),'net_pips':float(net)})
 return pd.DataFrame(rows)

def metric(df):
 if df is None or len(df)==0:return {'n':0}
 a=df.net_pips.values;gp=a[a>0].sum();gl=-a[a<0].sum();sd=a.std(ddof=1) if len(a)>1 else np.nan;m=df.set_index('dt').net_pips.resample('MS').sum();y=df.set_index('dt').net_pips.resample('YS').sum()
 return {'n':len(df),'trades_per_month':round(len(df)/max(1,len(m)),1),'mean_net_pips':round(float(a.mean()),3),'median_net_pips':round(float(np.median(a)),3),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((a>0).mean()*100),1),'t':round(float(a.mean()/(sd/math.sqrt(len(a)))),2) if len(a)>1 and sd>0 else None,'positive_month_pct':round(float((m>0).mean()*100),1),'positive_year_pct':round(float((y>0).mean()*100),1),'avg_gap_pips':round(float(df.gap_pips.abs().mean()),2)}

def split(df,a,b):return df[(df.dt.dt.year>=a)&(df.dt.dt.year<=b)].copy()

def portfolio(df,yr0,yr1,risk=0.004,cap=500000,max_same_day=3):
 x=split(df,yr0,yr1).sort_values(['dt','gap_pips'],ascending=[True,False]).copy()
 if len(x)==0:return {'n':0}
 # Risk unit: stop proxy = max(abs reopen gap), 5 pips floor. This is only a screening equity proxy; exact SL comes in the next phase.
 kept=[]
 for day,g in x.groupby(x.dt.dt.date):
  g=g.assign(abs_gap=g.gap_pips.abs()).sort_values('abs_gap',ascending=False).head(max_same_day)
  kept.append(g)
 x=pd.concat(kept).sort_values('dt');eq=cap;peak=cap;md=0;vals=[]
 for _,r in x.iterrows():
  stop=max(5.0,abs(float(r.gap_pips)));R=float(r.net_pips)/stop;pnl=eq*risk*R;eq+=pnl;peak=max(peak,eq);md=max(md,(peak-eq)/peak);vals.append((r.dt,pnl,R))
 q=pd.DataFrame(vals,columns=['dt','pnl','R']);m=q.set_index('dt').pnl.resample('MS').sum();m=m.reindex(pd.date_range(f'{yr0}-01-01',f'{yr1}-12-01',freq='MS',tz='UTC'),fill_value=0);a=q.R.values;gp=a[a>0].sum();gl=-a[a<0].sum()
 return {'n':len(q),'trades_per_month':round(len(q)/max(1,len(m)),1),'avg_monthly_jpy':round(float(m.mean())),'median_monthly_jpy':round(float(m.median())),'positive_month_pct':round(float((m>0).mean()*100),1),'p10_month_jpy':round(float(m.quantile(.1))),'p90_month_jpy':round(float(m.quantile(.9))),'return_pct':round(float((eq/cap-1)*100),2),'max_dd_pct':round(float(md*100),2),'avg_R':round(float(a.mean()),4),'pf_R':round(float(gp/gl),3) if gl>0 else None}

def main():
 data={p:load(p) for p in PAIRS};out={'status':'FX_REOPEN_GAP_PHASE13','source':'Dukascopy free H1','mechanism':'fade 17:00 New York daily reopen gap for one hour','paper_rule':'thresholds 1/2/5 pips; one-hour exit; no parameter optimization','warning':'H1 source may not represent executable bid/ask at rollover; any survivor requires bid/ask tick validation','results':{}}
 for cm in [1.0,1.5,2.0]:
  out['results'][str(cm)]={}
  for th in [1,2,5]:
   parts=[];pairm={}
   for p,d in data.items():
    t=trades_for(p,d,th,cm);parts.append(t)
    pairm[p]={s:metric(split(t,*ab)) for s,ab in SPLITS.items()}
   alltr=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame();pooled={s:metric(split(alltr,*ab)) for s,ab in SPLITS.items()};eq={}
   for r in [0.003,0.004,0.006,0.008]:eq[f'test_{r}']=portfolio(alltr,2022,2024,r);eq[f'holdout_{r}']=portfolio(alltr,2025,2026,r)
   out['results'][str(cm)][str(th)]={'pooled':pooled,'by_pair':pairm,'equity_proxy':eq}
 Path('tmp/fx_reopen_gap_phase13_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));
 print(json.dumps({cm:{th:{'pooled':z['pooled'],'h008':z['equity_proxy']['holdout_0.008']} for th,z in d.items()} for cm,d in out['results'].items()},indent=2,default=str))
if __name__=='__main__':main()
