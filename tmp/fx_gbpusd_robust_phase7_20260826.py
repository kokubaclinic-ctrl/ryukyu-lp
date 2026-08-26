import json,math
from pathlib import Path
import numpy as np,pandas as pd
PIP=0.0001;COST=1.3
CENTRAL={'lb':120,'th':1.5,'hold':48,'stop':2.0}

def load():
 d=pd.read_csv('/tmp/gbpusd_robust_h1.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 d=d.dropna();prev=d.close.shift(1);tr=pd.concat([(d.high-d.low)/PIP,(d.high-prev).abs()/PIP,(d.low-prev).abs()/PIP],axis=1).max(axis=1);d['atr']=tr.rolling(14,min_periods=10).mean().shift(1);r=d.close.diff()/PIP;d['sig']=r.rolling(240,min_periods=120).std().shift(1)
 for lb in [72,120,240]:d[f'z{lb}']=((d.close-d.close.shift(lb))/PIP)/(d.sig*np.sqrt(lb))
 return d

def trades(d,cfg,costmult=1.0):
 z=d[f"z{cfg['lb']}"];dire=np.where(z.abs()>=cfg['th'],np.sign(z),0);idx=d.index;sched=(idx.hour%6==0)&(idx.weekday<5)&~((idx.weekday==4)&(idx.hour>=12));pos=np.flatnonzero((dire!=0)&sched);rows=[];last=-1
 for i in pos:
  ei=i+1;xi=ei+cfg['hold']
  if i<=last or xi>=len(d):continue
  atr=float(d.atr.iloc[i]);
  if not np.isfinite(atr) or atr<=0:continue
  stp=max(4,atr*cfg['stop']);direction=float(dire[i]);entry=float(d.open.iloc[ei]);sp=entry-direction*stp*PIP
  st=(direction>0 and d.low.iloc[ei:xi+1].min()<=sp) or (direction<0 and d.high.iloc[ei:xi+1].max()>=sp);cost=COST*costmult
  R=(-1-cost/stp) if st else direction*(float(d.open.iloc[xi])-entry)/PIP/stp-cost/stp
  rows.append({'dt':idx[ei],'end':idx[xi],'R':float(R),'stop_pips':stp});last=xi
 return pd.DataFrame(rows)

def met(x):
 if x is None or len(x)==0:return {'n':0}
 a=x.R.values;gp=a[a>0].sum();gl=-a[a<0].sum();sd=a.std(ddof=1) if len(a)>1 else np.nan;m=x.set_index('dt').R.resample('MS').sum();eq=np.cumsum(a);peak=np.maximum.accumulate(eq);dd=eq-peak
 return {'n':len(a),'trades_per_month':round(len(a)/max(1,len(m)),2),'avg_R':round(float(a.mean()),4),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((a>0).mean()*100),1),'t':round(float(a.mean()/(sd/math.sqrt(len(a)))),2) if len(a)>1 and sd>0 else None,'positive_month_pct':round(float((m>0).mean()*100),1),'max_dd_R':round(float(-dd.min()),2)}

def splits(df):
 return {'train':met(df[(df.dt.dt.year>=2014)&(df.dt.dt.year<=2018)]),'val':met(df[(df.dt.dt.year>=2019)&(df.dt.dt.year<=2021)]),'test':met(df[(df.dt.dt.year>=2022)&(df.dt.dt.year<=2024)]),'holdout':met(df[df.dt.dt.year>=2025])}

def monte(df,risk,nsim=10000,months=24,seed=42):
 rng=np.random.default_rng(seed);df=df.copy();df['month']=df.dt.dt.to_period('M').astype(str);blocks=[g.R.values for _,g in df.groupby('month')]
 if not blocks:return {}
 ends=[];dds=[]
 for _ in range(nsim):
  eq=1.0;peak=1.0;md=0
  for j in rng.integers(0,len(blocks),size=months):
   for R in blocks[j]:
    eq*=max(0.01,1+risk*R);peak=max(peak,eq);md=max(md,(peak-eq)/peak)
  ends.append(eq-1);dds.append(md)
 return {'risk_pct':risk*100,'months':months,'median_return_pct':round(float(np.median(ends)*100),2),'p10_return_pct':round(float(np.quantile(ends,.1)*100),2),'p90_return_pct':round(float(np.quantile(ends,.9)*100),2),'prob_positive_pct':round(float((np.array(ends)>0).mean()*100),1),'median_maxdd_pct':round(float(np.median(dds)*100),2),'p95_maxdd_pct':round(float(np.quantile(dds,.95)*100),2),'prob_dd_gt15_pct':round(float((np.array(dds)>.15).mean()*100),1)}

def main():
 d=load();central={}
 for cm in [1.0,1.5,2.0]:
  x=trades(d,CENTRAL,cm);central[str(cm)]={'splits':splits(x),'full':met(x)}
 # Neighbor grid is robustness only; central is NEVER replaced based on these results.
 grid=[]
 for lb in [72,120,240]:
  for th in [1.25,1.5,1.75]:
   for hold in [24,48,72]:
    for stop in [1.5,2.0,2.5]:
     cfg={'lb':lb,'th':th,'hold':hold,'stop':stop};x=trades(d,cfg,1.0);sp=splits(x);grid.append({'cfg':cfg,'test':sp['test'],'holdout':sp['holdout'],'both_positive':sp['test'].get('avg_R',-99)>0 and sp['holdout'].get('avg_R',-99)>0})
 c=trades(d,CENTRAL,1.0);post=c[c.dt.dt.year>=2022].copy().sort_values('R',ascending=False)
 removals={}
 for n in [1,3,5,10]:removals[f'remove_top_{n}']=met(post.iloc[n:].sort_values('dt')) if len(post)>n else {'n':0}
 yearly={str(y):met(c[c.dt.dt.year==y]) for y in range(2014,2027)}
 mc={str(r):monte(post,r) for r in [.004,.006,.008]}
 out={'status':'GBPUSD_ROBUST_PHASE7','source':'Dukascopy H1 free','central_config':CENTRAL,'note':'neighbors are robustness audit only; no re-selection after holdout','central_cost_stress':central,'neighbor_count':len(grid),'neighbor_both_positive_count':sum(x['both_positive'] for x in grid),'neighbor_both_positive_pct':round(100*sum(x['both_positive'] for x in grid)/len(grid),1),'neighbor_grid':grid,'top_profit_removal_2022_26':removals,'yearly':yearly,'monte_carlo_month_block_2022_26':mc}
 Path('tmp/fx_gbpusd_robust_phase7_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'central':central,'neighbor_positive_pct':out['neighbor_both_positive_pct'],'removals':removals,'mc':mc},indent=2))
if __name__=='__main__':main()
