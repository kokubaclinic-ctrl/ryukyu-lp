import json,math
from pathlib import Path
import numpy as np,pandas as pd
PAIRS=['eurusd','usdjpy','gbpusd','audusd','eurjpy','gbpjpy','audjpy']
PIP={p:(0.01 if p.endswith('jpy') else 0.0001) for p in PAIRS};COST={'eurusd':0.8,'usdjpy':1.0,'gbpusd':1.3,'audusd':1.1,'eurjpy':1.4,'gbpjpy':2.4,'audjpy':1.3}
EXPERTS=[
 {'name':'TS_FAST','kind':'MOM','lb':24,'th':1.0,'hold':24,'stop':1.5,'sign':1},
 {'name':'TS_SLOW','kind':'MOM','lb':120,'th':1.25,'hold':48,'stop':2.0,'sign':1},
 {'name':'DON72','kind':'DON','lb':72,'hold':24,'stop':1.5,'sign':1},
 {'name':'PB240','kind':'PB','lb':240,'th':1.0,'pb':.75,'hold':12,'stop':1.5,'sign':1},
 {'name':'REV_FAST','kind':'MOM','lb':24,'th':1.5,'hold':12,'stop':1.5,'sign':-1},
]
def load(p):
 d=pd.read_csv(f'/tmp/{p}_trend_h1.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 d=d.dropna();pip=PIP[p];prev=d.close.shift(1);tr=pd.concat([(d.high-d.low)/pip,(d.high-prev).abs()/pip,(d.low-prev).abs()/pip],axis=1).max(axis=1);d['atr']=tr.rolling(14,min_periods=10).mean().shift(1);r=d.close.diff()/pip;d['sig']=r.rolling(240,min_periods=120).std().shift(1)
 for lb in [24,72,120,240]:d[f'z{lb}']=((d.close-d.close.shift(lb))/pip)/(d.sig*np.sqrt(lb))
 d['r12']=((d.close-d.close.shift(12))/pip)/(d.sig*np.sqrt(12));return d

def gen(d,p,e,costmult):
 pip=PIP[p];cost=COST[p]*costmult;vals=[];rows=[];i=260;N=len(d);idx=d.index
 while i<N-e['hold']-2:
  if idx[i].hour%6 or idx[i].weekday()>=5 or (idx[i].weekday()==4 and idx[i].hour>=12):i+=1;continue
  dire=0
  if e['kind']=='MOM':
   z=float(d.iloc[i][f"z{e['lb']}"]);
   if np.isfinite(z) and abs(z)>=e['th']:dire=np.sign(z)*e['sign']
  elif e['kind']=='DON':
   hi=float(d.high.iloc[i-e['lb']:i].max());lo=float(d.low.iloc[i-e['lb']:i].min());c=float(d.close.iloc[i]);dire=(1 if c>hi else (-1 if c<lo else 0))*e['sign']
  else:
   z=float(d.iloc[i][f"z{e['lb']}"]);r=float(d.iloc[i].r12)
   if np.isfinite(z) and np.isfinite(r) and abs(z)>=e['th'] and np.sign(r)==-np.sign(z) and abs(r)>=e['pb']:dire=np.sign(z)*e['sign']
  if dire==0:i+=1;continue
  ei=i+1;xi=ei+e['hold'];entry=float(d.open.iloc[ei]);atr=float(d.atr.iloc[ei]);
  if not np.isfinite(atr) or atr<=0:i+=1;continue
  stop=max(5 if p.endswith('jpy') else 4,atr*e['stop']);sp=entry-dire*stop*pip;st=False
  for k in range(ei,xi+1):
   if dire>0 and float(d.low.iloc[k])<=sp:st=True;break
   if dire<0 and float(d.high.iloc[k])>=sp:st=True;break
  R=(-1-cost/stop) if st else (dire*(float(d.open.iloc[xi])-entry)/pip/stop-cost/stop)
  rows.append({'dt':idx[ei],'end':idx[xi],'pair':p,'expert':e['name'],'R':float(R)});i=xi+1
 return pd.DataFrame(rows)

def hist_score(x,minmean):
 if len(x)<8:return None
 mean=x.R.mean();gp=x.loc[x.R>0,'R'].sum();gl=-x.loc[x.R<0,'R'].sum();pf=gp/gl if gl>0 else 99
 if mean<=minmean or pf<=1.03:return None
 sd=x.R.std(ddof=1)
 return float(mean*np.sqrt(min(len(x),100))/(sd+0.5))

def walk(expert_trades,window,topk,minmean,start,end):
 months=pd.date_range(start,end,freq='MS',tz='UTC');out=[];sel_log=[]
 for m in months:
  hist_start=m-pd.DateOffset(months=window);scores=[]
  for key,df in expert_trades.items():
   h=df[(df.dt>=hist_start)&(df.dt<m)];sc=hist_score(h,minmean)
   if sc is not None:scores.append((sc,key))
  scores.sort(reverse=True);used_pairs=set();chosen=[]
  for sc,key in scores:
   p=key[0]
   if p in used_pairs:continue
   chosen.append((sc,key));used_pairs.add(p)
   if len(chosen)>=topk:break
  sel_log.append({'month':str(m.date()),'chosen':[k for _,k in chosen]})
  candidates=[]
  nxt=m+pd.offsets.MonthBegin(1)
  for sc,key in chosen:
   z=expert_trades[key];q=z[(z.dt>=m)&(z.dt<nxt)].copy();q['rankscore']=sc;candidates.append(q)
  if not candidates:continue
  q=pd.concat(candidates).sort_values(['dt','rankscore'],ascending=[True,False]);active=[]
  for _,r in q.iterrows():
   active=[a for a in active if a>r.dt]
   if len(active)>=2:continue
   active.append(r.end);out.append(r)
 return pd.DataFrame(out),sel_log

def perf(df):
 if df is None or len(df)==0:return {'n':0,'score':-999}
 m=df.set_index('dt').R.resample('MS').sum();m=m.reindex(pd.date_range(m.index.min(),m.index.max(),freq='MS',tz='UTC'),fill_value=0);eq=df.R.cumsum();peak=np.maximum.accumulate(eq);dd=eq-peak
 return {'n':len(df),'months':len(m),'trades_per_month':round(len(df)/len(m),1),'mean_month_R':round(float(m.mean()),3),'median_month_R':round(float(m.median()),3),'positive_month_pct':round(float((m>0).mean()*100),1),'std_month_R':round(float(m.std()),3),'max_dd_R':round(float(-dd.min()),2),'score':float(m.mean()-.25*m.std())}
def yenperf(df,risk,cap=500000):
 if df is None or len(df)==0:return {'n':0}
 m=df.set_index('dt').R.resample('MS').sum()*cap*risk;m=m.reindex(pd.date_range(m.index.min(),m.index.max(),freq='MS',tz='UTC'),fill_value=0);p=df.R*cap*risk;eq=cap+p.cumsum();peak=np.maximum.accumulate(eq);dd=(eq-peak)/peak;gp=df.loc[df.R>0,'R'].sum();gl=-df.loc[df.R<0,'R'].sum()
 return {'n':len(df),'trades_per_month':round(len(df)/len(m),1),'avg_monthly_jpy':round(float(m.mean())),'median_monthly_jpy':round(float(m.median())),'positive_month_pct':round(float((m>0).mean()*100),1),'p10':round(float(m.quantile(.1))),'p90':round(float(m.quantile(.9))),'max_dd_pct':round(float(-dd.min()*100),2),'avg_R':round(float(df.R.mean()),4),'pf_R':round(float(gp/gl),3) if gl>0 else None}
def main():
 data={p:load(p) for p in PAIRS};base={};stress={}
 for p,d in data.items():
  for e in EXPERTS:
   base[(p,e['name'])]=gen(d,p,e,1.0);stress[(p,e['name'])]=gen(d,p,e,1.5)
 configs=[]
 for w in [12,24,36]:
  for k in [2,3,4]:
   for mm in [0.0,.03,.06]:
    df,_=walk(base,w,k,mm,'2019-01-01','2024-12-01');pr=perf(df);configs.append({'window':w,'topk':k,'minmean':mm,'dev':pr})
 eligible=[c for c in configs if c['dev'].get('trades_per_month',0)>=8]
 best=max(eligible,key=lambda c:c['dev']['score']) if eligible else max(configs,key=lambda c:c['dev']['score'])
 hb,log=walk(base,best['window'],best['topk'],best['minmean'],'2025-01-01','2026-07-01');hs,_=walk(stress,best['window'],best['topk'],best['minmean'],'2025-01-01','2026-07-01')
 y={}
 for r in [.004,.005,.006,.007,.008]:y[f'base_{r}']=yenperf(hb,r);y[f'stress_{r}']=yenperf(hs,r)
 out={'status':'FX_ADAPTIVE_PHASE4','protocol':'expert library fixed; selector hyperparameters chosen on 2019-2024 only; 2025-2026 holdout untouched','experts':EXPERTS,'best_config':best,'all_dev_configs':configs,'holdout_R':perf(hb),'holdout_yen':y,'selection_log':log};Path('tmp/fx_adaptive_phase4_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'best':best,'holdout_R':out['holdout_R'],'yen':y},indent=2))
if __name__=='__main__':main()
