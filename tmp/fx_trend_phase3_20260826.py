import json,math
from pathlib import Path
import numpy as np,pandas as pd

PAIRS=['eurusd','usdjpy','gbpusd','audusd','eurjpy','gbpjpy','audjpy']
PIP={p:(0.01 if p.endswith('jpy') else 0.0001) for p in PAIRS}
COST={'eurusd':0.8,'usdjpy':1.0,'gbpusd':1.3,'audusd':1.1,'eurjpy':1.4,'gbpjpy':2.4,'audjpy':1.3}
SPLITS={'train':(2014,2018),'val':(2019,2021),'test':(2022,2024),'holdout':(2025,2026)}

def load(p):
 d=pd.read_csv(f'/tmp/{p}_trend_h1.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 d=d.dropna(subset=['open','high','low','close']);pip=PIP[p]
 prev=d.close.shift(1);tr=pd.concat([(d.high-d.low)/pip,(d.high-prev).abs()/pip,(d.low-prev).abs()/pip],axis=1).max(axis=1);d['atr']=tr.rolling(14,min_periods=10).mean().shift(1)
 r=d.close.diff()/pip;d['sig1']=r.rolling(240,min_periods=120).std().shift(1)
 for lb in [24,72,120,240]:d[f'z{lb}']=((d.close-d.close.shift(lb))/pip)/(d.sig1*np.sqrt(lb))
 d['r12']=((d.close-d.close.shift(12))/pip)/(d.sig1*np.sqrt(12))
 return d

def metric(a,dates=None):
 a=np.asarray(a,float);n=len(a)
 if n==0:return {'n':0}
 gp=a[a>0].sum();gl=-a[a<0].sum();sd=a.std(ddof=1) if n>1 else np.nan;o={'n':n,'mean_R':round(float(a.mean()),4),'median_R':round(float(np.median(a)),4),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((a>0).mean()*100),1),'t':round(float(a.mean()/(sd/math.sqrt(n))),2) if n>1 and sd>0 else None,'sum_R':round(float(a.sum()),2)}
 if dates is not None and n:
  s=pd.Series(a,index=pd.DatetimeIndex(dates));m=s.groupby([s.index.year,s.index.month]).sum();y=s.groupby(s.index.year).sum();o['positive_month_pct']=round(float((m>0).mean()*100),1);o['positive_year_pct']=round(float((y>0).mean()*100),1)
 return o

def smask(dates,s):lo,hi=SPLITS[s];y=pd.DatetimeIndex(dates).year;return (y>=lo)&(y<=hi)

def run(d,p,spec,cost_mult=1.0):
 pip=PIP[p];cost=COST[p]*cost_mult;idx=d.index;vals=[];dates=[];meta=[];N=len(d);i=260
 while i<N-spec['hold']-2:
  if idx[i].hour%6!=0 or idx[i].weekday()>=5 or (idx[i].weekday()==4 and idx[i].hour>=12):i+=1;continue
  direction=0
  if spec['family']=='TSMOM':
   z=float(d.iloc[i][f"z{spec['lb']}"])
   if np.isfinite(z) and abs(z)>=spec['th']:direction=1 if z>0 else -1
  elif spec['family']=='DONCHIAN':
   lb=spec['lb'];hi=float(d.high.iloc[i-lb:i].max());lo=float(d.low.iloc[i-lb:i].min());c=float(d.close.iloc[i]);direction=1 if c>hi else (-1 if c<lo else 0)
  elif spec['family']=='PULLBACK':
   z=float(d.iloc[i][f"z{spec['lb']}"]);r12=float(d.iloc[i].r12)
   if np.isfinite(z) and np.isfinite(r12) and abs(z)>=spec['th'] and np.sign(r12)==-np.sign(z) and abs(r12)>=spec['pb']:direction=1 if z>0 else -1
  if direction==0:i+=1;continue
  ei=i+1;xi=ei+spec['hold'];entry=float(d.open.iloc[ei]);atr=float(d.atr.iloc[ei]);
  if not np.isfinite(atr) or atr<=0:i+=1;continue
  stop=max(4 if not p.endswith('jpy') else 5,atr*spec['stop']);stop_price=entry-direction*stop*pip;stopped=False
  for k in range(ei,xi+1):
   if direction>0 and float(d.low.iloc[k])<=stop_price:stopped=True;break
   if direction<0 and float(d.high.iloc[k])>=stop_price:stopped=True;break
  if stopped:R=-1-cost/stop
  else:R=direction*(float(d.open.iloc[xi])-entry)/pip/stop-cost/stop
  vals.append(R);dates.append(idx[ei]);meta.append((idx[ei],idx[xi],direction));i=xi+1
 return np.asarray(vals),pd.DatetimeIndex(dates),meta

def evalc(d,p,spec,costmult=1.0):
 vals,dates,_=run(d,p,spec,costmult);o=dict(spec);o['pair']=p
 for s in SPLITS:
  m=smask(dates,s);o[s]=metric(vals[m],dates[m])
 return o

def score(o):
 tr=o['train'];va=o['val']
 if tr.get('n',0)<120 or va.get('n',0)<70:return -1e9
 if tr.get('mean_R',-99)<=0 or va.get('mean_R',-99)<=0:return -1e9
 if (tr.get('pf') or 0)<=1 or (va.get('pf') or 0)<=1:return -1e9
 return min(tr['mean_R'],va['mean_R'])+.002*min(tr.get('positive_month_pct',0),va.get('positive_month_pct',0))

def main():
 data={p:load(p) for p in PAIRS};allc=[]
 for p,d in data.items():
  for lb in [24,72,120,240]:
   for th in [.5,1.0,1.5]:
    for hold in [12,24,48]:
     for stop in [1.5,2.0]:allc.append(evalc(d,p,{'family':'TSMOM','lb':lb,'th':th,'hold':hold,'stop':stop}))
  for lb in [24,72,120]:
   for hold in [12,24,48]:
    for stop in [1.5,2.0]:allc.append(evalc(d,p,{'family':'DONCHIAN','lb':lb,'hold':hold,'stop':stop}))
  for lb in [120,240]:
   for th in [.75,1.0]:
    for pb in [.5,.75]:
     for hold in [12,24]:
      for stop in [1.5,2.0]:allc.append(evalc(d,p,{'family':'PULLBACK','lb':lb,'th':th,'pb':pb,'hold':hold,'stop':stop}))
 # Stage1: one winner per pair+family chosen train/val only.
 famwin=[]
 for p in PAIRS:
  for fam in ['TSMOM','DONCHIAN','PULLBACK']:
   pool=sorted([x for x in allc if x['pair']==p and x['family']==fam],key=score,reverse=True)
   if pool and score(pool[0])>-1e8:famwin.append(pool[0])
 # Stage2: TEST selects at most one family per pair. HOLDOUT untouched.
 accepted={}
 for p in PAIRS:
  pool=[x for x in famwin if x['pair']==p and x['test'].get('mean_R',-99)>0 and (x['test'].get('pf') or 0)>1]
  if pool:accepted[p]=max(pool,key=lambda x:x['test']['mean_R'])
 # Create holdout trades and stress same frozen rules.
 base=[];stress=[]
 for p,x in accepted.items():
  spec={k:v for k,v in x.items() if k in ['family','lb','th','pb','hold','stop']}
  for cm,target in [(1.0,base),(1.5,stress)]:
   vals,dates,meta=run(data[p],p,spec,cm)
   for R,dt,mm in zip(vals,dates,meta):
    if dt.year>=2025:target.append((dt,mm[1],p,float(R),score(x)))
 def port(rows,risk,maxopen=2,cap=500000):
  rows=sorted(rows,key=lambda z:(z[0],-z[4]));active=[];kept=[]
  for st,en,p,R,sc in rows:
   active=[a for a in active if a[0]>st]
   if len(active)>=maxopen:continue
   active.append((en,p));kept.append((st,p,R))
  if not kept:return {'n':0}
  df=pd.DataFrame(kept,columns=['dt','pair','R']);df['pnl']=df.R*(cap*risk);m=df.set_index('dt').pnl.resample('MS').sum();m=m.reindex(pd.date_range(m.index.min(),m.index.max(),freq='MS'),fill_value=0);eq=cap+df.pnl.cumsum();peak=np.maximum.accumulate(eq);dd=(eq-peak)/peak;gp=df.loc[df.R>0,'R'].sum();gl=-df.loc[df.R<0,'R'].sum()
  return {'n':len(df),'trades_per_month':round(len(df)/len(m),1),'avg_monthly_jpy':round(float(m.mean())),'median_monthly_jpy':round(float(m.median())),'positive_month_pct':round(float((m>0).mean()*100),1),'p10':round(float(m.quantile(.1))),'p90':round(float(m.quantile(.9))),'max_dd_pct':round(float(-dd.min()*100),2),'avg_R':round(float(df.R.mean()),4),'pf_R':round(float(gp/gl),3) if gl>0 else None}
 ports={}
 for r in [.004,.005,.006,.007,.008]:ports[f'base_{r}']=port(base,r);ports[f'stress_{r}']=port(stress,r)
 out={'status':'FX_TREND_PHASE3','source':'Dukascopy H1 2014-2026','candidate_count':len(allc),'protocol':'params selected train 2014-18 + val 2019-21; family screened on test 2022-24; final holdout 2025-26 untouched','family_winners':famwin,'accepted':accepted,'holdout_portfolio':ports,'top_global':sorted(allc,key=score,reverse=True)[:30]};Path('tmp/fx_trend_phase3_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'family_winners':[(x['pair'],x['family'],round(score(x),3),x['test']['mean_R'],x['holdout']['mean_R']) for x in famwin],'accepted':{p:[x['family'],x['test']['mean_R'],x['holdout']['mean_R'],x['holdout']['pf']] for p,x in accepted.items()},'portfolio':ports},indent=2))
if __name__=='__main__':main()
