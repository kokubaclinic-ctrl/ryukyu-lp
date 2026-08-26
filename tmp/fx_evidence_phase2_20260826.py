import json,math
from pathlib import Path
import numpy as np,pandas as pd

PAIRS=['eurusd','usdjpy','gbpusd','audusd','eurjpy','gbpjpy','audjpy']
PIP={p:(0.01 if p.endswith('jpy') else 0.0001) for p in PAIRS}
COST={'eurusd':0.8,'usdjpy':1.0,'gbpusd':1.3,'audusd':1.1,'eurjpy':1.4,'gbpjpy':2.4,'audjpy':1.3}
SPLITS={'train':(2014,2018),'val':(2019,2021),'test':(2022,2024),'holdout':(2025,2026)}
CUR_TZ={'eur':('Europe/Berlin',8,0),'gbp':('Europe/London',8,0),'aud':('Australia/Sydney',9,0),'jpy':('Asia/Tokyo',9,0),'usd':('America/New_York',8,0)}

def load(p):
 d=pd.read_csv(f'/tmp/{p}_ev_m5.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 d=d.dropna(subset=['open','high','low','close']); pip=PIP[p]
 r=d.close.diff()/pip; d['rv240']=np.sqrt((r*r).rolling(48,min_periods=36).sum()).shift(1)
 d['mom5d']=d.close/d.close.shift(12*24*5)-1
 return d

def at(idx,t,maxmin=6):
 i=idx.searchsorted(t)
 if i>=len(idx) or (idx[i]-t).total_seconds()>maxmin*60:return None
 return i

def metric(a,dates=None):
 a=np.asarray(a,float);n=len(a)
 if not n:return {'n':0}
 gp=a[a>0].sum();gl=-a[a<0].sum();sd=a.std(ddof=1) if n>1 else np.nan
 o={'n':n,'mean':round(float(a.mean()),4),'median':round(float(np.median(a)),4),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((a>0).mean()*100),1),'t':round(float(a.mean()/(sd/math.sqrt(n))),2) if n>1 and sd>0 else None,'sum':round(float(a.sum()),2)}
 if dates is not None:
  s=pd.Series(a,index=pd.DatetimeIndex(dates));m=s.groupby([s.index.year,s.index.month]).sum();y=s.groupby(s.index.year).sum();o['positive_month_pct']=round(float((m>0).mean()*100),1);o['positive_year_pct']=round(float((y>0).mean()*100),1)
 return o

def smask(dates,s):
 lo,hi=SPLITS[s];y=pd.DatetimeIndex(dates).year;return (y>=lo)&(y<=hi)

def trade_path(d,p,entry_i,exit_i,direction,costmult=1.0,stop_mult=None):
 pip=PIP[p];cost=COST[p]*costmult;entry=float(d.iloc[entry_i].open);exitp=float(d.iloc[exit_i].open);gross=direction*(exitp-entry)/pip
 if stop_mult is None:return gross-cost
 rv=float(d.iloc[entry_i].rv240)
 if not np.isfinite(rv) or rv<=0:return None
 floor=5 if p.endswith('jpy') else 4;stop=max(floor,rv*stop_mult);stop_price=entry-direction*stop*pip
 for k in range(entry_i,min(exit_i+1,len(d))):
  b=d.iloc[k]
  if direction>0 and float(b.low)<=stop_price:return -1-cost/stop
  if direction<0 and float(b.high)>=stop_price:return -1-cost/stop
 return gross/stop-cost/stop

def dates_for(d,tz,h,m):
 out=[]
 for day in pd.date_range(d.index.min().date(),d.index.max().date(),freq='D'):
  t=(pd.Timestamp(day.date()).tz_localize(tz)+pd.Timedelta(hours=h,minutes=m)).tz_convert('UTC')
  if t.weekday()<5:out.append(t)
 return out

def run_candidate(d,p,c,costmult=1.0,stop_mult=None):
 vals=[];dates=[];meta=[];idx=d.index
 if c['family']=='LONDON_OPEN':
  for t0 in dates_for(d,'Europe/London',8,0):
   a=at(idx,t0);b=at(idx,t0+pd.Timedelta(minutes=30));e=at(idx,t0+pd.Timedelta(minutes=30+c['hold']))
   if None in (a,b,e):continue
   move=float(d.iloc[b].open-d.iloc[a].open);direction=np.sign(move)*(1 if c['mode']=='CONT' else -1)
   if direction==0:continue
   if c['filter']=='MOM' and np.sign(float(d.iloc[b].mom5d))!=direction:continue
   v=trade_path(d,p,b,e,direction,costmult,stop_mult)
   if v is not None:vals.append(v);dates.append(idx[b]);meta.append((idx[b],idx[e],direction))
 elif c['family']=='LOCAL':
  cur=c['currency'];tz,h,m=CUR_TZ[cur]
  base=p[:3];quote=p[3:];direction=-1 if cur==base else 1 # local currency depreciation
  for t0 in dates_for(d,tz,h,m):
   a=at(idx,t0);e=at(idx,t0+pd.Timedelta(minutes=c['hold']))
   if None in (a,e):continue
   if c['filter']=='MOM' and np.sign(float(d.iloc[a].mom5d))!=direction:continue
   v=trade_path(d,p,a,e,direction,costmult,stop_mult)
   if v is not None:vals.append(v);dates.append(idx[a]);meta.append((idx[a],idx[e],direction))
 return np.asarray(vals),pd.DatetimeIndex(dates),meta

def evalc(d,p,c,costmult=1.0,stop_mult=None):
 vals,dates,_=run_candidate(d,p,c,costmult,stop_mult);o=dict(c);o['pair']=p;o['cost_mult']=costmult;o['stop_mult']=stop_mult
 for s in SPLITS:
  m=smask(dates,s);o[s]=metric(vals[m],dates[m])
 return o

def score(o):
 tr=o['train'];va=o['val']
 if tr.get('n',0)<500 or va.get('n',0)<280:return -1e9
 if tr.get('mean',-99)<=0 or va.get('mean',-99)<=0:return -1e9
 if (tr.get('pf') or 0)<=1 or (va.get('pf') or 0)<=1:return -1e9
 return min(tr['mean'],va['mean'])+.002*min(tr.get('positive_month_pct',0),va.get('positive_month_pct',0))

def main():
 data={p:load(p) for p in PAIRS};raw=[]
 for p,d in data.items():
  for mode in ['CONT','FADE']:
   for hold in [30,60,90,120]:
    for filt in ['NONE','MOM']:raw.append(evalc(d,p,{'family':'LONDON_OPEN','mode':mode,'hold':hold,'filter':filt}))
  for cur in [p[:3],p[3:]]:
   for hold in [120,240,360]:
    for filt in ['NONE','MOM']:raw.append(evalc(d,p,{'family':'LOCAL','currency':cur,'hold':hold,'filter':filt}))
 # Select one raw family per pair using train+validation only.
 selected={}
 for p in PAIRS:
  r=sorted([x for x in raw if x['pair']==p],key=score,reverse=True);selected[p]=r[0] if score(r[0])>-1e8 else None
 # Choose stop width only on train+validation, freeze; TEST then screens pair, HOLDOUT remains untouched.
 frozen={};accepted={}
 for p,c in selected.items():
  if c is None:continue
  grid=[]
  for sm in [1.0,1.5,2.0,2.5]:grid.append(evalc(data[p],p,{k:v for k,v in c.items() if k in ['family','mode','hold','filter','currency']},1.0,sm))
  g=sorted(grid,key=score,reverse=True);best=g[0]
  frozen[p]={'raw':c,'risk':best,'grid':grid}
  if score(best)>-1e8 and best['test'].get('mean',-99)>0 and (best['test'].get('pf') or 0)>1:accepted[p]=best
 # Honest final holdout portfolio: accepted based only through TEST, then measure 2025-26.
 trades=[];stress=[]
 for p,best in accepted.items():
  spec={k:v for k,v in best.items() if k in ['family','mode','hold','filter','currency']}
  vals,dates,meta=run_candidate(data[p],p,spec,1.0,best['stop_mult']);sv,sdates,smeta=run_candidate(data[p],p,spec,1.5,best['stop_mult'])
  for R,dt,mm in zip(vals,dates,meta):
   if dt.year>=2025:trades.append((dt,mm[1],p,float(R)))
  for R,dt,mm in zip(sv,sdates,smeta):
   if dt.year>=2025:stress.append((dt,mm[1],p,float(R)))
 def portfolio(rows,risk=0.004,cap=500000,max_open=2):
  rows=sorted(rows,key=lambda x:x[0]);active=[];kept=[]
  for st,en,p,R in rows:
   active=[x for x in active if x[0]>st]
   if len(active)>=max_open:continue
   active.append((en,p));kept.append((st,p,R))
  if not kept:return {'n':0}
  df=pd.DataFrame(kept,columns=['dt','pair','R']).sort_values('dt');df['pnl']=df.R*(cap*risk);m=df.set_index('dt').pnl.resample('MS').sum();m=m.reindex(pd.date_range(m.index.min(),m.index.max(),freq='MS'),fill_value=0);eq=cap+df.pnl.cumsum();peak=np.maximum.accumulate(eq);dd=(eq-peak)/peak
  gp=df.loc[df.R>0,'R'].sum();gl=-df.loc[df.R<0,'R'].sum()
  return {'n':len(df),'trades_per_month':round(len(df)/len(m),1),'avg_monthly_jpy':round(float(m.mean())),'median_monthly_jpy':round(float(m.median())),'positive_month_pct':round(float((m>0).mean()*100),1),'p10':round(float(m.quantile(.1))),'p90':round(float(m.quantile(.9))),'max_dd_pct':round(float(-dd.min()*100),2),'avg_R':round(float(df.R.mean()),4),'pf_R':round(float(gp/gl),3) if gl>0 else None}
 ports={}
 for r in [.003,.004,.005,.006]:ports[f'base_{r}']=portfolio(trades,r);ports[f'stress_{r}']=portfolio(stress,r)
 out={'status':'FX_EVIDENCE_PHASE2','source':'Dukascopy M5 2014-2026','candidate_count':len(raw),'protocol':'raw+stop selected train/val; pair acceptance requires positive 2022-24 TEST; 2025-26 HOLDOUT untouched','selected':selected,'frozen':frozen,'accepted':accepted,'holdout_portfolio':ports,'top_by_pair':{p:sorted([x for x in raw if x['pair']==p],key=score,reverse=True)[:5] for p in PAIRS}}
 Path('tmp/fx_evidence_phase2_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'selected':{p:(None if c is None else [c['family'],c.get('mode'),c.get('currency'),c['hold'],c['filter'],c['train']['mean'],c['val']['mean'],c['test']['mean'],c['holdout']['mean']]) for p,c in selected.items()},'accepted':{p:[x['family'],x.get('mode'),x.get('currency'),x['hold'],x['filter'],x['stop_mult'],x['test']['mean'],x['holdout']['mean']] for p,x in accepted.items()},'portfolio':ports},indent=2))
if __name__=='__main__':main()
