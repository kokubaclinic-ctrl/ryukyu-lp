import json,math,os
from pathlib import Path
import numpy as np,pandas as pd

PAIRS=['eurusd','usdjpy','gbpusd','audusd','nzdusd','usdcad','usdchf','eurjpy','gbpjpy','audjpy','nzdjpy','cadjpy','chfjpy','eurgbp','euraud','eurcad','eurnzd','gbpaud','gbpcad','gbpnzd','audcad','audnzd','audchf','nzdcad','nzdchf','cadchf']
PIP={p:(0.01 if p.endswith('jpy') else 0.0001) for p in PAIRS}
COST={'eurusd':0.8,'usdjpy':1.0,'gbpusd':1.3,'audusd':1.1,'nzdusd':1.5,'usdcad':1.5,'usdchf':1.5,'eurjpy':1.4,'gbpjpy':2.4,'audjpy':1.3,'nzdjpy':2.0,'cadjpy':2.0,'chfjpy':2.2,'eurgbp':1.4,'euraud':2.0,'eurcad':2.0,'eurnzd':3.0,'gbpaud':3.0,'gbpcad':3.0,'gbpnzd':4.0,'audcad':2.0,'audnzd':2.5,'audchf':2.0,'nzdcad':2.5,'nzdchf':2.5,'cadchf':2.0}
SPLITS={'train':(2014,2018),'val':(2019,2021),'test':(2022,2024),'holdout':(2025,2026)}

def load_pair(p):
 f=f'/tmp/{p}_u_h1.csv'
 if not os.path.exists(f) or os.path.getsize(f)<1000:return None
 try:d=pd.read_csv(f)
 except:return None
 if 'timestamp' not in d or len(d)<10000:return None
 d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 d=d.dropna(subset=['open','high','low','close']);pip=PIP[p]
 prev=d.close.shift(1);tr=pd.concat([(d.high-d.low)/pip,(d.high-prev).abs()/pip,(d.low-prev).abs()/pip],axis=1).max(axis=1)
 d['atr']=tr.rolling(14,min_periods=10).mean().shift(1)
 r=d.close.diff()/pip;d['sig']=r.rolling(240,min_periods=120).std().shift(1)
 for lb in [24,72,120,240]:
  d[f'z{lb}']=((d.close-d.close.shift(lb))/pip)/(d.sig*np.sqrt(lb))
  d[f'hi{lb}']=d.high.rolling(lb,min_periods=lb).max().shift(1)
  d[f'lo{lb}']=d.low.rolling(lb,min_periods=lb).min().shift(1)
 return d

def direction_series(d,cfg):
 if cfg['family']=='TSMOM':
  z=d[f"z{cfg['lb']}"];return pd.Series(np.where(z.abs()>=cfg['th'],np.sign(z),0),index=d.index,dtype=float)
 hi=d[f"hi{cfg['lb']}"];lo=d[f"lo{cfg['lb']}"];c=d.close
 return pd.Series(np.where(c>hi,1,np.where(c<lo,-1,0)),index=d.index,dtype=float)

def signal_positions(d,cfg):
 dire=direction_series(d,cfg);idx=d.index
 sched=(idx.hour%6==0)&(idx.weekday<5)&~((idx.weekday==4)&(idx.hour>=12))
 arr=np.flatnonzero((dire.values!=0)&sched)
 return dire.values,arr

def splitname(dt):
 y=dt.year
 for s,(a,b) in SPLITS.items():
  if a<=y<=b:return s
 return None

def screen_pair(d,p,cfg,costmult=1.0):
 dire,pos=signal_positions(d,cfg);pip=PIP[p];cost=COST[p]*costmult;op=d.open.values;atr=d.atr.values;idx=d.index;hold=cfg['hold'];rows=[];last=-1
 N=len(d)
 for i in pos:
  ei=i+1;xi=ei+hold
  if i<=last or xi>=N:continue
  stp=max(5 if p.endswith('jpy') else 4,float(atr[i])*cfg['stop']) if np.isfinite(atr[i]) else np.nan
  if not np.isfinite(stp) or stp<=0:continue
  R=float(dire[i]*(op[xi]-op[ei])/pip/stp-cost/stp)
  rows.append((idx[ei],R));last=xi
 return rows

def metric(vals):
 a=np.asarray(vals,float);n=len(a)
 if n==0:return {'n':0,'mean_R':None,'pf':None,'win_pct':None}
 gp=a[a>0].sum();gl=-a[a<0].sum();sd=a.std(ddof=1) if n>1 else np.nan
 return {'n':int(n),'mean_R':round(float(a.mean()),4),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((a>0).mean()*100),1),'t':round(float(a.mean()/(sd/math.sqrt(n))),2) if n>1 and sd>0 else None}

def screen_config(data,cfg):
 bysplit={s:[] for s in SPLITS};pairmeans={s:[] for s in SPLITS}
 for p,d in data.items():
  rows=screen_pair(d,p,cfg,1.0)
  for s,(a,b) in SPLITS.items():
   v=[R for dt,R in rows if a<=dt.year<=b]
   if v:
    bysplit[s].extend(v);pairmeans[s].append(float(np.mean(v)))
 out=dict(cfg)
 for s in SPLITS:
  out[s]=metric(bysplit[s]);pm=pairmeans[s];out[s]['positive_pair_pct']=round(100*sum(x>0 for x in pm)/len(pm),1) if pm else 0;out[s]['pairs']=len(pm)
 tr,va=out['train'],out['val']
 if tr['n']<200 or va['n']<120 or tr['mean_R'] is None or va['mean_R'] is None or tr['mean_R']<=0 or va['mean_R']<=0 or (tr['pf'] or 0)<=1 or (va['pf'] or 0)<=1:
  out['score']=-999
 else:
  out['score']=round(min(tr['mean_R'],va['mean_R'])+0.0015*min(tr['positive_pair_pct'],va['positive_pair_pct'])-0.25*abs(tr['mean_R']-va['mean_R']),6)
 return out

def exact_trades(d,p,cfg,costmult,factor):
 dire,pos=signal_positions(d,cfg);pip=PIP[p];cost=COST[p]*costmult;op=d.open.values;hi=d.high.values;lo=d.low.values;atr=d.atr.values;idx=d.index;hold=cfg['hold'];N=len(d);rows=[];last=-1
 for i in pos:
  ei=i+1;xi=ei+hold
  if i<=last or xi>=N:continue
  stp=max(5 if p.endswith('jpy') else 4,float(atr[i])*cfg['stop']) if np.isfinite(atr[i]) else np.nan
  if not np.isfinite(stp) or stp<=0:continue
  direction=float(dire[i]);entry=float(op[ei]);sp=entry-direction*stp*pip;stopped=False
  if direction>0:stopped=bool(np.nanmin(lo[ei:xi+1])<=sp)
  else:stopped=bool(np.nanmax(hi[ei:xi+1])>=sp)
  R=(-1-cost/stp) if stopped else (direction*(float(op[xi])-entry)/pip/stp-cost/stp)
  rows.append({'dt':idx[ei],'end':idx[xi],'pair':p,'factor':factor,'direction':direction,'R':float(R),'stop_pips':float(stp)})
  last=xi
 return rows

def aggregate_exact(rows):
 out={}
 df=pd.DataFrame(rows)
 for s,(a,b) in SPLITS.items():
  if len(df)==0:out[s]=metric([]);continue
  x=df[(df.dt.dt.year>=a)&(df.dt.dt.year<=b)]
  o=metric(x.R.values);o['trades_per_month']=round(len(x)/max(1,(b-a+1)*12),1);o['positive_month_pct']=None
  if len(x):
   m=x.set_index('dt').R.resample('MS').sum();o['positive_month_pct']=round(float((m>0).mean()*100),1)
  out[s]=o
 return out

def currencies(pair,direction):
 b,q=pair[:3],pair[3:]
 return {b:direction,q:-direction}

def portfolio(rows,yr0,yr1,target_risk,cap=500000,maxopen=2):
 if not rows:return {'n':0}
 q=pd.DataFrame(rows);q=q[(q.dt.dt.year>=yr0)&(q.dt.dt.year<=yr1)].sort_values(['dt','factor','pair'])
 active=[];kept=[]
 for _,r in q.iterrows():
  t=r['dt'];active=[a for a in active if a['end']>t]
  # same pair cannot duplicate, and avoid stacking same currency risk direction
  if any(a['pair']==r['pair'] for a in active):continue
  expo=currencies(r['pair'],r['direction']);conflict=False
  for a in active:
   ae=currencies(a['pair'],a['direction'])
   for c,v in expo.items():
    if c in ae and v*ae[c]>0:conflict=True;break
   if conflict:break
  if conflict or len(active)>=maxopen:continue
  used=sum(a['risk'] for a in active);risk=min(target_risk,max(0,0.012-used))
  if risk<0.002:continue
  rec={'dt':t,'end':r['end'],'pair':r['pair'],'factor':r['factor'],'direction':r['direction'],'R':r['R'],'risk':risk};active.append(rec);kept.append(rec)
 if not kept:return {'n':0}
 df=pd.DataFrame(kept).sort_values('dt');df['pnl']=df.R*df.risk*cap
 start=pd.Timestamp(f'{yr0}-01-01',tz='UTC');end=pd.Timestamp(f'{yr1}-12-01',tz='UTC');m=df.set_index('dt').pnl.resample('MS').sum().reindex(pd.date_range(start,end,freq='MS'),fill_value=0)
 eq=cap+df.pnl.cumsum();peak=np.maximum.accumulate(eq);dd=(eq-peak)/peak;gp=df.loc[df.R>0,'R'].sum();gl=-df.loc[df.R<0,'R'].sum()
 return {'n':len(df),'trades_per_month':round(len(df)/len(m),1),'avg_actual_risk_pct':round(float(df.risk.mean()*100),3),'avg_monthly_jpy':round(float(m.mean())),'median_monthly_jpy':round(float(m.median())),'positive_month_pct':round(float((m>0).mean()*100),1),'p10_month_jpy':round(float(m.quantile(.1))),'p90_month_jpy':round(float(m.quantile(.9))),'max_dd_pct':round(float(-dd.min()*100),2),'avg_R':round(float(df.R.mean()),4),'pf_R':round(float(gp/gl),3) if gl>0 else None}

def main():
 data={}
 for p in PAIRS:
  d=load_pair(p)
  if d is not None:data[p]=d
 print('loaded',len(data),sorted(data))
 configs=[]
 for lb in [24,72,120,240]:
  for th in [.5,1.0,1.5]:
   for hold in [12,24,48]:
    for stop in [1.5,2.0]:configs.append({'family':'TSMOM','lb':lb,'th':th,'hold':hold,'stop':stop})
 for lb in [24,72,120,240]:
  for hold in [12,24,48]:
   for stop in [1.5,2.0]:configs.append({'family':'DONCHIAN','lb':lb,'hold':hold,'stop':stop})
 screens=[screen_config(data,c) for c in configs]
 chosen={}
 for fam in ['TSMOM','DONCHIAN']:
  pool=[x for x in screens if x['family']==fam];chosen[fam]=max(pool,key=lambda x:x['score'])
 print('chosen',chosen)
 base=[];stress=[];exact={}
 for fam,c in chosen.items():
  spec={k:v for k,v in c.items() if k in ['family','lb','th','hold','stop']}
  br=[];sr=[]
  for p,d in data.items():br.extend(exact_trades(d,p,spec,1.0,fam));sr.extend(exact_trades(d,p,spec,1.5,fam))
  exact[fam]={'base':aggregate_exact(br),'stress':aggregate_exact(sr)};base.extend(br);stress.extend(sr)
 ports={}
 for r in [.004,.005,.006,.007,.008]:
  ports[f'test_base_{r}']=portfolio(base,2022,2024,r);ports[f'holdout_base_{r}']=portfolio(base,2025,2026,r);ports[f'holdout_stress_{r}']=portfolio(stress,2025,2026,r)
 out={'status':'FX_UNIVERSE_PHASE5','source':'Dukascopy H1 free','loaded_pairs':sorted(data),'candidate_count':len(configs),'protocol':'one GLOBAL parameter set per factor selected only from 2014-18 train + 2019-21 validation across full universe; unchanged on all pairs in 2022-24 test and 2025-26 holdout; no post-test pair picking','cost_pips':COST,'chosen_global':chosen,'exact_factor_results':exact,'portfolio':ports,'top10_trainval':sorted(screens,key=lambda x:x['score'],reverse=True)[:10]}
 Path('tmp/fx_universe_phase5_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'loaded':len(data),'chosen':chosen,'exact':exact,'portfolio':ports},indent=2))
if __name__=='__main__':main()
