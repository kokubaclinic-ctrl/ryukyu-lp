import json,math
from pathlib import Path
import numpy as np,pandas as pd
from fx_universe_phase5_20260826 import PAIRS,PIP,COST,load_pair,currencies

SPLITS={'train':(2014,2018),'val':(2019,2021),'test':(2022,2024),'holdout':(2025,2026)}

def daily(d):
 # Research daily bars UTC; signals use completed day only, entry next day open.
 x=d[['open','high','low','close']].resample('1D').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna();return x

def indicators(x):
 x=x.copy();prev=x.close.shift(1);tr=pd.concat([x.high-x.low,(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1);x['atr20']=tr.rolling(20,min_periods=15).mean().shift(1)
 for n in [20,60,120]:x[f'mom{n}']=x.close/x.close.shift(n)-1
 for n in [10,20,55]:x[f'hi{n}']=x.high.rolling(n,min_periods=n).max().shift(1);x[f'lo{n}']=x.low.rolling(n,min_periods=n).min().shift(1)
 for n in [20,50,60,200]:x[f'ema{n}']=x.close.ewm(span=n,adjust=False,min_periods=n).mean().shift(1)
 return x

def signal(x,cfg):
 if cfg['family']=='MOM':return np.sign(x[f"mom{cfg['lb']}"]).fillna(0)
 if cfg['family']=='EMA':return np.sign(x[f"ema{cfg['fast']}"]-x[f"ema{cfg['slow']}"]).fillna(0)
 # Donchian state: persistent signal until opposite breakout; no lookahead.
 s=pd.Series(0.0,index=x.index);state=0
 for i in range(len(x)):
  c=x.close.iloc[i];hi=x[f"hi{cfg['entry']}"].iloc[i];lo=x[f"lo{cfg['entry']}"].iloc[i]
  ehi=x[f"hi{cfg['exit']}"].iloc[i] if cfg['exit'] in [10,20,55] else np.nan;elo=x[f"lo{cfg['exit']}"].iloc[i] if cfg['exit'] in [10,20,55] else np.nan
  if np.isfinite(hi) and c>hi:state=1
  elif np.isfinite(lo) and c<lo:state=-1
  elif state==1 and np.isfinite(elo) and c<elo:state=0
  elif state==-1 and np.isfinite(ehi) and c>ehi:state=0
  s.iloc[i]=state
 return s

def pair_stream(x,p,cfg,costmult=1.0):
 s=signal(x,cfg);pip=PIP[p];cost=COST[p]*costmult;rows=[];pos=0;entry=None;entry_atr=None;entry_dt=None
 # Trade only when state changes. ATR-risk normalized R realized on exit. No intraday stop at screening stage; Phase9b will add stop to survivors.
 for i in range(1,len(x)-1):
  new=int(s.iloc[i])
  if new==pos:continue
  exit_i=i+1
  if pos!=0 and entry is not None:
   px=float(x.open.iloc[exit_i]);R=pos*(px-entry)/(entry_atr if entry_atr>0 else 1)-cost*pip/(entry_atr if entry_atr>0 else 1)
   rows.append({'dt':entry_dt,'end':x.index[exit_i],'pair':p,'direction':pos,'R':float(R),'factor':cfg['name']})
   entry=None
  pos=new
  if pos!=0:
   atr=float(x.atr20.iloc[i]);
   if not np.isfinite(atr) or atr<=0:pos=0;continue
   entry=float(x.open.iloc[exit_i]);entry_atr=atr*cfg['risk_atr'];entry_dt=x.index[exit_i]
 return rows

def met(rows,a,b):
 z=[r['R'] for r in rows if a<=r['dt'].year<=b];a0=np.asarray(z,float)
 if len(a0)==0:return {'n':0}
 gp=a0[a0>0].sum();gl=-a0[a0<0].sum();sd=a0.std(ddof=1) if len(a0)>1 else np.nan
 return {'n':len(a0),'avg_R':round(float(a0.mean()),4),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((a0>0).mean()*100),1),'t':round(float(a0.mean()/(sd/math.sqrt(len(a0)))),2) if len(a0)>1 and sd>0 else None}

def score(metrics):
 tr,va=metrics['train'],metrics['val']
 if tr.get('n',0)<100 or va.get('n',0)<50 or tr.get('avg_R',-99)<=0 or va.get('avg_R',-99)<=0 or (tr.get('pf') or 0)<=1 or (va.get('pf') or 0)<=1:return -999
 return min(tr['avg_R'],va['avg_R'])-.2*abs(tr['avg_R']-va['avg_R'])

def portfolio(rows,a,b,risk,cap=500000,maxopen=4):
 q=[r for r in rows if a<=r['dt'].year<=b];q=sorted(q,key=lambda r:r['dt']);active=[];kept=[]
 for r in q:
  t=r['dt'];active=[x for x in active if x['end']>t]
  if any(x['pair']==r['pair'] for x in active):continue
  # avoid stacking same signed currency exposure; allow max one same direction per currency
  ex=currencies(r['pair'],r['direction']);bad=False
  for x in active:
   xe=currencies(x['pair'],x['direction'])
   if any(c in xe and ex[c]*xe[c]>0 for c in ex):bad=True;break
  if bad or len(active)>=maxopen:continue
  used=sum(x['risk'] for x in active);rr=min(risk,max(0,.02-used))
  if rr<.002:continue
  y=dict(r);y['risk']=rr;kept.append(y);active.append(y)
 if not kept:return {'n':0}
 df=pd.DataFrame(kept);df['pnl']=df.R*df.risk*cap;m=df.set_index('end').pnl.resample('MS').sum();m=m.reindex(pd.date_range(f'{a}-01-01',f'{b}-12-01',freq='MS',tz='UTC'),fill_value=0);eq=cap+df.pnl.cumsum();pk=np.maximum.accumulate(eq);dd=(eq-pk)/pk;gp=df.loc[df.R>0,'R'].sum();gl=-df.loc[df.R<0,'R'].sum()
 return {'n':len(df),'entries_per_month':round(len(df)/len(m),1),'avg_monthly_jpy':round(float(m.mean())),'median_monthly_jpy':round(float(m.median())),'positive_month_pct':round(float((m>0).mean()*100),1),'p10':round(float(m.quantile(.1))),'p90':round(float(m.quantile(.9))),'max_dd_pct':round(float(-dd.min()*100),2),'avg_R':round(float(df.R.mean()),4),'pf':round(float(gp/gl),3) if gl>0 else None}

def main():
 data={};
 for p in PAIRS:
  d=load_pair(p)
  if d is not None:data[p]=indicators(daily(d))
 cfgs=[]
 for lb in [20,60,120]:
  for ra in [1.5,2.0,2.5]:cfgs.append({'name':f'MOM{lb}_R{ra}','family':'MOM','lb':lb,'risk_atr':ra})
 for fast,slow in [(20,60),(50,200)]:
  for ra in [1.5,2.0,2.5]:cfgs.append({'name':f'EMA{fast}_{slow}_R{ra}','family':'EMA','fast':fast,'slow':slow,'risk_atr':ra})
 for ent,ex in [(20,10),(55,20)]:
  for ra in [1.5,2.0,2.5]:cfgs.append({'name':f'DON{ent}_{ex}_R{ra}','family':'DON','entry':ent,'exit':ex,'risk_atr':ra})
 res=[]
 for c in cfgs:
  rows=[]
  for p,x in data.items():rows.extend(pair_stream(x,p,c,1.0))
  mm={s:met(rows,a,b) for s,(a,b) in SPLITS.items()};res.append({'cfg':c,'metrics':mm,'score':score(mm),'rows':rows})
 # choose one per family on train+val only
 chosen=[]
 for fam in ['MOM','EMA','DON']:
  z=[r for r in res if r['cfg']['family']==fam];chosen.append(max(z,key=lambda r:r['score']))
 # Include a factor only if train/val valid AND 2022-24 positive. Holdout untouched.
 base=[];summary=[]
 for r in chosen:
  ok=r['score']>-900 and r['metrics']['test'].get('avg_R',-99)>0 and (r['metrics']['test'].get('pf') or 0)>1
  summary.append({'cfg':r['cfg'],'metrics':r['metrics'],'score':r['score'],'test_pass':ok})
  if ok:base.extend(r['rows'])
 ports={}
 for risk in [.004,.006,.008,.01]:ports[f'test_{risk}']=portfolio(base,2022,2024,risk);ports[f'holdout_{risk}']=portfolio(base,2025,2026,risk)
 out={'status':'FX_LONGTREND_PHASE9','source':'Dukascopy H1->daily free','protocol':'GLOBAL daily trend factors selected 2014-21 across 26 pairs; factor gate 2022-24; 2025-26 untouched; entries only on state change','loaded_pairs':len(data),'candidate_count':len(cfgs),'chosen':[{'cfg':r['cfg'],'metrics':r['metrics'],'score':r['score']} for r in chosen],'selected':summary,'portfolio':ports}
 Path('tmp/fx_longtrend_phase9_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'chosen':out['chosen'],'selected':summary,'portfolio':ports},indent=2,default=str))
if __name__=='__main__':main()
