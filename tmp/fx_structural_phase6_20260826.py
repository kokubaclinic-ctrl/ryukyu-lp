import json,math
from pathlib import Path
import numpy as np,pandas as pd
from fx_universe_phase5_20260826 import PAIRS,PIP,COST,SPLITS,load_pair,metric,currencies,portfolio

TZH={'eur':('Europe/Berlin',8),'gbp':('Europe/London',8),'aud':('Australia/Sydney',9),'nzd':('Pacific/Auckland',9),'jpy':('Asia/Tokyo',9),'cad':('America/Toronto',8),'chf':('Europe/Zurich',8),'usd':('America/New_York',8)}
SESS={'LONDON':('Europe/London',8),'NEWYORK':('America/New_York',8),'TOKYO':('Asia/Tokyo',9),'SYDNEY':('Australia/Sydney',9)}

def addx(d):
 d=d.copy();d['mom5d']=d.close/d.close.shift(120)-1;return d

def split_metric(rows):
 df=pd.DataFrame(rows);out={}
 for s,(a,b) in SPLITS.items():
  if len(df)==0:out[s]=metric([]);continue
  x=df[(df.dt.dt.year>=a)&(df.dt.dt.year<=b)];o=metric(x.R.values);o['positive_month_pct']=None;o['trades_per_month']=round(len(x)/((b-a+1)*12),1)
  if len(x):o['positive_month_pct']=round(float((x.set_index('dt').R.resample('MS').sum()>0).mean()*100),1)
  out[s]=o
 return out

def score(m):
 tr,va=m['train'],m['val']
 if tr.get('n',0)<300 or va.get('n',0)<180 or tr.get('mean_R') is None or va.get('mean_R') is None:return -999
 if tr['mean_R']<=0 or va['mean_R']<=0 or (tr.get('pf') or 0)<=1 or (va.get('pf') or 0)<=1:return -999
 return min(tr['mean_R'],va['mean_R'])-0.25*abs(tr['mean_R']-va['mean_R'])

def event_trade(d,p,i,direction,hold,stopmult,costmult,factor):
 ei=i+1;xi=ei+hold
 if xi>=len(d):return None
 pip=PIP[p];atr=float(d.atr.iloc[i]);
 if not np.isfinite(atr) or atr<=0:return None
 stp=max(5 if p.endswith('jpy') else 4,atr*stopmult);entry=float(d.open.iloc[ei]);sp=entry-direction*stp*pip
 lo=d.low.iloc[ei:xi+1].min();hi=d.high.iloc[ei:xi+1].max();st=(direction>0 and lo<=sp) or (direction<0 and hi>=sp);cost=COST[p]*costmult
 R=(-1-cost/stp) if st else direction*(float(d.open.iloc[xi])-entry)/pip/stp-cost/stp
 return {'dt':d.index[ei],'end':d.index[xi],'pair':p,'factor':factor,'direction':float(direction),'R':float(R)}

def local_rows(data,cfg,costmult=1.0):
 rows=[]
 for p,d0 in data.items():
  d=addx(d0);base,quote=p[:3],p[3:]
  for cur in [base,quote]:
   if cur not in TZH:continue
   tz,h=TZH[cur];li=d.index.tz_convert(tz);mask=(li.hour==h)&(li.minute==0)&(li.weekday<5);poss=np.flatnonzero(mask);direction=-1 if cur==base else 1;last=-1
   for i in poss:
    if i<=last:continue
    if cfg['filter']=='MOM':
     m=float(d.mom5d.iloc[i]);
     if not np.isfinite(m) or np.sign(m)!=direction:continue
    r=event_trade(d,p,i,direction,cfg['hold'],cfg['stop'],costmult,'LOCAL')
    if r is not None:rows.append(r);last=i+1+cfg['hold']
 return rows

def session_rows(data,cfg,costmult=1.0):
 rows=[];tz,h=SESS[cfg['session']]
 for p,d0 in data.items():
  d=addx(d0);li=d.index.tz_convert(tz);mask=(li.hour==h)&(li.minute==0)&(li.weekday<5);poss=np.flatnonzero(mask);last=-1
  for i in poss:
   if i<=last:continue
   imp=float(d.close.iloc[i]-d.open.iloc[i]);direction=np.sign(imp)*(1 if cfg['mode']=='CONT' else -1)
   if direction==0:continue
   if cfg['filter']=='MOM':
    m=float(d.mom5d.iloc[i]);
    if not np.isfinite(m) or np.sign(m)!=direction:continue
   r=event_trade(d,p,i,direction,cfg['hold'],cfg['stop'],costmult,cfg['session'])
   if r is not None:rows.append(r);last=i+1+cfg['hold']
 return rows

def pair_for(a,b,data):
 p=(a+b).lower()
 if p in data:return p,1
 q=(b+a).lower()
 if q in data:return q,-1
 return None,None

def cs_rows(data,cfg,costmult=1.0):
 req=['eurusd','gbpusd','audusd','nzdusd','usdjpy','usdcad','usdchf']
 if any(p not in data for p in req):return []
 f=pd.concat({
  'eur':np.log(data['eurusd'].close),'gbp':np.log(data['gbpusd'].close),'aud':np.log(data['audusd'].close),'nzd':np.log(data['nzdusd'].close),
  'jpy':-np.log(data['usdjpy'].close),'cad':-np.log(data['usdcad'].close),'chf':-np.log(data['usdchf'].close)},axis=1,join='inner').dropna();f['usd']=0.0
 stren=f-f.shift(cfg['lb']);idx=f.index;mask=(idx.hour==0)&(idx.minute==0)&(idx.weekday<5)
 if cfg['freq']=='WEEKLY':mask=mask&(idx.weekday==0)
 poss=np.flatnonzero(mask);rows=[];active_until=None
 for j in poss:
  t=idx[j]
  if active_until is not None and t<=active_until:continue
  s=stren.iloc[j].dropna();
  if len(s)<8:continue
  a=s.idxmax();b=s.idxmin();p,orient=pair_for(a,b,data)
  if p is None:continue
  d=data[p];i=d.index.searchsorted(t)
  if i>=len(d) or abs((d.index[i]-t).total_seconds())>3600:continue
  direction=orient
  r=event_trade(d,p,i,direction,cfg['hold'],cfg['stop'],costmult,'CSMOM')
  if r is not None:rows.append(r);active_until=r['end']
 return rows

def choose(cands,fn,data):
 res=[]
 for c in cands:
  rows=fn(data,c,1.0);m=split_metric(rows);res.append({'cfg':c,'metric':m,'score':score(m)})
 return max(res,key=lambda x:x['score']),res

def main():
 data={}
 for p in PAIRS:
  d=load_pair(p)
  if d is not None:data[p]=d
 print('loaded',len(data))
 localc=[{'hold':h,'filter':f,'stop':s} for h in [2,4,6] for f in ['NONE','MOM'] for s in [1.5,2.0,2.5]]
 bestlocal,alllocal=choose(localc,local_rows,data)
 sessions={};allsess={}
 for se in SESS:
  cc=[{'session':se,'mode':mo,'hold':h,'filter':f,'stop':s} for mo in ['CONT','FADE'] for h in [2,4,6] for f in ['NONE','MOM'] for s in [1.5,2.0]]
  sessions[se],allsess[se]=choose(cc,session_rows,data)
 csc=[]
 for lb in [120,240,480]:
  for freq in ['DAILY','WEEKLY']:
   holds=[24,48] if freq=='DAILY' else [120]
   for h in holds:
    for st in [1.5,2.0]:csc.append({'lb':lb,'freq':freq,'hold':h,'stop':st})
 bestcs,allcs=choose(csc,cs_rows,data)
 # Factor inclusion decided only on 2022-24 TEST; holdout remains final evaluation.
 selected=[]
 for name,best,fn in [('LOCAL',bestlocal,local_rows)]+[(se,sessions[se],session_rows) for se in SESS]+[('CSMOM',bestcs,cs_rows)]:
  base=fn(data,best['cfg'],1.0);stress=fn(data,best['cfg'],1.5);bm=split_metric(base);sm=split_metric(stress);passed=(bm['test'].get('mean_R') or -99)>0 and (bm['test'].get('pf') or 0)>1
  selected.append({'name':name,'best':best,'base_metric':bm,'stress_metric':sm,'test_pass':passed,'base_rows':base,'stress_rows':stress})
 base=[];stress=[]
 for x in selected:
  if x['test_pass']:base.extend(x['base_rows']);stress.extend(x['stress_rows'])
 ports={}
 for r in [.004,.005,.006,.007,.008]:
  ports[f'test_base_{r}']=portfolio(base,2022,2024,r);ports[f'holdout_base_{r}']=portfolio(base,2025,2026,r);ports[f'holdout_stress_{r}']=portfolio(stress,2025,2026,r)
 def slim(x):return {k:v for k,v in x.items() if k not in ['base_rows','stress_rows']}
 out={'status':'FX_STRUCTURAL_PHASE6','source':'Dukascopy H1 free','protocol':'GLOBAL config per structural factor selected 2014-21; factor inclusion by 2022-24 TEST; unchanged 2025-26 holdout','loaded_pairs':sorted(data),'best_local':bestlocal,'best_sessions':sessions,'best_csmom':bestcs,'selected_factors':[slim(x) for x in selected],'portfolio':ports}
 Path('tmp/fx_structural_phase6_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'bestlocal':bestlocal,'sessions':sessions,'bestcs':bestcs,'factors':[(x['name'],x['test_pass'],x['base_metric']['test'],x['base_metric']['holdout']) for x in selected],'portfolio':ports},indent=2,default=str))
if __name__=='__main__':main()
