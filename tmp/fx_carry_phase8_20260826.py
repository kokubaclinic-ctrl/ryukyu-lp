import json,math,requests,io
from pathlib import Path
import numpy as np,pandas as pd
from fx_universe_phase5_20260826 import PAIRS,PIP,COST,SPLITS,load_pair,metric,portfolio
CURS=['usd','eur','gbp','aud','cad','jpy']
SERIES={'usd':'IRSTCI01USM156N','eur':'IRSTCI01EZM156N','gbp':'IRSTCI01GBM156N','aud':'IRSTCI01AUM156N','cad':'IRSTCI01CAM156N','jpy':'IRSTCI01JPM156N'}

def rates():
 out={}
 for c,sid in SERIES.items():
  u=f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}'
  r=requests.get(u,timeout=30);r.raise_for_status();x=pd.read_csv(io.StringIO(r.text));x.columns=['date','rate'];x['date']=pd.to_datetime(x.date,utc=True);x['rate']=pd.to_numeric(x.rate,errors='coerce');x=x.dropna()
  # Conservative publication lag: observation for month M is usable only from first day of M+2.
  x['available']=x['date']+pd.offsets.MonthBegin(2);out[c]=x.set_index('available').rate.sort_index()
 return out

def pair_for(a,b,data):
 p=a+b
 if p in data:return p,1
 q=b+a
 if q in data:return q,-1
 return None,None

def rate_at(rs,c,t):
 s=rs[c];z=s[s.index<=t]
 return None if len(z)==0 else float(z.iloc[-1])

def addm(d):
 d=d.copy();d['mom5d']=d.close/d.close.shift(120)-1;d['mom20d']=d.close/d.close.shift(480)-1;return d

def make_rows(data,rs,cfg,costmult=1.0):
 rows=[];dates=pd.date_range('2014-03-03','2026-07-27',freq='W-MON',tz='UTC') if cfg['freq']=='WEEKLY' else pd.date_range('2014-03-03','2026-07-31',freq='B',tz='UTC')
 active_until=None
 for t in dates:
  if active_until is not None and t<=active_until:continue
  rr={c:rate_at(rs,c,t) for c in CURS};rr={c:v for c,v in rr.items() if v is not None and np.isfinite(v)}
  if len(rr)<6:continue
  hi=max(rr,key=rr.get);lo=min(rr,key=rr.get);diff=rr[hi]-rr[lo]
  if diff<cfg['mindiff']:continue
  p,orient=pair_for(hi,lo,data)
  if p is None:continue
  d=data[p];i=d.index.searchsorted(t)
  if i>=len(d) or abs((d.index[i]-t).total_seconds())>7200:continue
  direction=float(orient)
  if cfg['mom']!='NONE':
   col='mom5d' if cfg['mom']=='MOM5' else 'mom20d';m=float(d.iloc[i][col])
   if not np.isfinite(m) or np.sign(m)!=direction:continue
  ei=i+1;xi=ei+cfg['hold']
  if xi>=len(d):continue
  atr=float(d.atr.iloc[i]);pip=PIP[p]
  if not np.isfinite(atr) or atr<=0:continue
  stp=max(5 if p.endswith('jpy') else 4,atr*cfg['stop']);entry=float(d.open.iloc[ei]);sp=entry-direction*stp*pip
  st=(direction>0 and d.low.iloc[ei:xi+1].min()<=sp) or (direction<0 and d.high.iloc[ei:xi+1].max()>=sp);cost=COST[p]*costmult
  R=(-1-cost/stp) if st else direction*(float(d.open.iloc[xi])-entry)/pip/stp-cost/stp
  rows.append({'dt':d.index[ei],'end':d.index[xi],'pair':p,'factor':'CARRY','direction':direction,'R':float(R),'rate_diff':diff,'hi':hi,'lo':lo});active_until=d.index[xi]
 return rows

def sm(rows):
 df=pd.DataFrame(rows);out={}
 for s,(a,b) in SPLITS.items():
  if len(df)==0:out[s]=metric([]);continue
  x=df[(df.dt.dt.year>=a)&(df.dt.dt.year<=b)];o=metric(x.R.values);o['trades_per_month']=round(len(x)/((b-a+1)*12),1);o['positive_month_pct']=None
  if len(x):o['positive_month_pct']=round(float((x.set_index('dt').R.resample('MS').sum()>0).mean()*100),1)
  out[s]=o
 return out

def score(m):
 tr,va=m['train'],m['val']
 if tr.get('n',0)<120 or va.get('n',0)<60 or tr.get('mean_R') is None or va.get('mean_R') is None:return -999
 if tr['mean_R']<=0 or va['mean_R']<=0 or (tr.get('pf') or 0)<=1 or (va.get('pf') or 0)<=1:return -999
 return min(tr['mean_R'],va['mean_R'])-.25*abs(tr['mean_R']-va['mean_R'])

def main():
 data={}
 for p in PAIRS:
  d=load_pair(p)
  if d is not None:data[p]=addm(d)
 rs=rates();cands=[]
 for freq in ['DAILY','WEEKLY']:
  holds=[24,48] if freq=='DAILY' else [120]
  for h in holds:
   for mom in ['NONE','MOM5','MOM20']:
    for md in [0.5,1.0,2.0]:
     for st in [1.5,2.0,2.5]:
      cfg={'freq':freq,'hold':h,'mom':mom,'mindiff':md,'stop':st};r=make_rows(data,rs,cfg,1.0);m=sm(r);cands.append({'cfg':cfg,'metric':m,'score':score(m)})
 best=max(cands,key=lambda x:x['score']);base=make_rows(data,rs,best['cfg'],1.0);stress=make_rows(data,rs,best['cfg'],1.5);bm=sm(base);stm=sm(stress)
 valid_dev=best['score']>-900;testpass=valid_dev and (bm['test'].get('mean_R') or -99)>0 and (bm['test'].get('pf') or 0)>1
 ports={}
 if testpass:
  for r in [.004,.006,.008]:ports[f'holdout_base_{r}']=portfolio(base,2025,2026,r);ports[f'holdout_stress_{r}']=portfolio(stress,2025,2026,r)
 out={'status':'FX_CARRY_PHASE8','source':'Dukascopy H1 + FRED/OECD free monthly rates','rate_series':SERIES,'leakage_rule':'monthly observation usable only from month+2','protocol':'global config selected 2014-21; requires positive 2022-24 test; 2025-26 untouched','candidate_count':len(cands),'best':best,'base_metric':bm,'stress_metric':stm,'valid_trainval':valid_dev,'test_pass':testpass,'holdout_portfolio':ports,'top10':sorted(cands,key=lambda x:x['score'],reverse=True)[:10]}
 Path('tmp/fx_carry_phase8_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'best':best,'base':bm,'stress':stm,'testpass':testpass,'portfolio':ports},indent=2,default=str))
if __name__=='__main__':main()
