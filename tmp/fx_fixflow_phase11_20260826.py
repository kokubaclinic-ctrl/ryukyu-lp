import json,math
from pathlib import Path
import numpy as np,pandas as pd

ALL=['eurusd','gbpusd','audusd','nzdusd','usdjpy','usdcad','usdchf','usdnok','usdsek']
ORIENT={p:(-1 if p.startswith('usd') else 1) for p in ALL}
PIP={p:(0.01 if p=='usdjpy' else 0.0001) for p in ALL}
# Conservative round-trip pips. Major-pair values are at/above recent OANDA Japan typical spreads; NOK/SEK deliberately conservative.
COST={'eurusd':0.8,'gbpusd':1.3,'audusd':1.1,'nzdusd':1.5,'usdjpy':1.0,'usdcad':1.5,'usdchf':1.5,'usdnok':20.0,'usdsek':20.0}
GROUPS={
 'G9':ALL,
 'G7':['eurusd','gbpusd','audusd','nzdusd','usdjpy','usdcad','usdchf'],
 'LIQ4':['eurusd','gbpusd','audusd','usdjpy'],
 'LIQ3':['eurusd','gbpusd','usdjpy'],
}
SPLITS={'dev':(2020,2022),'test':(2023,2024),'holdout':(2025,2026)}
VARIANTS=['PAPER','PAPER_EXJPY','CONFIRM','HEALTH60','HYBRID60','POSTCONFIRM']

def load(p):
 d=pd.read_csv(f'/tmp/{p}_fixflow_m5.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 return d.dropna(subset=['open','close'])

def at(d,t,maxmin=10):
 i=d.index.searchsorted(t)
 if i>=len(d) or (d.index[i]-t).total_seconds()>maxmin*60:return None
 return i

def prev_ny_17(t):
 x=t.tz_convert('America/New_York');day=pd.Timestamp(x.date()).tz_localize('America/New_York')+pd.Timedelta(hours=17)
 if day>=x:day-=pd.Timedelta(days=1)
 return day.tz_convert('UTC')

def next_ny(t,h):
 x=t.tz_convert('America/New_York');day=pd.Timestamp(x.date()).tz_localize('America/New_York')+pd.Timedelta(hours=h)
 if day<=x:day+=pd.Timedelta(days=1)
 return day.tz_convert('UTC')

def pair_seg(d,p,t0,t1):
 i=at(d,t0);j=at(d,t1)
 if i is None or j is None or j<=i:return None
 e=float(d.open.iloc[i]);x=float(d.open.iloc[j]);foreign_bps=ORIENT[p]*math.log(x/e)*10000.0
 cost_bps=COST[p]*PIP[p]/e*10000.0
 return foreign_bps,cost_bps

def avg_seg(data,pairs,t0,t1):
 vals=[];cost=[]
 for p in pairs:
  z=pair_seg(data[p],p,t0,t1)
  if z is None:return None
  vals.append(z[0]);cost.append(z[1])
 return float(np.mean(vals)),float(np.mean(cost))

def build_group(data,name,pairs):
 rows=[];missing=0
 for day in pd.date_range('2020-01-06','2026-07-31',freq='B'):
  tfix=(pd.Timestamp(day.date()).tz_localize('Asia/Tokyo')+pd.Timedelta(hours=9,minutes=55)).tz_convert('UTC')
  tpre=prev_ny_17(tfix);teu=next_ny(tfix,2)
  ld=teu.tz_convert('Europe/London').date();bd=teu.tz_convert('Europe/Berlin').date()
  tecb=(pd.Timestamp(bd).tz_localize('Europe/Berlin')+pd.Timedelta(hours=14,minutes=15)).tz_convert('UTC')
  tlon=(pd.Timestamp(ld).tz_localize('Europe/London')+pd.Timedelta(hours=16)).tz_convert('UTC');tend=next_ny(tlon,17)
  z1=avg_seg(data,pairs,tpre,tfix);z2=avg_seg(data,pairs,tfix,teu);z3=avg_seg(data,pairs,teu,tecb);z4=avg_seg(data,pairs,tlon,tend)
  exj=[p for p in pairs if p!='usdjpy'];z4x=avg_seg(data,exj,tlon,tend) if exj else None
  if any(z is None for z in [z1,z2,z3,z4,z4x]):missing+=1;continue
  rows.append({'date':tfix,
   'preT_g':-z1[0],'preT_c':z1[1],
   'postT_g':z2[0],'postT_c':z2[1],
   'preE_g':-z3[0],'preE_c':z3[1],
   'postL_g':z4[0],'postL_c':z4[1],
   'postLX_g':z4x[0],'postLX_c':z4x[1]})
 df=pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
 for c in ['preT_g','postT_g','preE_g','postL_g','postLX_g']:
  df['h_'+c]=df[c].rolling(60,min_periods=40).mean().shift(1)
 return df,missing

def apply_variant(df,variant,costmult=1.0):
 out=[]
 for _,r in df.iterrows():
  legs=[]
  if variant=='PAPER':
   legs=[('preT',r.preT_g,r.preT_c),('postT',r.postT_g,r.postT_c),('preE',r.preE_g,r.preE_c),('postL',r.postL_g,r.postL_c)]
  elif variant=='PAPER_EXJPY':
   legs=[('preT',r.preT_g,r.preT_c),('postT',r.postT_g,r.postT_c),('preE',r.preE_g,r.preE_c),('postLX',r.postLX_g,r.postLX_c)]
  elif variant=='CONFIRM':
   legs=[('preT',r.preT_g,r.preT_c),('preE',r.preE_g,r.preE_c)]
   if r.preT_g>0:legs.append(('postT',r.postT_g,r.postT_c))
   if r.preE_g>0:legs.append(('postLX',r.postLX_g,r.postLX_c))
  elif variant=='HEALTH60':
   if r.h_preT_g>0:legs.append(('preT',r.preT_g,r.preT_c))
   if r.h_postT_g>0:legs.append(('postT',r.postT_g,r.postT_c))
   if r.h_preE_g>0:legs.append(('preE',r.preE_g,r.preE_c))
   if r.h_postLX_g>0:legs.append(('postLX',r.postLX_g,r.postLX_c))
  elif variant=='HYBRID60':
   if r.h_preT_g>0:legs.append(('preT',r.preT_g,r.preT_c))
   if r.h_postT_g>0 and r.preT_g>0:legs.append(('postT',r.postT_g,r.postT_c))
   if r.h_preE_g>0:legs.append(('preE',r.preE_g,r.preE_c))
   if r.h_postLX_g>0 and r.preE_g>0:legs.append(('postLX',r.postLX_g,r.postLX_c))
  elif variant=='POSTCONFIRM':
   if r.preT_g>0:legs.append(('postT',r.postT_g,r.postT_c))
   if r.preE_g>0:legs.append(('postLX',r.postLX_g,r.postLX_c))
  net=sum(g-costmult*c for _,g,c in legs);gross=sum(g for _,g,_ in legs)
  out.append({'date':r.date,'net_bps':float(net),'gross_bps':float(gross),'legs':len(legs)})
 return pd.DataFrame(out)

def metric(x):
 if len(x)==0:return {'n_days':0}
 a=x.net_bps.values;gp=a[a>0].sum();gl=-a[a<0].sum();sd=a.std(ddof=1) if len(a)>1 else np.nan
 m=x.set_index('date').net_bps.resample('MS').sum();legs=x.legs.sum()
 return {'n_days':len(x),'mean_bps_day':round(float(a.mean()),4),'median_bps_day':round(float(np.median(a)),4),'pf':round(float(gp/gl),3) if gl>0 else None,'win_day_pct':round(float((a>0).mean()*100),1),'t':round(float(a.mean()/(sd/math.sqrt(len(a)))),2) if len(a)>1 and sd>0 else None,'positive_month_pct':round(float((m>0).mean()*100),1),'strategy_legs_per_month':round(float(legs/max(1,len(m))),1),'active_day_pct':round(float((x.legs>0).mean()*100),1)}

def split(df,yr0,yr1):return df[(df.date.dt.year>=yr0)&(df.date.dt.year<=yr1)].copy()

def dev_score(m):
 if m.get('n_days',0)<250 or m.get('mean_bps_day',-99)<=0 or (m.get('pf') or 0)<=1:return -999
 # Prefer positive expectancy and month consistency; no holdout information enters this score.
 return m['mean_bps_day']+0.03*(m['positive_month_pct']-50)

def equity(x,leverage,cap=500000):
 if len(x)==0:return {'n_days':0}
 x=x.copy();x['ret']=x.net_bps/10000.0*leverage;eq=cap;peak=cap;md=0;vals=[]
 for _,r in x.iterrows():
  eq*=max(.01,1+float(r.ret));peak=max(peak,eq);md=max(md,(peak-eq)/peak);vals.append((r.date,eq))
 q=pd.DataFrame(vals,columns=['date','equity']);q['pnl']=q.equity.diff().fillna(q.equity.iloc[0]-cap);m=q.set_index('date').pnl.resample('MS').sum();m=m.reindex(pd.date_range(x.date.min().to_period('M').to_timestamp().tz_localize('UTC'),x.date.max().to_period('M').to_timestamp().tz_localize('UTC'),freq='MS'),fill_value=0)
 return {'days':len(x),'leverage':leverage,'final_equity':round(float(eq)),'return_pct':round(float((eq/cap-1)*100),2),'avg_monthly_jpy':round(float(m.mean())),'median_monthly_jpy':round(float(m.median())),'positive_month_pct':round(float((m>0).mean()*100),1),'p10_month_jpy':round(float(m.quantile(.1))),'p90_month_jpy':round(float(m.quantile(.9))),'max_dd_pct':round(float(md*100),2),'strategy_legs_per_month':round(float(x.legs.sum()/max(1,len(m))),1)}

def main():
 data={p:load(p) for p in ALL};coverage={p:{'rows':len(d),'start':str(d.index.min()),'end':str(d.index.max())} for p,d in data.items()};built={};cands=[]
 for gn,ps in GROUPS.items():
  df,miss=build_group(data,gn,ps);built[gn]=df
  for v in VARIANTS:
   base=apply_variant(df,v,1.0);dev=metric(split(base,2020,2022));cands.append({'group':gn,'variant':v,'missing_days':miss,'available_days':len(df),'dev':dev,'score':dev_score(dev)})
 ranked=sorted(cands,key=lambda z:z['score'],reverse=True);best=ranked[0]
 g=best['group'];v=best['variant'];base=apply_variant(built[g],v,1.0);stress=apply_variant(built[g],v,1.5)
 test=split(base,2023,2024);hold=split(base,2025,2026);test_s=split(stress,2023,2024);hold_s=split(stress,2025,2026)
 accepted=best['score']>-900 and metric(test).get('mean_bps_day',-99)>0 and (metric(test).get('pf') or 0)>1
 eq={}
 if accepted:
  for lev in [2,4,6,8,10]:eq[f'base_{lev}']=equity(hold,lev);eq[f'stress_{lev}']=equity(hold_s,lev)
 out={'status':'FX_FIXFLOW_PHASE11','source':'Dukascopy free M5','protocol':'theory-defined groups/variants; select on 2020-22 DEV only; confirm 2023-24 TEST; 2025-26 HOLDOUT untouched','groups':GROUPS,'variants':VARIANTS,'coverage':coverage,'ranked_dev':ranked,'frozen':best,'test_base':metric(test),'test_stress':metric(test_s),'accepted_before_holdout':accepted,'holdout_base':metric(hold),'holdout_stress':metric(hold_s),'holdout_equity':eq}
 Path('tmp/fx_fixflow_phase11_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'best':best,'test':out['test_base'],'accepted':accepted,'holdout':out['holdout_base'],'equity':eq},indent=2,default=str))
if __name__=='__main__':main()
