import json,math
from pathlib import Path
import numpy as np,pandas as pd
import holidays

SPLITS={'train':(2014,2019),'val':(2020,2022),'test':(2023,2024),'holdout':(2025,2026)}
PIP=0.01;STOP_PIPS=60.0

def load():
 d=pd.read_csv('/tmp/usdjpy_gotobi_m5.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 return d.dropna(subset=['open','high','low','close'])

def bank_business(date,jph):
 if date.weekday()>=5 or date in jph:return False
 if (date.month==12 and date.day==31) or (date.month==1 and date.day in [1,2,3]):return False
 return True

def gotobi_dates(start_year,end_year):
 jph=holidays.Japan(years=range(start_year,end_year+1));out=[];targets=[]
 for y in range(start_year,end_year+1):
  for m in range(1,13):
   for day in [5,10,15,20,25,30]:
    try:t=pd.Timestamp(y,m,day).date()
    except ValueError:continue
    a=t
    while not bank_business(a,jph):a=(pd.Timestamp(a)-pd.Timedelta(days=1)).date()
    targets.append({'nominal':str(t),'trade_date':a})
    out.append(a)
 return sorted(set(out)),targets

def all_biz_dates(start,end):
 jph=holidays.Japan(years=range(start.year,end.year+1));d=start;out=[]
 while d<=end:
  if bank_business(d,jph):out.append(d)
  d=(pd.Timestamp(d)+pd.Timedelta(days=1)).date()
 return out

def at(idx,t,maxmin=10):
 i=idx.searchsorted(t)
 if i>=len(idx) or (idx[i]-t).total_seconds()>maxmin*60:return None
 return i

def one_trade(d,date,cost_pips=1.0):
 # Tokyo fix 09:55 JST. External frozen survivor: enter 9h before, long USDJPY, SL 60 pips, exit at fix.
 fix=(pd.Timestamp(date).tz_localize('Asia/Tokyo')+pd.Timedelta(hours=9,minutes=55)).tz_convert('UTC');entry_t=fix-pd.Timedelta(hours=9)
 i=at(d.index,entry_t);j=at(d.index,fix)
 if i is None or j is None or j<=i:return None
 entry=float(d.open.iloc[i]);stop=entry-STOP_PIPS*PIP;sl=False
 for k in range(i,j+1):
  if float(d.low.iloc[k])<=stop:sl=True;break
 if sl:
  net_pips=-STOP_PIPS-cost_pips;R=-1-cost_pips/STOP_PIPS
 else:
  gross=(float(d.open.iloc[j])-entry)/PIP;net_pips=gross-cost_pips;R=net_pips/STOP_PIPS
 return {'date':fix,'entry':entry_t,'R':float(R),'net_pips':float(net_pips),'stopped':sl}

def metric(df):
 if len(df)==0:return {'n':0}
 a=df.R.values;gp=a[a>0].sum();gl=-a[a<0].sum();sd=a.std(ddof=1) if len(a)>1 else np.nan;m=df.set_index('date').R.resample('MS').sum()
 return {'n':len(df),'trades_per_month':round(len(df)/max(1,len(m)),1),'avg_R':round(float(a.mean()),4),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((a>0).mean()*100),1),'t':round(float(a.mean()/(sd/math.sqrt(len(a)))),2) if len(a)>1 and sd>0 else None,'mean_net_pips':round(float(df.net_pips.mean()),2),'positive_month_pct':round(float((m>0).mean()*100),1),'stop_pct':round(float(df.stopped.mean()*100),1)}

def split(df,a,b):return df[(df.date.dt.year>=a)&(df.date.dt.year<=b)].copy()

def equity(df,risk,cap=500000):
 if len(df)==0:return {'n':0}
 eq=cap;peak=cap;md=0;vals=[]
 for _,r in df.sort_values('date').iterrows():
  pnl=eq*risk*float(r.R);eq+=pnl;peak=max(peak,eq);md=max(md,(peak-eq)/peak);vals.append((r.date,eq,pnl))
 q=pd.DataFrame(vals,columns=['date','equity','pnl']);m=q.set_index('date').pnl.resample('MS').sum()
 return {'n':len(df),'risk_pct':risk*100,'final_equity':round(float(eq)),'return_pct':round(float((eq/cap-1)*100),2),'avg_monthly_jpy':round(float(m.mean())),'median_monthly_jpy':round(float(m.median())),'positive_month_pct':round(float((m>0).mean()*100),1),'max_dd_pct':round(float(md*100),2)}

def main():
 d=load();gdates,targetmap=gotobi_dates(2014,2026);biz=all_biz_dates(pd.Timestamp('2014-01-01').date(),pd.Timestamp('2026-07-31').date());out={'status':'USDJPY_GOTOBI_PHASE12','source':'Dukascopy free M5','rule':'Japanese bank-business Gotobi dates; long USDJPY 9h before 09:55 JST fix; exit fix; SL60p; no swap credited','coverage':{'rows':len(d),'start':str(d.index.min()),'end':str(d.index.max())},'cost_scenarios':{},'control_same_window':{}}
 for cost in [1.0,1.5,2.0]:
  rows=[one_trade(d,x,cost) for x in gdates];df=pd.DataFrame([x for x in rows if x is not None]);sc={'splits':{}}
  for s,(a,b) in SPLITS.items():sc['splits'][s]=metric(split(df,a,b))
  h=split(df,2025,2026);sc['holdout_equity']={str(r):equity(h,r) for r in [0.004,0.006,0.008,0.01]};out['cost_scenarios'][str(cost)]=sc
 # Control: all Japanese bank business days, same window/stop/cost=1.0, split Gotobi vs non-Gotobi.
 allrows=[one_trade(d,x,1.0) for x in biz];adf=pd.DataFrame([x for x in allrows if x is not None]);gset=set(gdates);adf['gotobi']=adf.date.dt.tz_convert('Asia/Tokyo').dt.date.isin(gset)
 for s,(a,b) in SPLITS.items():
  z=split(adf,a,b);out['control_same_window'][s]={'gotobi':metric(z[z.gotobi]),'non_gotobi':metric(z[~z.gotobi])}
 Path('tmp/fx_gotobi_phase12_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'base':out['cost_scenarios']['1.0'],'control':out['control_same_window']},indent=2,default=str))
if __name__=='__main__':main()
