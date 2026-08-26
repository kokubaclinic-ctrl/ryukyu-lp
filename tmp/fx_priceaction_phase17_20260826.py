import json,math
from pathlib import Path
import numpy as np,pandas as pd

PAIRS=['eurusd','usdjpy','gbpusd','audusd','eurjpy','gbpjpy','usdcad','usdchf']
PIP={p:(0.01 if p.endswith('jpy') else 0.0001) for p in PAIRS}
COST={'eurusd':0.8,'usdjpy':1.0,'gbpusd':1.3,'audusd':1.1,'eurjpy':1.4,'gbpjpy':2.4,'usdcad':1.5,'usdchf':1.5}
SPLITS={'train':(2014,2018),'val':(2019,2021),'test':(2022,2024),'holdout':(2025,2026)}
FAMS=['PD_SWEEP','ASIA_SWEEP','FAILED_BREAK','COMP_EXP','IMPULSE_PB','INSIDE_BREAK']

def load(p):
 d=pd.read_csv(f'/tmp/{p}_pa_m15.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 d=d.dropna(subset=['open','high','low','close']);pip=PIP[p];prev=d.close.shift(1)
 tr=pd.concat([(d.high-d.low)/pip,(d.high-prev).abs()/pip,(d.low-prev).abs()/pip],axis=1).max(axis=1);d['atr']=tr.rolling(14,min_periods=14).mean().shift(1)
 return d

def execute(d,p,i,dire,stop_price,tpR=1.5,maxbars=8,costmult=1.0):
 # Signal is known at close i; enter next bar open. Conservative same-bar rule: stop wins if stop+TP both touched.
 ei=i+1
 if ei>=len(d)-1:return None
 entry=float(d.open.iloc[ei]);pip=PIP[p];risk=abs(entry-stop_price)/pip
 if not np.isfinite(risk) or risk<max(4,0.25*float(d.atr.iloc[ei] if np.isfinite(d.atr.iloc[ei]) else 0)):return None
 tp=entry+dire*tpR*risk*pip;last=min(len(d)-1,ei+maxbars);exitp=float(d.close.iloc[last]);why='TIME'
 for k in range(ei,last+1):
  lo=float(d.low.iloc[k]);hi=float(d.high.iloc[k]);hitS=(lo<=stop_price if dire>0 else hi>=stop_price);hitT=(hi>=tp if dire>0 else lo<=tp)
  if hitS:exitp=stop_price;why='SL';last=k;break
  if hitT:exitp=tp;why='TP';last=k;break
 cost=COST[p]*costmult;R=dire*(exitp-entry)/pip/risk-cost/risk
 return {'dt':d.index[ei],'end':d.index[last],'pair':p,'R':float(R),'why':why,'risk_pips':float(risk)}

def prior_ny_day_levels(d):
 ny=d.index.tz_convert('America/New_York')
 # FX day starts 17:00 NY; shift timestamps by 17h so group date represents trading day.
 key=(ny-pd.Timedelta(hours=17)).date
 tmp=pd.DataFrame({'key':key,'high':d.high.values,'low':d.low.values},index=d.index);g=tmp.groupby('key').agg({'high':'max','low':'min'});prev=g.shift(1)
 mpH={k:v for k,v in prev.high.items()};mpL={k:v for k,v in prev.low.items()};return np.array([mpH.get(k,np.nan) for k in key]),np.array([mpL.get(k,np.nan) for k in key])

def asia_levels(d):
 tok=d.index.tz_convert('Asia/Tokyo');dates=tok.date;hours=tok.hour
 tmp=pd.DataFrame({'date':dates,'hour':hours,'high':d.high.values,'low':d.low.values},index=d.index);a=tmp[(tmp.hour>=9)&(tmp.hour<15)].groupby('date').agg({'high':'max','low':'min'})
 H={k:v for k,v in a.high.items()};L={k:v for k,v in a.low.items()};return np.array([H.get(x,np.nan) for x in dates]),np.array([L.get(x,np.nan) for x in dates])

def gen(p,d,fam,costmult=1.0):
 rows=[];pip=PIP[p];ph,pl=prior_ny_day_levels(d);ah,al=asia_levels(d);lon=d.index.tz_convert('Europe/London')
 start=40
 for i in range(start,len(d)-10):
  atr=float(d.atr.iloc[i])
  if not np.isfinite(atr) or atr<=0:continue
  o,h,l,c=[float(d.iloc[i][x]) for x in ['open','high','low','close']];rng=max((h-l)/pip,1e-6);body=abs(c-o)/pip;dire=0;sp=None;maxbars=8;tp=1.5
  if fam=='PD_SWEEP':
   if lon[i].weekday()>=5 or not (7<=lon[i].hour<13):continue
   if np.isfinite(ph[i]) and h>ph[i]+0.10*atr*pip and c<ph[i] and (h-max(o,c))/pip>=0.35*rng:dire=-1;sp=h+0.20*atr*pip
   elif np.isfinite(pl[i]) and l<pl[i]-0.10*atr*pip and c>pl[i] and (min(o,c)-l)/pip>=0.35*rng:dire=1;sp=l-0.20*atr*pip
  elif fam=='ASIA_SWEEP':
   if lon[i].weekday()>=5 or not (7<=lon[i].hour<12):continue
   if np.isfinite(ah[i]) and h>ah[i]+0.10*atr*pip and c<ah[i] and (h-max(o,c))/pip>=0.35*rng:dire=-1;sp=h+0.20*atr*pip
   elif np.isfinite(al[i]) and l<al[i]-0.10*atr*pip and c>al[i] and (min(o,c)-l)/pip>=0.35*rng:dire=1;sp=l-0.20*atr*pip
  elif fam=='FAILED_BREAK':
   hi=float(d.high.iloc[i-16:i].max());lo=float(d.low.iloc[i-16:i].min())
   if h>hi+0.10*atr*pip and c<hi and (h-max(o,c))/pip>=0.40*rng:dire=-1;sp=h+0.15*atr*pip
   elif l<lo-0.10*atr*pip and c>lo and (min(o,c)-l)/pip>=0.40*rng:dire=1;sp=l-0.15*atr*pip
  elif fam=='COMP_EXP':
   hi=float(d.high.iloc[i-8:i].max());lo=float(d.low.iloc[i-8:i].min());box=(hi-lo)/pip
   if box<=1.4*atr and rng>=1.2*atr and body>=0.60*rng:
    if c>hi+0.05*atr*pip:dire=1;sp=min(l,lo)-0.10*atr*pip;tp=1.7
    elif c<lo-0.05*atr*pip:dire=-1;sp=max(h,hi)+0.10*atr*pip;tp=1.7
  elif fam=='IMPULSE_PB':
   # Four-bar impulse, prior bar retraces 20-60%, current bar resumes through prior extreme.
   a=float(d.open.iloc[i-5]);b=float(d.close.iloc[i-2]);imp=(b-a)/pip
   if abs(imp)>=1.5*atr:
    idir=np.sign(imp);imp_hi=max(a,b);imp_lo=min(a,b);po=float(d.open.iloc[i-1]);pc=float(d.close.iloc[i-1]);retr=(b-pc)/pip*idir
    if retr>=0.20*abs(imp) and retr<=0.60*abs(imp):
     if idir>0 and c>float(d.high.iloc[i-1]) and c>o:dire=1;sp=min(float(d.low.iloc[i-1]),l)-0.15*atr*pip;tp=1.8;maxbars=12
     elif idir<0 and c<float(d.low.iloc[i-1]) and c<o:dire=-1;sp=max(float(d.high.iloc[i-1]),h)+0.15*atr*pip;tp=1.8;maxbars=12
  elif fam=='INSIDE_BREAK':
   phh=float(d.high.iloc[i-2]);pll=float(d.low.iloc[i-2]);ih=float(d.high.iloc[i-1]);il=float(d.low.iloc[i-1])
   if ih<phh and il>pll:
    if c>phh and c>o:dire=1;sp=pll-0.10*atr*pip
    elif c<pll and c<o:dire=-1;sp=phh+0.10*atr*pip
  if dire and sp is not None:
   r=execute(d,p,i,dire,sp,tp,maxbars,costmult)
   if r is not None: r['family']=fam;rows.append(r)
 return pd.DataFrame(rows)

def metric(q):
 if q is None or len(q)==0:return {'n':0}
 a=q.R.values;gp=a[a>0].sum();gl=-a[a<0].sum();sd=a.std(ddof=1) if len(a)>1 else np.nan;m=q.set_index('dt').R.resample('MS').sum();y=q.set_index('dt').R.resample('YS').sum()
 return {'n':len(q),'trades_per_month':round(len(q)/max(1,len(m)),1),'avg_R':round(float(a.mean()),4),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((a>0).mean()*100),1),'t':round(float(a.mean()/(sd/math.sqrt(len(a)))),2) if len(a)>1 and sd>0 else None,'positive_month_pct':round(float((m>0).mean()*100),1),'positive_year_pct':round(float((y>0).mean()*100),1),'avg_risk_pips':round(float(q.risk_pips.mean()),1)}

def spl(q,a,b):return q[(q.dt.dt.year>=a)&(q.dt.dt.year<=b)].copy()

def main():
 data={p:load(p) for p in PAIRS};out={'status':'FX_PRICEACTION_PHASE17','source':'Dukascopy free M15','protocol':'six fixed market-structure rules; no holdout tuning; conservative same-bar SL priority','families':FAMS,'results':{},'survivors_all4':[]}
 for cm in [1.0,1.5]:
  rr={}
  for fam in FAMS:
   rr[fam]={};po=[]
   for p,d in data.items():
    q=gen(p,d,fam,cm);po.append(q);rr[fam][p]={s:metric(spl(q,*ab)) for s,ab in SPLITS.items()}
    if cm==1.0:
     ms=rr[fam][p]
     if all(ms[s].get('avg_R',-99)>0 and (ms[s].get('pf') or 0)>1 for s in SPLITS):out['survivors_all4'].append({'family':fam,'pair':p,'metrics':ms})
   z=pd.concat(po,ignore_index=True) if po else pd.DataFrame();rr[fam]['pooled']={s:metric(spl(z,*ab)) for s,ab in SPLITS.items()}
  out['results'][str(cm)]=rr
 Path('tmp/fx_priceaction_phase17_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'survivors':out['survivors_all4'],'pooled':{f:out['results']['1.0'][f]['pooled'] for f in FAMS}},indent=2,default=str))
if __name__=='__main__':main()
