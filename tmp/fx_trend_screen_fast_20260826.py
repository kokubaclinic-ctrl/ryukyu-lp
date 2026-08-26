import json,math
from pathlib import Path
import numpy as np,pandas as pd
PAIRS=['eurusd','usdjpy','gbpusd','audusd','eurjpy','gbpjpy','audjpy']
PIP={p:(0.01 if p.endswith('jpy') else 0.0001) for p in PAIRS};COST={'eurusd':0.8,'usdjpy':1.0,'gbpusd':1.3,'audusd':1.1,'eurjpy':1.4,'gbpjpy':2.4,'audjpy':1.3};SPLITS={'train':(2014,2018),'val':(2019,2021),'test':(2022,2024),'holdout':(2025,2026)}
def load(p):
 d=pd.read_csv(f'/tmp/{p}_trend_h1.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 d=d.dropna();pip=PIP[p];prev=d.close.shift(1);tr=pd.concat([(d.high-d.low)/pip,(d.high-prev).abs()/pip,(d.low-prev).abs()/pip],axis=1).max(axis=1);d['atr']=tr.rolling(14,min_periods=10).mean().shift(1);r=d.close.diff()/pip;d['sig']=r.rolling(240,min_periods=120).std().shift(1)
 for lb in [24,72,120,240]:d[f'z{lb}']=((d.close-d.close.shift(lb))/pip)/(d.sig*np.sqrt(lb))
 d['r12']=((d.close-d.close.shift(12))/pip)/(d.sig*np.sqrt(12));return d

def met(a):
 a=np.asarray(a,float);n=len(a)
 if not n:return {'n':0}
 gp=a[a>0].sum();gl=-a[a<0].sum();sd=a.std(ddof=1) if n>1 else np.nan;return {'n':n,'mean_R':round(float(a.mean()),4),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((a>0).mean()*100),1),'t':round(float(a.mean()/(sd/math.sqrt(n))),2) if n>1 and sd>0 else None}

def ev(d,p,s):
 N=len(d);mask=(d.index.hour%6==0)&(d.index.weekday<5)&~((d.index.weekday==4)&(d.index.hour>=12));direction=np.zeros(N)
 if s['family']=='TSMOM':
  z=d[f"z{s['lb']}"].to_numpy();direction=np.where(mask&(np.abs(z)>=s['th']),np.sign(z),0)
 elif s['family']=='DONCHIAN':
  hi=d.high.rolling(s['lb']).max().shift(1).to_numpy();lo=d.low.rolling(s['lb']).min().shift(1).to_numpy();c=d.close.to_numpy();direction=np.where(mask&(c>hi),1,np.where(mask&(c<lo),-1,0))
 else:
  z=d[f"z{s['lb']}"].to_numpy();r=d.r12.to_numpy();direction=np.where(mask&(np.abs(z)>=s['th'])&(np.sign(r)==-np.sign(z))&(np.abs(r)>=s['pb']),np.sign(z),0)
 h=s['hold'];entry=d.open.shift(-1).to_numpy();exitp=d.open.shift(-(1+h)).to_numpy();stop=(d.atr.shift(-1).to_numpy()*1.5);R=direction*(exitp-entry)/PIP[p]/stop-COST[p]/stop
 ok=(direction!=0)&np.isfinite(R)&np.isfinite(stop)&(stop>0);dates=d.index[ok];R=R[ok];o=dict(s);o['pair']=p
 for name,(a,b) in SPLITS.items():m=(dates.year>=a)&(dates.year<=b);o[name]=met(R[m])
 return o

def score(x):
 tr=x['train'];va=x['val']
 if tr.get('n',0)<150 or va.get('n',0)<80 or tr.get('mean_R',-9)<=0 or va.get('mean_R',-9)<=0:return -999
 return min(tr['mean_R'],va['mean_R'])
def main():
 allc=[]
 for p in PAIRS:
  d=load(p)
  for lb in [24,72,120,240]:
   for th in [.5,1,1.5]:
    for h in [12,24,48]:allc.append(ev(d,p,{'family':'TSMOM','lb':lb,'th':th,'hold':h}))
  for lb in [24,72,120]:
   for h in [12,24,48]:allc.append(ev(d,p,{'family':'DONCHIAN','lb':lb,'hold':h}))
  for lb in [120,240]:
   for th in [.75,1]:
    for pb in [.5,.75]:
     for h in [12,24]:allc.append(ev(d,p,{'family':'PULLBACK','lb':lb,'th':th,'pb':pb,'hold':h}))
 winners=[]
 for p in PAIRS:
  for fam in ['TSMOM','DONCHIAN','PULLBACK']:
   q=sorted([x for x in allc if x['pair']==p and x['family']==fam],key=score,reverse=True)
   if q and score(q[0])>-900:winners.append(q[0])
 out={'status':'FX_TREND_FAST_SCREEN','candidate_count':len(allc),'winners':winners,'passing_test':[x for x in winners if x['test'].get('mean_R',-9)>0 and (x['test'].get('pf') or 0)>1],'top':sorted(allc,key=score,reverse=True)[:40]};Path('tmp/fx_trend_screen_fast_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'winners':[(x['pair'],x['family'],x['train']['mean_R'],x['val']['mean_R'],x['test']['mean_R'],x['holdout']['mean_R']) for x in winners],'passing':[(x['pair'],x['family'],x['test']['mean_R'],x['holdout']['mean_R']) for x in out['passing_test']]},indent=2))
if __name__=='__main__':main()
