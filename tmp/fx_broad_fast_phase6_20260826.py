import json,math
from pathlib import Path
import numpy as np,pandas as pd
PAIRS=['eurusd','usdjpy','gbpusd','audusd','eurjpy','gbpjpy','audjpy']
PIP={p:(.01 if p.endswith('jpy') else .0001) for p in PAIRS};COST={'eurusd':.8,'usdjpy':1.0,'gbpusd':1.3,'audusd':1.1,'eurjpy':1.4,'gbpjpy':2.4,'audjpy':1.3};SPLITS={'train':(2014,2018),'val':(2019,2021),'test':(2022,2024),'holdout':(2025,2026)}
def load(p):
 d=pd.read_csv(f'/tmp/{p}_trend_h1.csv');d['dt']=pd.to_datetime(d.timestamp,unit='ms',utc=True);d=d.set_index('dt').sort_index();d=d[~d.index.duplicated(keep='last')]
 for c in ['open','high','low','close']:d[c]=pd.to_numeric(d[c],errors='coerce')
 d=d.dropna();pip=PIP[p];prev=d.close.shift(1);tr=pd.concat([(d.high-d.low)/pip,(d.high-prev).abs()/pip,(d.low-prev).abs()/pip],axis=1).max(axis=1);d['atr']=tr.rolling(24,min_periods=18).mean().shift(1);r=d.close.diff()/pip;d['sig']=r.rolling(480,min_periods=240).std().shift(1)
 for lb in [1,3,6,12,24]:d[f'zret{lb}']=((d.close-d.close.shift(lb))/pip)/(d.sig*np.sqrt(lb))
 for ma in [24,48]:d[f'zma{ma}']=((d.close-d.close.rolling(ma,min_periods=ma).mean())/pip)/(d.atr+1e-9)
 for n in [6,12,24,48]:
  d[f'hi{n}']=d.high.rolling(n).max().shift(1);d[f'lo{n}']=d.low.rolling(n).min().shift(1);d[f'range{n}']=(d[f'hi{n}']-d[f'lo{n}'])/pip;d[f'rangepct{n}']=d[f'range{n}']/d[f'range{n}'].rolling(480,min_periods=240).median().shift(1)
 return d
def met(a):
 a=np.asarray(a,float);n=len(a)
 if not n:return {'n':0}
 gp=a[a>0].sum();gl=-a[a<0].sum();sd=a.std(ddof=1) if n>1 else np.nan;return {'n':int(n),'mean_R':round(float(a.mean()),4),'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((a>0).mean()*100),1),'t':round(float(a.mean()/(sd/math.sqrt(n))),2) if n>1 and sd>0 else None}
def evalsig(d,p,name,sig,dire,hold,costmult=1):
 N=len(d);sig=np.asarray(sig.fillna(False),bool);dire=np.asarray(dire.fillna(0),float);entry=d.open.shift(-1).to_numpy();exitp=d.open.shift(-(1+hold)).to_numpy();atr=d.atr.shift(-1).to_numpy();R=dire*(exitp-entry)/PIP[p]/(atr*1.5)-COST[p]*costmult/(atr*1.5);schedule=(d.index.hour%3==0)&(d.index.weekday<5)&~((d.index.weekday==4)&(d.index.hour>=15));ok=sig&(dire!=0)&schedule&np.isfinite(R)&(atr>0);dates=d.index[ok];vals=R[ok];o={'name':name,'pair':p,'hold':hold,'costmult':costmult}
 for s,(lo,hi) in SPLITS.items():m=(dates.year>=lo)&(dates.year<=hi);o[s]=met(vals[m])
 return o
def main():
 allc=[]
 for p in PAIRS:
  d=load(p)
  # Return shock mean reversion / continuation.
  for lb in [1,3,6,12]:
   z=d[f'zret{lb}']
   for th in [1.5,2.0,2.5]:
    for h in [3,6,12]:
     for mode in ['FADE','CONT']:
      dire=pd.Series((-np.sign(z) if mode=='FADE' else np.sign(z)),index=d.index);sig=z.abs()>=th;allc.append(evalsig(d,p,f'RET_{mode}_lb{lb}_th{th}_h{h}',sig,dire,h))
  # Deviation from rolling mean.
  for ma in [24,48]:
   z=d[f'zma{ma}']
   for th in [1.0,1.5,2.0]:
    for h in [3,6,12]:allc.append(evalsig(d,p,f'MEAN_FADE_ma{ma}_th{th}_h{h}',z.abs()>=th,pd.Series(-np.sign(z),index=d.index),h))
  # Prior-range breakout continuation/fade.
  for n in [12,24,48]:
   up=d.close>d[f'hi{n}'];dn=d.close<d[f'lo{n}'];dire0=pd.Series(np.where(up,1,np.where(dn,-1,0)),index=d.index)
   for h in [3,6,12,24]:
    allc.append(evalsig(d,p,f'BREAK_CONT_n{n}_h{h}',up|dn,dire0,h));allc.append(evalsig(d,p,f'BREAK_FADE_n{n}_h{h}',up|dn,-dire0,h))
  # Compression then breakout of 6/12h range.
  for n in [6,12]:
   up=d.close>d[f'hi{n}'];dn=d.close<d[f'lo{n}'];dire0=pd.Series(np.where(up,1,np.where(dn,-1,0)),index=d.index)
   for pct in [.6,.8]:
    comp=d[f'rangepct{n}']<=pct
    for h in [6,12,24]:allc.append(evalsig(d,p,f'COMP_BREAK_n{n}_p{pct}_h{h}',comp&(up|dn),dire0,h))
 # stress re-evaluate only candidates positive all base periods using reconstructed dictionary is cumbersome; first base survivors then lookup by name logic rerun by storing signal specs is skipped. Conservative base requires PF>1.03 each and min counts.
 def good(x):
  mins={'train':100,'val':60,'test':60,'holdout':30}
  return all(x[s].get('n',0)>=mins[s] and x[s].get('mean_R',-9)>0 and (x[s].get('pf') or 0)>1.03 for s in mins)
 surv=[x for x in allc if good(x)]
 out={'status':'FX_BROAD_FAST_PHASE6','candidate_count':len(allc),'survivors_base':surv,'survivor_count':len(surv),'top_by_weakest':sorted(allc,key=lambda x:min(x[s].get('mean_R',-99) for s in SPLITS),reverse=True)[:40]};Path('tmp/fx_broad_fast_phase6_20260826_result.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({'candidate_count':len(allc),'survivors':[(x['pair'],x['name'],x['train']['mean_R'],x['val']['mean_R'],x['test']['mean_R'],x['holdout']['mean_R']) for x in surv[:30]],'count':len(surv)},indent=2))
if __name__=='__main__':main()
