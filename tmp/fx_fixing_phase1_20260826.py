import json, math
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

PAIRS=['eurusd','usdjpy','gbpusd','audusd']
PIP={'eurusd':0.0001,'usdjpy':0.01,'gbpusd':0.0001,'audusd':0.0001}
# OANDA Tokyo Standard current upper-end spread + modest retail slippage allowance, round-turn pips.
BASE_COST={'eurusd':0.8,'usdjpy':1.0,'gbpusd':1.3,'audusd':1.1}
FIXES={
 'TOKYO':('Asia/Tokyo',9,55),
 'FRANKFURT':('Europe/Berlin',14,15),
 'LONDON':('Europe/London',16,0),
}
USD_APP_DIR={'eurusd':-1,'usdjpy':1,'gbpusd':-1,'audusd':-1}
SPLITS={'train':(2014,2018),'val':(2019,2021),'test':(2022,2024),'holdout':(2025,2026)}

def load(pair):
    d=pd.read_csv(f'/tmp/{pair}_m5.csv')
    d['dt']=pd.to_datetime(d['timestamp'],unit='ms',utc=True)
    d=d.set_index('dt').sort_index()
    d=d[~d.index.duplicated(keep='last')]
    for c in ['open','high','low','close']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['open','high','low','close'])
    # prior-60-minute realized volatility in pips; shifted so entry bar is never used.
    r=d['close'].diff()/PIP[pair]
    d['rv60']=np.sqrt((r*r).rolling(12,min_periods=10).sum()).shift(1)
    return d

def exact_or_next(idx,t,max_min=6):
    i=idx.searchsorted(t)
    if i>=len(idx): return None
    if (idx[i]-t).total_seconds()>max_min*60:return None
    return i

def metric(vals,dates=None):
    a=np.asarray(vals,float); n=len(a)
    if n==0:return {'n':0}
    gp=a[a>0].sum(); gl=-a[a<0].sum(); sd=a.std(ddof=1) if n>1 else np.nan
    out={'n':int(n),'mean':round(float(a.mean()),4),'median':round(float(np.median(a)),4),
         'pf':round(float(gp/gl),3) if gl>0 else None,'win_pct':round(float((a>0).mean()*100),1),
         't':round(float(a.mean()/(sd/math.sqrt(n))),2) if n>1 and sd>0 else None,
         'sum':round(float(a.sum()),2)}
    if dates is not None and n:
        s=pd.Series(a,index=pd.DatetimeIndex(dates))
        ym=s.groupby([s.index.year,s.index.month]).sum()
        yy=s.groupby(s.index.year).sum()
        out['positive_month_pct']=round(float((ym>0).mean()*100),1)
        out['positive_year_pct']=round(float((yy>0).mean()*100),1)
    return out

def split_mask(dates,name):
    lo,hi=SPLITS[name]
    y=pd.DatetimeIndex(dates).year
    return (y>=lo)&(y<=hi)

def generate_event_times(d,fix):
    tz,h,m=FIXES[fix]
    start=d.index.min().date(); end=d.index.max().date()
    days=pd.date_range(start,end,freq='D')
    out=[]
    for day in days:
        local=pd.Timestamp(day.date()).tz_localize(tz)+pd.Timedelta(hours=h,minutes=m)
        u=local.tz_convert('UTC')
        if u.weekday()<5: out.append(u)
    return out

def trade_raw(d,pair,fix,leg,param,cost_mult=1.0):
    idx=d.index; pip=PIP[pair]; cost=BASE_COST[pair]*cost_mult; vals=[]; dates=[]; rows=[]
    usd=USD_APP_DIR[pair]
    for ft in generate_event_times(d,fix):
        if leg=='PRE':
            entry_t=ft-pd.Timedelta(minutes=param); exit_t=ft
            direction=usd
        else:
            entry_t=ft+pd.Timedelta(minutes=5); exit_t=entry_t+pd.Timedelta(minutes=param)
            direction=-usd
        i=exact_or_next(idx,entry_t); j=exact_or_next(idx,exit_t)
        if i is None or j is None or j<=i: continue
        entry=float(d.iloc[i].open); exitp=float(d.iloc[j].open)
        gross=direction*(exitp-entry)/pip
        net=gross-cost
        vals.append(net); dates.append(idx[i])
        rows.append((idx[i],net,gross,i,j,d.iloc[i].rv60,direction,entry))
    return np.asarray(vals),pd.DatetimeIndex(dates),rows

def evaluate_candidate(d,pair,fix,leg,param,cost_mult=1.0):
    vals,dates,_=trade_raw(d,pair,fix,leg,param,cost_mult)
    out={'pair':pair,'fix':fix,'leg':leg,'minutes':param,'cost_mult':cost_mult}
    for s in SPLITS:
        m=split_mask(dates,s); out[s]=metric(vals[m],dates[m])
    return out

def candidate_score(c):
    tr=c['train']; va=c['val']
    if tr.get('n',0)<700 or va.get('n',0)<400:return -1e9
    if tr.get('mean',-99)<=0 or va.get('mean',-99)<=0:return -1e9
    if (tr.get('pf') or 0)<=1 or (va.get('pf') or 0)<=1:return -1e9
    return min(tr['mean'],va['mean']) + 0.002*min(tr.get('positive_month_pct',0),va.get('positive_month_pct',0))

def risk_trades(d,pair,fix,leg,param,stop_mult,cost_mult=1.0):
    _,_,rows=trade_raw(d,pair,fix,leg,param,cost_mult)
    pip=PIP[pair]; cost=BASE_COST[pair]*cost_mult
    vals=[]; dates=[]
    for dt,net,gross,i,j,rv,direction,entry in rows:
        if not np.isfinite(rv) or rv<=0: continue
        # Floor avoids pathological micro-stops in quiet periods.
        floor=3.0 if 'jpy' not in pair else 4.0
        stop_pips=max(floor,float(rv)*stop_mult)
        stop_price=entry-direction*stop_pips*pip
        stopped=False
        # Check every completed M5 bar from entry through planned exit.
        for k in range(i,min(j+1,len(d))):
            bar=d.iloc[k]
            if direction>0 and float(bar.low)<=stop_price:
                stopped=True; break
            if direction<0 and float(bar.high)>=stop_price:
                stopped=True; break
        if stopped:
            r=-1.0-cost/stop_pips
        else:
            r=gross/stop_pips-cost/stop_pips
        vals.append(r);dates.append(dt)
    return np.asarray(vals),pd.DatetimeIndex(dates)

def risk_eval(d,cand,stop_mult,cost_mult):
    vals,dates=risk_trades(d,cand['pair'],cand['fix'],cand['leg'],cand['minutes'],stop_mult,cost_mult)
    out={'stop_mult':stop_mult,'cost_mult':cost_mult}
    for s in SPLITS:
        m=split_mask(dates,s); out[s]=metric(vals[m],dates[m])
    return out,vals,dates

def choose_stop(d,cand):
    tests=[]
    for sm in [1.0,1.5,2.0,2.5]:
        x,_,_=risk_eval(d,cand,sm,1.0); tests.append(x)
    def sc(x):
        tr=x['train'];va=x['val']
        if tr.get('n',0)<700 or va.get('n',0)<400:return -1e9
        if tr.get('mean',-99)<=0 or va.get('mean',-99)<=0:return -1e9
        return min(tr['mean'],va['mean'])
    best=max(tests,key=sc)
    return best,tests

def portfolio_stats(trades,risk_pct=0.008,capital=500000):
    if not trades:return {'n':0}
    df=pd.DataFrame(trades,columns=['dt','R','fix']).sort_values('dt')
    df=df.drop_duplicates(['dt','fix'])
    per_trade_yen=capital*risk_pct
    df['pnl']=df.R*per_trade_yen
    m=df.set_index('dt').pnl.resample('MS').sum()
    # Include zero months between first and last trade.
    full=pd.date_range(m.index.min(),m.index.max(),freq='MS');m=m.reindex(full,fill_value=0.0)
    eq=capital+df.pnl.cumsum(); peak=np.maximum.accumulate(eq.values); dd=(eq.values-peak)/peak
    return {'n':int(len(df)),'months':int(len(m)),'trades_per_month':round(len(df)/len(m),1),
            'avg_monthly_jpy':round(float(m.mean())), 'median_monthly_jpy':round(float(m.median())),
            'positive_month_pct':round(float((m>0).mean()*100),1),
            'p10_monthly_jpy':round(float(m.quantile(.10))), 'p90_monthly_jpy':round(float(m.quantile(.90))),
            'max_dd_pct':round(float(-dd.min()*100),2), 'total_jpy':round(float(df.pnl.sum())),
            'avg_R':round(float(df.R.mean()),4),'pf_R':round(float(df.loc[df.R>0,'R'].sum()/-df.loc[df.R<0,'R'].sum()),3) if (df.R<0).any() else None}

def main():
    data={p:load(p) for p in PAIRS}
    # Phase 1: evidence-backed benchmark fixing candidates.
    candidates=[]
    for fix in FIXES:
        for p in PAIRS:
            for w in [60,90,120]: candidates.append(evaluate_candidate(data[p],p,fix,'PRE',w,1.0))
            for h in [30,60,90,120]: candidates.append(evaluate_candidate(data[p],p,fix,'POST',h,1.0))
    selected={}
    for fix in FIXES:
        pool=[c for c in candidates if c['fix']==fix]
        ranked=sorted(pool,key=candidate_score,reverse=True)
        selected[fix]=ranked[0] if candidate_score(ranked[0])>-1e8 else None
    # Freeze chosen pair/leg/window from train+validation. Then choose only stop width from train+validation.
    risk_selected={}; all_trades_base=[];all_trades_stress=[]
    for fix,c in selected.items():
        if c is None: continue
        best,grid=choose_stop(data[c['pair']],c)
        if min(best['train'].get('mean',-99),best['val'].get('mean',-99))<=0: continue
        base,vals,dates=risk_eval(data[c['pair']],c,best['stop_mult'],1.0)
        stress,svals,sdates=risk_eval(data[c['pair']],c,best['stop_mult'],1.5)
        risk_selected[fix]={'raw':c,'stop_choice':best,'stop_grid':grid,'base':base,'stress_1.5x':stress}
        # Portfolio uses only untouched TEST+HOLDOUT for honest forward-like assessment.
        for R,dt in zip(vals,dates):
            if 2022<=dt.year<=2026: all_trades_base.append((dt,float(R),fix))
        for R,dt in zip(svals,sdates):
            if 2022<=dt.year<=2026: all_trades_stress.append((dt,float(R),fix))
    ports={}
    for r in [0.006,0.007,0.008]:
        ports[f'base_risk_{r}']=portfolio_stats(all_trades_base,r)
        ports[f'stress1.5x_risk_{r}']=portfolio_stats(all_trades_stress,r)
    # Also report final untouched HOLDOUT only.
    hb=[x for x in all_trades_base if x[0].year>=2025]; hs=[x for x in all_trades_stress if x[0].year>=2025]
    holdout_ports={}
    for r in [0.006,0.007,0.008]:
        holdout_ports[f'base_risk_{r}']=portfolio_stats(hb,r)
        holdout_ports[f'stress1.5x_risk_{r}']=portfolio_stats(hs,r)
    out={'status':'FX_FIXING_PHASE1','data':'Dukascopy M5 2014-2026','pairs':PAIRS,
         'cost_pips_base':BASE_COST,'selection_protocol':'pair/leg/window selected only on 2014-18 train + 2019-21 validation; stop selected only there; 2022-24 test and 2025-26 holdout untouched',
         'candidate_count':len(candidates),'selected_raw':selected,'risk_selected':risk_selected,
         'portfolio_test_plus_holdout_2022_2026':ports,'portfolio_final_holdout_2025_2026':holdout_ports,
         'top_by_fix':{fix:sorted([c for c in candidates if c['fix']==fix],key=candidate_score,reverse=True)[:5] for fix in FIXES}}
    Path('tmp/fx_fixing_phase1_20260826_result.json').write_text(json.dumps(out,indent=2,default=str))
    print(json.dumps({'selected':{k:(None if v is None else [v['pair'],v['leg'],v['minutes'],v['train']['mean'],v['val']['mean'],v['test']['mean'],v['holdout']['mean']]) for k,v in selected.items()},
                      'risk_selected':{k:[v['raw']['pair'],v['raw']['leg'],v['raw']['minutes'],v['stop_choice']['stop_mult'],v['base']['train']['mean'],v['base']['val']['mean'],v['base']['test']['mean'],v['base']['holdout']['mean']] for k,v in risk_selected.items()},
                      'portfolio':ports,'holdout':holdout_ports},indent=2))

if __name__=='__main__':main()
