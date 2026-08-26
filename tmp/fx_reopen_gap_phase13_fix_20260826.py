import pandas as pd, numpy as np
import fx_reopen_gap_phase13_20260826 as m

def portfolio(df,yr0,yr1,risk=0.004,cap=500000,max_same_day=3):
    x=m.split(df,yr0,yr1).sort_values(['dt','gap_pips'],ascending=[True,False]).copy()
    if len(x)==0:return {'n':0}
    kept=[]
    for day,g in x.groupby(x.dt.dt.date):
        g=g.assign(abs_gap=g.gap_pips.abs()).sort_values('abs_gap',ascending=False).head(max_same_day);kept.append(g)
    x=pd.concat(kept).sort_values('dt');eq=cap;peak=cap;md=0;vals=[]
    for _,r in x.iterrows():
        stop=max(5.0,abs(float(r['gap_pips'])));R=float(r['net_pips'])/stop;pnl=eq*risk*R;eq+=pnl;peak=max(peak,eq);md=max(md,(peak-eq)/peak);vals.append((r['dt'],pnl,R))
    q=pd.DataFrame(vals,columns=['dt','pnl','R']);mo=q.set_index('dt').pnl.resample('MS').sum();mo=mo.reindex(pd.date_range(f'{yr0}-01-01',f'{yr1}-12-01',freq='MS',tz='UTC'),fill_value=0);a=q.R.values;gp=a[a>0].sum();gl=-a[a<0].sum()
    return {'n':len(q),'trades_per_month':round(len(q)/max(1,len(mo)),1),'avg_monthly_jpy':round(float(mo.mean())),'median_monthly_jpy':round(float(mo.median())),'positive_month_pct':round(float((mo>0).mean()*100),1),'p10_month_jpy':round(float(mo.quantile(.1))),'p90_month_jpy':round(float(mo.quantile(.9))),'return_pct':round(float((eq/cap-1)*100),2),'max_dd_pct':round(float(md*100),2),'avg_R':round(float(a.mean()),4),'pf_R':round(float(gp/gl),3) if gl>0 else None}

m.portfolio=portfolio
m.main()
