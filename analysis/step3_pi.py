import pandas as pd, numpy as np
b=pd.read_pickle('/home/claude/work/balls2.pkl')
b['lateness']=((b.legal_before-30)/90).clip(0,1)           # ramps from over 6 to over 20
# --- recent wickets in last 18 legal balls (3 overs), same innings
b['wk_i']=b.wk.astype(int)
b['ball_idx']=b.groupby(['match_id','innings']).cumcount()
cs=b.groupby(['match_id','innings']).wk_i.cumsum()
lag=b.groupby(['match_id','innings']).wk_i.apply(lambda s: s.cumsum().shift(20).fillna(0)).reset_index(level=[0,1],drop=True)
b['w_recent']=(cs-b.wk_i)-lag.reindex(b.index).values   # wickets in last ~20 deliveries before this ball
# expected wickets by delivery count
ew=b.groupby(['level','legal_before']).wkts_before.mean().rename('E_w').reset_index()
b=b.merge(ew,on=['level','legal_before'],how='left')
# ---- chase components
D=(b.need/b.E_rem.clip(lower=1))
b['difficulty']=((D-0.7)/0.9).clip(0,1)
b['closeness']=1-(2*b.wp_before-1).abs()
PI2=100*(0.45*b.closeness+0.30*b.difficulty+0.25*b.lateness)
b['decided']=(b.innings==2)&((b.wp_before<0.05)|(b.wp_before>0.95))   # game effectively decided -> excluded from both classes
# ---- first innings components
par=np.where(b.level=='IPL',187,173)
S_col=(b.w_recent/3).clip(0,1)
S_dep=((b.wkts_before-b.E_w)/3).clip(0,1)
S_par=((par-b.proj)/40).clip(0,1)
PI1=100*(0.30*S_col+0.25*S_dep+0.25*b.lateness+0.20*S_par)
b['PI']=np.where(b.innings==2,PI2,PI1)
b['PI_pct']=b.PI.where(~b.decided).groupby(b.innings).rank(pct=True)
b['pressure']=np.where(b.decided,'Decided',np.where(b.PI_pct>=0.75,'High',np.where(b.PI_pct<=0.50,'Low','Mid')))
b.to_pickle('/home/claude/work/balls3.pkl')
# validation table: league-wide outcomes by class
b['bat_ball']=b.legal
leg=b[b.legal]
def summ(d):
    return pd.Series({'balls':len(d),'RR':d.bat_runs.mean()*6,'dot%':(d.bat_runs==0).mean()*100,'bound%':(d.is_four|d.is_six).mean()*100,'wkt/100':d.wk.mean()*100,
                      'avg_rrr':d.rrr[d.rrr>0].mean(),'avg_wkts':d.wkts_before.mean(),'avg_bl':d.balls_left.mean()})
print(leg.groupby(['innings','pressure']).apply(summ).round(2))
print(leg.groupby('pressure').apply(summ).round(2))
# does PI relate to eventual closeness of match?
