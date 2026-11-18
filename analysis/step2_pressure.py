import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGC, HistGradientBoostingRegressor as HGR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
b=pd.read_pickle('/home/claude/work/balls.pkl')
b['ipl']=(b.level=='IPL').astype(int)
b['wkts_left']=10-b.wkts_before
# expected remaining runs E(b,w) by level, smoothed via GBM on (balls_left,wkts_before,level) using both innings (innings-1 starting states + chase)
b['rem_runs']=b.groupby(['match_id','innings']).team_runs.transform('sum')-(b.score_before)  # remaining runs incl this ball (innings total uses balls kept)
# NOTE: rows after filtering may drop final chase balls; recompute from full cumulative
full=pd.read_pickle('/home/claude/work/balls.pkl')
FE=['balls_left','wkts_before','ipl']
b['inn_end']=b.groupby(['match_id','innings']).score_before.transform('max')+b.groupby(['match_id','innings']).team_runs.transform('last')
b['rem_runs']=b.inn_end-b.score_before
Er=HGR(max_iter=150,learning_rate=0.08,max_depth=4,min_samples_leaf=200)
# first-innings states are unbiased for "natural" remaining runs (chase truncates at target)
f1=b[b.innings==1]
Er.fit(f1[FE],f1.rem_runs)
b['E_rem']=Er.predict(b[FE])
# next-state expected remaining runs
nxt=b[FE].copy(); nxt['balls_left']=(b.balls_left-b.legal.astype(int)).clip(lower=0); nxt['wkts_before']=(b.wkts_before+b.wk.astype(int)).clip(upper=10)
b['E_rem_next']=np.where(nxt.balls_left==0,0,np.where(nxt.wkts_before>=10,0,Er.predict(nxt)))
# win-probability model, out-of-fold by match
b['crr']=b.crr.fillna(0)
b['rrr']=b.rrr.fillna(0)
b['proj']=b.score_before+b.E_rem
b['need']=b.need.fillna(0)
WF=['innings','score_before','wkts_before','balls_left','ipl','need','rrr','crr','proj']
def wp_at(df,model_cols=WF):
    return df[model_cols]
oof=np.zeros(len(b)); gkf=GroupKFold(5)
for tr,te in gkf.split(b,b.bat_won,b.match_id):
    mdl=HGC(max_iter=200,learning_rate=0.06,max_depth=5,min_samples_leaf=300,l2_regularization=1.0)
    mdl.fit(b.iloc[tr][WF],b.bat_won.iloc[tr]); oof[te]=mdl.predict_proba(b.iloc[te][WF])[:,1]
b['wp_before']=oof
print('WP AUC all',roc_auc_score(b.bat_won,oof),'Brier',brier_score_loss(b.bat_won,oof))
for i in (1,2):
    s=b.innings==i; print(' inn',i,roc_auc_score(b.bat_won[s],oof[s]))
# WP after = wp_before of next ball within same innings; last ball -> outcome (for 2nd innings; 1st innings last ball -> wp of start of chase = 1-wp_before(2nd inn ball1))
b['wp_after']=b.groupby(['match_id','innings']).wp_before.shift(-1)
last=b.wp_after.isna()
b.loc[last,'wp_after']=b.loc[last,'bat_won'].astype(float)
b['dwp']=b.wp_after-b.wp_before
# state leverage: expected |dWP| from state (hindsight free): regress |dwp| on state features OOF
b['absd']=b.dwp.abs()
oofL=np.zeros(len(b))
LF=['innings','balls_left','wkts_before','ipl','need','rrr','crr','proj','wp_before']
for tr,te in gkf.split(b,b.bat_won,b.match_id):
    r=HGR(max_iter=200,learning_rate=0.06,max_depth=5,min_samples_leaf=300); r.fit(b.iloc[tr][LF],b.absd.iloc[tr]); oofL[te]=r.predict(b.iloc[te][LF])
b['LI']=oofL/b.absd.mean()
# percentile of LI within innings => PI 0-100
b['PI']=b.groupby('innings').LI.rank(pct=True)*100
b.to_pickle('/home/claude/work/balls2.pkl')
print(b.groupby('innings').LI.describe())
print(b.groupby(pd.cut(b.PI,[0,50,75,90,100])).agg(n=('LI','size'),LI=('LI','mean'),rrr=('rrr','mean'),wk=('wkts_before','mean'),bl=('balls_left','mean')))
