import pandas as pd, numpy as np
D="/home/claude/data/Data Pack/"
b=pd.read_csv(D+'ball_by_ball.csv',keep_default_na=False,na_values=[''])
m=pd.read_csv(D+'matches.csv'); c=pd.read_csv(D+'competitions.csv')
m=m.merge(c,on='comp_id',how='left')
# scope: exclude A-team / emerging tournaments, super overs, rain-revised matches, no-result
bad_comp=c[c.comp_name.str.contains('Emerging|Rising Stars',case=False)].comp_id
m=m[~m.comp_id.isin(bad_comp)]
m=m[m.first_inn_revised_overs.isna() & m.second_inn_revised_overs.isna() & m.revised_target.isna()]
m=m[m.win_team_id.notna()]
m['level']=np.where(m.comp_name.str.startswith('IPL'),'IPL','T20I')
b=b[(b.innings<=2)&b.match_id.isin(m.match_id)].copy()
b=b.merge(m[['match_id','level','win_team_id','match_date','comp_name']],on='match_id')
b=b.sort_values(['match_id','innings','innings_ball_num']).reset_index(drop=True)
b['legal']=~b.extras_type.isin(['wide','Overstep','other'])
b['team_runs']=b.runs+b.extras1_runs+b.extras_2_runs
b['bat_runs']=b.runs
b['bowler_runs']=b.runs+np.where(b.extras_type.isin(['wide','Overstep','other']),b.extras1_runs+b.extras_2_runs,0)
b['wk']=b.is_wicket & ~b.dismissal_type.isin(['Retire Out'])
b['bowler_wk']=b.wk & b.dismissal_type.isin(['Caught','Bowled','LBW','Stump','Hit Wicket'])
g=b.groupby(['match_id','innings'])
b['score_before']=g.team_runs.cumsum()-b.team_runs
b['wkts_before']=g.wk.cumsum()-b.wk
b['legal_before']=g.legal.cumsum()-b.legal
b['balls_left']=(120-b.legal_before).clip(lower=0)
tot=g.team_runs.sum().rename('inn_total').reset_index()
t1=tot[tot.innings==1][['match_id','inn_total']].rename(columns={'inn_total':'first_total'})
b=b.merge(t1,on='match_id',how='left')
b['target']=np.where(b.innings==2,b.first_total+1,np.nan)
b['need']=b.target-b.score_before
b['rrr']=np.where((b.innings==2)&(b.balls_left>0),b.need/b.balls_left*6,np.nan)
b['crr']=np.where(b.legal_before>0,b.score_before/b.legal_before*6,np.nan)
b['bat_won']=(b.batting_team_id==b.win_team_id).astype(int)
# drop deliveries after chase completed / innings > 120 legal
b=b[b.balls_left>0]
b=b[~((b.innings==2)&(b.need<=0))]
b['phase']=pd.cut(b.legal_before,[-1,35,89,200],labels=['PP','Middle','Death'])
b.to_pickle('/home/claude/work/balls.pkl'); m.to_pickle('/home/claude/work/matches.pkl')
print(len(b), b.match_id.nunique(), b.level.value_counts().to_dict(), b.groupby('innings').size().to_dict())
print(b.groupby(['level','innings']).team_runs.sum()/b.groupby(['level','innings']).legal.sum()*6)
