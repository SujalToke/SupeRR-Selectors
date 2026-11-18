import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
D="/home/claude/data/Data Pack/"
bat=pd.read_pickle('/home/claude/work/bat.pkl'); bwl=pd.read_pickle('/home/claude/work/bwl.pkl'); RB,RW,K=pd.read_pickle('/home/claude/work/ratings.pkl'); out,sens,base,boot=pd.read_pickle('/home/claude/work/robust.pkl')
mp=pd.read_csv(D+'competition_team_player_mapping.csv')
# team label: IPL franchise in latest IPL they appeared in, else national team
mp=mp.merge(pd.read_csv(D+'competitions.csv')[['comp_id','season']],on='comp_id'); mp['ipl']=mp.comp_name.str.startswith('IPL')
def team(pid):
    d=mp[mp.player_id==pid]; i=d[d.ipl].sort_values('season').tail(1)
    nat=d[~d.ipl].team_name.mode()
    return (i.team_name.iloc[0] if len(i) else '')+ (' / ' if len(i) and len(nat) else '') + (nat.iloc[0] if len(nat) else '')
def detail_bat(pid):
    d=bat[bat.striker_id==pid]; h=d[d.p_bat=='High']; l=d[d.p_bat.isin(['Low','Mid'])]
    hl=h[h.legal]; ll=l[l.legal]
    return dict(inn_total=d.match_id.nunique(), p_balls=len(h), p_runs=int(h.bat_runs.sum()), p_sr=h.bat_runs.sum()/len(h)*100, u_sr=l.bat_runs.sum()/len(l)*100,
                p_dis=int(h.wk.sum()), p_bpd=len(h)/max(h.wk.sum(),1), p_bound=(h.is_four|h.is_six).mean()*100, p_dot=(h.bat_runs==0).mean()*100,
                ipl_share=(h.level=='IPL').mean()*100)
def detail_bowl(pid):
    d=bwl[bwl.bowler_id==pid]; h=d[d.p_bowl=='High']; l=d[d.p_bowl.isin(['Low','Mid'])]
    def econ(x): return x.bowler_runs.sum()/max(x.legal.sum(),1)*6
    return dict(p_balls=int(h.legal.sum()), p_econ=econ(h), u_econ=econ(l), p_wk=int(h.bowler_wk.sum()), p_bpw=h.legal.sum()/max(h.bowler_wk.sum(),1),
                p_dot=(h[h.legal].bowler_runs==0).mean()*100, ipl_share=(h.level=='IPL').mean()*100, pace=h.pace_or_spin.mode().iloc[0] if len(h) else '')
topB=RB.sort_values('rating',ascending=False).head(15).copy(); topW=RW.sort_values('rating',ascending=False).head(15).copy()
for t,f in ((topB,detail_bat),(topW,detail_bowl)):
    det=pd.DataFrame([f(i) for i in t.index],index=t.index); 
    for c in det: t[c]=det[c]
    t['team']=[team(i) for i in t.index]
topB=topB.join(boot['bat'][['p_top10','med_rank','p10','p90']]); topW=topW.join(boot['bowl'][['p_top10','med_rank','p10','p90']])
topB['rank']=range(1,16); topW['rank']=range(1,16)
pd.to_pickle((topB,topW),'/home/claude/work/top.pkl')
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
print(topB.head(12)[['rank','name','team','n_p','p_balls','p_runs','p_sr','u_sr','p_dis','p_bound','press','usual','delta','hit','rating','p_top10','med_rank']].round(2).to_string())
print(topW.head(12)[['rank','name','team','p_balls','p_econ','u_econ','p_wk','p_bpw','press','usual','delta','hit','rating','p_top10','med_rank']].round(2).to_string())
# name check vs mapping
print(mp[mp.player_id.isin(list(topB.index[:12])+list(topW.index[:12]))].drop_duplicates('player_id')[['player_id','player_name']].to_string())
# overall eligibility counts
print(len(RB),len(RW))
