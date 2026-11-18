import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
bat=pd.read_pickle('bat.pkl'); bwl=pd.read_pickle('bwl.pkl'); RB,RW,K=pd.read_pickle('ratings.pkl'); out,sens,base,boot=pd.read_pickle('robust.pkl'); b3=pd.read_pickle('balls3.pkl')
O='/home/claude/pkg/data/'
def db(pid):
    d=bat[bat.striker_id==pid]; h=d[d.p_bat=='High']; l=d[d.p_bat.isin(['Low','Mid'])]
    return dict(hp_balls=len(h),hp_runs=int(h.bat_runs.sum()),hp_sr=h.bat_runs.sum()/len(h)*100,usual_sr=l.bat_runs.sum()/len(l)*100,hp_outs=int(h.wk.sum()),hp_boundary_pct=(h.is_four|h.is_six).mean()*100,hp_dot_pct=(h.bat_runs==0).mean()*100)
def dw(pid):
    d=bwl[bwl.bowler_id==pid]; h=d[d.p_bowl=='High']; l=d[d.p_bowl.isin(['Low','Mid'])]
    e=lambda x:x.bowler_runs.sum()/max(x.legal.sum(),1)*6
    return dict(hp_balls=int(h.legal.sum()),hp_econ=e(h),usual_econ=e(l),hp_wkts=int(h.bowler_wk.sum()),hp_dot_pct=(h[h.legal].bowler_runs==0).mean()*100)
for R,f,nm,bt in ((RB,db,'batters',boot['bat']),(RW,dw,'bowlers',boot['bowl'])):
    R=R.copy(); R['rank']=R.rating.rank(ascending=False).astype(int)
    det=pd.DataFrame([f(i) for i in R.index],index=R.index); R=R.join(det).join(bt[['p_top10','med_rank','p10','p90']])
    R=R.drop(columns=['sum_p','sum_n','pos']).rename(columns={'n_p':'hp_balls_all','n_inn':'hp_innings','press':'pressure_value','usual':'usual_value','hit':'hit_rate'})
    R.sort_values('rank').to_csv(O+nm+'.csv')
# PI by over, validation
b3['over']=(b3.legal_before//6)+1
b3[~b3.decided].groupby(['innings','over']).PI.mean().reset_index().to_csv(O+'pi_by_over.csv',index=False)
l=b3[b3.legal&b3.pressure.isin(['High','Mid','Low'])]
v=l.groupby('pressure').agg(balls=('legal','size'),wicket_rate=('wk',lambda s:s.mean()*100),dot_pct=('bat_runs',lambda s:(s==0).mean()*100),avg_wkts_lost=('wkts_before','mean'),avg_balls_left=('balls_left','mean'),run_rate=('bat_runs',lambda s:s.mean()*6)).reset_index(); v.to_csv(O+'validation.csv',index=False)
# chase heatmap: mean PI by balls_left bucket x wickets lost
c=b3[(b3.innings==2)&~b3.decided].copy(); c['bl']=pd.cut(c.balls_left,[0,12,24,36,60,90,120],labels=['1-12','13-24','25-36','37-60','61-90','91-120'])
c.pivot_table(index='wkts_before',columns='bl',values='PI',aggfunc='mean').round(1).to_csv(O+'chase_heatmap.csv')
# sensitivity
pd.DataFrame([(k[0],k[1],v[0]) for k,v in sens.items()],columns=['role','scenario','overlap_with_final_top10']).to_csv(O+'sensitivity.csv',index=False)
print('ok')
