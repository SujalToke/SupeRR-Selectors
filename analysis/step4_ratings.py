import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
b=pd.read_pickle('/home/claude/work/balls3.pkl')
# ---------- bowling-side pressure index (mirror of batting-side resource terms)
D=(b.need/b.E_rem.clip(lower=1))
ease=((1.3-D)/0.9).clip(0,1)
par=np.where(b.level=='IPL',187,173)
S_hand=((b.E_w-b.wkts_before)/3).clip(0,1)
S_ahead=((b.proj-par)/40).clip(0,1)
S_col=(b.w_recent/3).clip(0,1)
PIb2=100*(0.45*b.closeness+0.30*ease+0.25*b.lateness)
PIb1=100*(0.30*S_hand+0.25*S_ahead+0.25*b.lateness+0.20*(1-S_col))
b['PI_bowl']=np.where(b.innings==2,PIb2,PIb1)
b['PIb_pct']=b.PI_bowl.where(~b.decided).groupby(b.innings).rank(pct=True)
b['p_bowl']=np.where(b.decided,'Decided',np.where(b.PIb_pct>=0.75,'High',np.where(b.PIb_pct<=0.50,'Low','Mid')))
b=b.rename(columns={'pressure':'p_bat'})
b['dE']=b.E_rem_next-b.E_rem
# ---------- value per delivery
nonstriker_out=b.wk&(b.dismissed_player_id!=b.striker_id)
bat=b[(b.extras_type!='wide')&~nonstriker_out].copy()
bat['v']=bat.bat_runs+bat.dE
bwl=b[~(b.wk&~b.bowler_wk)].copy()                 # drop run-outs etc. not credited to bowler
bwl['u']=-(bwl.bowler_runs+bwl.dE)                  # runs saved vs expectation (positive = good)
# ---------- opponent-quality adjustment (one pass, shrunk, using all balls)
def shrunk_mean(df,key,col,k):
    g=df.groupby(key)[col].agg(['sum','count']); return g['sum']/(g['count']+k)
bq=shrunk_mean(bwl,'bowler_id','u',300)             # bowler quality (runs saved/ball)
aq=shrunk_mean(bat,'striker_id','v',300)            # batter quality
bat['v_adj']=bat.v+bat.bowler_id.map(bq).fillna(0)*1.0   # facing a better bowler -> credit
bwl['u_adj']=bwl.u-bwl.striker_id.map(aq).fillna(0)*1.0  # bowling to a better batter -> credit
# ---------- empirical-Bayes shrinkage constants from variance components
def ebk(df,key,col,cls,col_cls):
    d=df[df[col_cls]==cls]; g=d.groupby(key)[col].agg(['mean','count','var']); g=g[g['count']>=40]
    s2=d[col].var(); tau2=max(g['mean'].var()-(s2/g['count']).mean(),1e-6); return s2/tau2, s2, tau2
kb,s2b,t2b=ebk(bat,'striker_id','v_adj','High','p_bat'); kw,s2w,t2w=ebk(bwl,'bowler_id','u_adj','High','p_bowl')
print('shrinkage k bat',round(kb),'bowl',round(kw))
K={'bat':kb,'bowl':kw}
def build(df,pid,pname,col,cls_col,K,minp,minm,matches=None):
    hp=df[df[cls_col]=='High']; lo=df[df[cls_col].isin(['Low','Mid'])]
    mu_p=hp[col].mean(); mu_n=lo[col].mean()
    P=hp.groupby(pid)[col].agg(n_p='count',sum_p='sum'); N=lo.groupby(pid)[col].agg(n_n='count',sum_n='sum')
    # per-innings pressure
    inn=hp.groupby([pid,'match_id',col.split('_')[0]=='v' and 'innings' or 'innings'])[col].agg(['sum','count']).reset_index()
    inn=inn[inn['count']>=6]; inn['pos']=inn['sum']>0
    C=inn.groupby(pid).pos.agg(n_inn='count',pos='sum')
    R=P.join(N,how='inner').join(C,how='left').fillna({'n_inn':0,'pos':0})
    R['press']=(R.sum_p+K*mu_p)/(R.n_p+K)                    # shrunk pressure value per ball
    R['usual']=(R.sum_n+K*mu_n)/(R.n_n+K)                    # shrunk usual (non-high) value per ball
    R['delta']=R.press-R.usual                               # resilience
    pri=(inn.pos.mean() if len(inn) else .5)
    R['hit']=(R.pos+8*pri)/(R.n_inn+8)                        # shrunk share of pressure innings with positive value
    R=R[(R.n_p>=minp)&(R.n_inn>=minm)]
    z=lambda s:(s-s.mean())/s.std()
    R['rating']=0.60*z(R.press)+0.20*z(R.delta)+0.20*z(R.hit)
    return R
bat['innings']=bat.innings; bwl['innings']=bwl.innings
RB=build(bat,'striker_id','striker_name','v_adj','p_bat',kb,90,10)
RW=build(bwl,'bowler_id','bowler_name','u_adj','p_bowl',kw,90,10)
names=pd.concat([bat[['striker_id','striker_name']].rename(columns={'striker_id':'id','striker_name':'name'}),bwl[['bowler_id','bowler_name']].rename(columns={'bowler_id':'id','bowler_name':'name'})]).drop_duplicates('id').set_index('id').name
RB['name']=RB.index.map(names); RW['name']=RW.index.map(names)
print(len(RB),len(RW))
pd.to_pickle((RB,RW,K),'/home/claude/work/ratings.pkl')
bat.to_pickle('/home/claude/work/bat.pkl'); bwl.to_pickle('/home/claude/work/bwl.pkl')
pd.set_option('display.width',200)
print(RB.sort_values('rating',ascending=False).head(20)[['name','n_p','n_inn','press','usual','delta','hit','rating']].round(3))
print(RW.sort_values('rating',ascending=False).head(20)[['name','n_p','n_inn','press','usual','delta','hit','rating']].round(3))
