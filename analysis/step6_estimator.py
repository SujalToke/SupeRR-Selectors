import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
bat=pd.read_pickle('/home/claude/work/bat.pkl'); bwl=pd.read_pickle('/home/claude/work/bwl.pkl')
mt=pd.read_pickle('/home/claude/work/matches.pkl').sort_values('match_date'); order={mid:i for i,mid in enumerate(mt.match_id)}
def agg(df,pid,col,cls):
    hp=df[df[cls]=='High'].groupby(pid)[col].agg(np_='count',sp='sum'); lo=df[df[cls].isin(['Low','Mid'])].groupby(pid)[col].agg(nn='count',sn='sum')
    return hp.join(lo,how='outer').fillna(0)
def test(df,pid,col,cls,label):
    res=[]
    for rep in range(30):                                   # random split of matches, repeated
        rng=np.random.default_rng(rep); mids=df.match_id.unique(); tr=set(rng.choice(mids,len(mids)//2,replace=False))
        A=agg(df[df.match_id.isin(tr)],pid,col,cls); B=agg(df[~df.match_id.isin(tr)],pid,col,cls)
        j=A.join(B,lsuffix='_a',rsuffix='_b',how='inner'); j=j[(j.np__a>=30)&(j.np__b>=30)]
        mu_p=A.sp.sum()/A.np_.sum(); mu_n=A.sn.sum()/A.nn.sum()
        for k in (50,150,300):
            lg=(j.sp_a+k*mu_p)/(j.np__a+k)                                  # (a) shrink to league
            usual=(j.sn_a+k*mu_n)/(j.nn_a+k); praw=j.sp_a/j.np__a
            # (b) partial pooling toward own usual: usual + shrunk(press-usual)
            for k2 in (50,150,300,600):
                own=usual+ (j.np__a*(praw-usual))/(j.np__a+k2)
                res.append((label,'own-usual',k,k2,np.corrcoef(own,j.sp_b/j.np__b)[0,1]))
            res.append((label,'league',k,0,np.corrcoef(lg,j.sp_b/j.np__b)[0,1]))
            res.append((label,'usual only',k,0,np.corrcoef(usual,j.sp_b/j.np__b)[0,1]))
        res.append((label,'raw',0,0,np.corrcoef(praw,j.sp_b/j.np__b)[0,1]))
    r=pd.DataFrame(res,columns=['who','est','k','k2','r']); print(r.groupby(['who','est','k','k2']).r.mean().round(3).to_string())
test(bat,'striker_id','v_adj','p_bat','bat'); test(bwl,'bowler_id','u_adj','p_bowl','bowl')
