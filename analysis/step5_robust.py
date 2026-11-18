import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/work/step4_ratings.py').read().split("# ---------- value per delivery")[0].split("b=pd.read_pickle")[0])  # imports only
bat=pd.read_pickle('/home/claude/work/bat.pkl'); bwl=pd.read_pickle('/home/claude/work/bwl.pkl'); RB,RW,K=pd.read_pickle('/home/claude/work/ratings.pkl')
src=open('/home/claude/work/step4_ratings.py').read()
fn=src[src.index('def build('):src.index("bat['innings']=bat.innings")]
exec(fn)
kb,kw=K['bat'],K['bowl']
def run(df,col,cls,k,pid,minp=90,minm=10,w=(0.6,0.2,0.2)):
    R=build(df,pid,None,col,cls,k,minp,minm)
    z=lambda s:(s-s.mean())/s.std()
    R['rating']=w[0]*z(R.press)+w[1]*z(R.delta)+w[2]*z(R.hit); return R
out={}
# ---- 1. split-half reliability (odd/even matches by date order)
mt=pd.read_pickle('/home/claude/work/matches.pkl').sort_values('match_date'); order={mid:i for i,mid in enumerate(mt.match_id)}
for nm,df,col,cls,k,pid in [('bat',bat,'v_adj','p_bat',kb,'striker_id'),('bowl',bwl,'u_adj','p_bowl',kw,'bowler_id')]:
    odd=df[df.match_id.map(order)%2==0]; even=df[df.match_id.map(order)%2==1]
    A=run(odd,col,cls,k,pid,40,4); B=run(even,col,cls,k,pid,40,4); j=A.join(B,lsuffix='_a',rsuffix='_b',how='inner')
    # raw (unshrunk) pressure value split-half too
    out[nm+'_split']={'n':len(j),'r_press':j.press_a.corr(j.press_b),'rho_rating':j.rating_a.corr(j.rating_b,method='spearman'),'rho_press':j.press_a.corr(j.press_b,method='spearman')}
    # usual-level reliability for comparison
    out[nm+'_split_usual']=j.usual_a.corr(j.usual_b)
print(out)
# ---- 2. sensitivity: weights, thresholds
base={'bat':RB.sort_values('rating',ascending=False).index[:10].tolist(),'bowl':RW.sort_values('rating',ascending=False).index[:10].tolist()}
sens={}
for nm,df,col,cls,k,pid in [('bat',bat,'v_adj','p_bat',kb,'striker_id'),('bowl',bwl,'u_adj','p_bowl',kw,'bowler_id')]:
    for lab,kw_ in {'press only (1,0,0)':dict(w=(1,0,0)),'equal (1/3 each)':dict(w=(1/3,1/3,1/3)),'(0.5,0.25,0.25)':dict(w=(.5,.25,.25)),'min 150 balls':dict(minp=150,minm=12)}.items():
        R=run(df,col,cls,k,pid,**kw_); top=R.sort_values('rating',ascending=False).index[:10].tolist()
        sens[(nm,lab)]=(len(set(top)&set(base[nm])),top)
for k_,v in sens.items(): print(k_,v[0])
# ---- 3. bootstrap over matches
rng=np.random.default_rng(7); res={}
for nm,df,col,cls,k,pid,R0 in [('bat',bat,'v_adj','p_bat',kb,'striker_id',RB),('bowl',bwl,'u_adj','p_bowl',kw,'bowler_id',RW)]:
    mids=df.match_id.unique(); idx_by={m:np.where(df.match_id.values==m)[0] for m in mids}
    ranks=pd.DataFrame(index=R0.index); top10=pd.Series(0,index=R0.index,dtype=float); rk=[]
    for r in range(120):
        samp=rng.choice(mids,len(mids)); idx=np.concatenate([idx_by[m] for m in samp]); d=df.iloc[idx]
        R=run(d,col,cls,k,pid,90,10); s=R.rating.rank(ascending=False)
        rk.append(s.reindex(R0.index)); 
    rk=pd.concat(rk,axis=1); res[nm]=pd.DataFrame({'p_top10':(rk<=10).sum(1)/rk.notna().sum(1),'med_rank':rk.median(1),'p10':rk.quantile(.1,axis=1),'p90':rk.quantile(.9,axis=1),'n_eligible_share':rk.notna().mean(1)})
pd.to_pickle((out,sens,base,res),'/home/claude/work/robust.pkl')
for nm,R0 in [('bat',RB),('bowl',RW)]:
    t=R0.sort_values('rating',ascending=False).head(14)
    print(res[nm].loc[t.index].assign(name=t.name).round(2))
