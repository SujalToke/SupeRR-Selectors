import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')
bat=pd.read_pickle('/home/claude/work/bat.pkl'); bwl=pd.read_pickle('/home/claude/work/bwl.pkl'); topB,topW=pd.read_pickle('/home/claude/work/top.pkl'); RB,RW,K=pd.read_pickle('/home/claude/work/ratings.pkl')
b3=pd.read_pickle('/home/claude/work/balls3.pkl')
NAVY='#1B2A5B'; PINK='#E91E8C'; GREY='#8A94A6'; LIGHT='#C9D1E6'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
# --- validation stats
def val(df,cls):
    l=df[df.legal]; g=l.groupby(['innings',cls]).agg(balls=('legal','size'),rr=('bat_runs','mean'),dot=('bat_runs',lambda s:(s==0).mean()*100),wk=('wk','mean'),wl=('wkts_before','mean'),bl=('balls_left','mean'),rrr=('rrr',lambda s:s[s>0].median()))
    g['rr']*=6; g['wk']*=100; return g.round(2)
vb=val(b3,'pressure'); vb.to_csv('/home/claude/work/val_bat.csv'); print(vb)
# fig1: PI by over
b3['over']=(b3.legal_before//6)+1
fig,ax=plt.subplots(1,2,figsize=(7.2,2.6))
for i,(inn,t) in enumerate([(1,'1st innings (setting)'),(2,'2nd innings (chasing)')]):
    d=b3[(b3.innings==inn)&~b3.decided].groupby('over').PI.mean(); ax[i].plot(d.index,d.values,color=NAVY,lw=2)
    thr=b3[(b3.innings==inn)&(b3.pressure=='High')].PI.min(); ax[i].axhline(thr,color=PINK,ls='--',lw=1); ax[i].text(1,thr+0.6,'High-pressure threshold (top 25%)',color=PINK,fontsize=7.5)
    ax[i].set_title(t,fontsize=9); ax[i].set_xlabel('Over'); ax[i].set_ylabel('Mean batting Pressure Index' if i==0 else '')
plt.tight_layout(); plt.savefig('/home/claude/work/f1.png',dpi=200); plt.close()
# fig2: High vs Low contrast
l=b3[b3.legal&(b3.pressure.isin(['High','Low']))]
fig,ax=plt.subplots(1,3,figsize=(7.2,2.4))
for a,(c,t,f) in zip(ax,[('wk','Wicket rate\n(per 100 balls)',100),('bat_runs','Dot-ball %',None),('wkts_before','Avg wickets already\nlost',1)]):
    vals=[]
    for p in ['Low','High']:
        d=l[l.pressure==p]; vals.append((d.wk.mean()*100) if c=='wk' else ((d.bat_runs==0).mean()*100 if c=='bat_runs' else d.wkts_before.mean()))
    a.bar(['Ordinary','High\npressure'],vals,color=[LIGHT,PINK],width=.6); a.set_title(t,fontsize=8.5)
    for x,v in enumerate(vals): a.text(x,v,f'{v:.1f}',ha='center',va='bottom',fontsize=8)
plt.tight_layout(); plt.savefig('/home/claude/work/f2.png',dpi=200); plt.close()
# fig3/4: dumbbell top 10
def dumb(top,fn,title,lab):
    t=top.head(10).iloc[::-1]; fig,ax=plt.subplots(figsize=(7.2,3.6)); y=np.arange(len(t))
    for yi,(u,p) in enumerate(zip(t.usual,t.press)): ax.plot([u,p],[yi,yi],color=LIGHT,lw=3,zorder=1)
    ax.scatter(t.usual,y,color=GREY,s=45,zorder=2,label='Usual (non-high-pressure)'); ax.scatter(t.press,y,color=PINK,s=55,zorder=3,label='Under high pressure')
    ax.axvline(0,color='k',lw=.6); ax.set_yticks(y); ax.set_yticklabels([f'{int(r)}. {n}' for r,n in zip(t['rank'],t.name)]); ax.set_xlabel(lab); ax.legend(frameon=False,loc='lower right',fontsize=8); ax.set_title(title,fontsize=10,loc='left')
    plt.tight_layout(); plt.savefig(fn,dpi=200); plt.close()
dumb(topB,'/home/claude/work/f3.png','Top 10 batters: value per ball, usual vs under pressure','Runs above expectation per ball (shrunk, opponent-adjusted)')
dumb(topW,'/home/claude/work/f4.png','Top 10 bowlers: value per ball, usual vs under pressure','Runs saved vs expectation per ball (shrunk, opponent-adjusted)')
# fig5: bootstrap rank intervals
fig,ax=plt.subplots(1,2,figsize=(7.2,3.2))
for a,(top,t) in zip(ax,[(topB,'Batters'),(topW,'Bowlers')]):
    d=top.head(10).iloc[::-1]; y=np.arange(len(d))
    a.hlines(y,d.p10,d.p90,color=LIGHT,lw=4); a.scatter(d.med_rank,y,color=NAVY,zorder=3,s=25); a.set_yticks(y); a.set_yticklabels(d.name,fontsize=8); a.set_xlabel('Rank across 120 bootstrap resamples\n(dot = median, bar = 10th–90th pct)',fontsize=8); a.set_title(t,fontsize=9); a.axvline(10.5,color=PINK,ls='--',lw=1); a.set_xlim(0,45)
plt.tight_layout(); plt.savefig('/home/claude/work/f5.png',dpi=200); plt.close()
# scope numbers
print(b3.match_id.nunique(), b3.groupby('level').match_id.nunique().to_dict(), len(b3), b3.legal.sum(), (b3.pressure=='High').sum())
print(len(RB),len(RW), bat.striker_id.nunique(), bwl.bowler_id.nunique())
print(b3.groupby('pressure').size())
