import pandas as pd, numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
topB,topW=pd.read_pickle('/home/claude/work/top.pkl'); out,sens,base,boot=pd.read_pickle('/home/claude/work/robust.pkl')
NAVY=colors.HexColor('#1B2A5B'); PINK=colors.HexColor('#E91E8C'); LIGHT=colors.HexColor('#EEF1F8')
ss=getSampleStyleSheet()
H1=ParagraphStyle('H1',fontName='Helvetica-Bold',fontSize=13,textColor=NAVY,spaceBefore=10,spaceAfter=4)
H2=ParagraphStyle('H2',fontName='Helvetica-Bold',fontSize=10,textColor=PINK,spaceBefore=6,spaceAfter=2)
B=ParagraphStyle('B',fontName='Helvetica',fontSize=8.8,leading=12.2,spaceAfter=4)
S=ParagraphStyle('S',parent=B,fontSize=7.6,leading=10,textColor=colors.HexColor('#444444'))
T=ParagraphStyle('T',fontName='Helvetica-Bold',fontSize=21,textColor=NAVY,leading=25)
BL=ParagraphStyle('BL',parent=B,leftIndent=10,bulletIndent=0,spaceAfter=1.5)
def bl(t): return Paragraph(t,BL,bulletText='•')
def tbl(data,cw,hdr=True,fs=7.4,zebra=True):
    t=Table(data,colWidths=cw,repeatRows=1)
    st=[('FONT',(0,0),(-1,-1),'Helvetica',fs),('FONT',(0,0),(-1,0),'Helvetica-Bold',fs),('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('GRID',(0,0),(-1,-1),0.25,colors.HexColor('#C9D1E6')),('TOPPADDING',(0,0),(-1,-1),2.2),('BOTTOMPADDING',(0,0),(-1,-1),2.2)]
    if zebra: st+= [('BACKGROUND',(0,i),(-1,i),LIGHT) for i in range(2,len(data),2)]
    t.setStyle(TableStyle(st)); return t
def P(t,s=B): return Paragraph(t,s)
def img(f,w): 
    from PIL import Image as PI; im=PI.open(f); return Image(f,width=w,height=w*im.size[1]/im.size[0])
W=A4[0]-30*mm
doc=SimpleDocTemplate('/mnt/user-data/outputs/SupeRR_Selector_S2_Round1_Methodology_Report.pdf',pagesize=A4,leftMargin=15*mm,rightMargin=15*mm,topMargin=14*mm,bottomMargin=14*mm,title='SupeRR Selector S2 - Round 1 - Pressure Performance Methodology',author='Participant')
def foot(c,d):
    c.saveState(); c.setFont('Helvetica',7); c.setFillColor(colors.grey); c.drawString(15*mm,8*mm,'SupeRR Selector Season 2  |  Round 1: Most Reliable Players Under Pressure'); c.drawRightString(A4[0]-15*mm,8*mm,f'Page {d.page}'); c.restoreState()
s=[]
s+= [P('Who Delivers When It Matters?',T),P('A Pressure Index and Pressure Rating system for IPL and T20I batters and bowlers (2022 onwards)',ParagraphStyle('sub',parent=B,fontSize=10,textColor=colors.HexColor('#555555'))),Spacer(1,4)]
s+= [P('Final rankings',H1)]
rows=[['Rank','Batter','Bowler']]+[[str(i+1),f"{topB.name.iloc[i]}",f"{topW.name.iloc[i]}"] for i in range(10)]
s+= [tbl(rows,[16*mm,W/2-8*mm,W/2-8*mm],fs=8.5),Spacer(1,5)]
s+= [P('<b>Summary.</b> We define pressure ball-by-ball with a transparent 0-100 <b>Pressure Index (PI)</b> built from the match situation (closeness of the contest, difficulty of the equation, how late in the innings it is, wickets in hand or lost, recent collapses and the projected score against par). The top quarter of deliveries in each innings are labelled <b>high pressure</b>. For every delivery we measure <b>runs above (or saved against) expectation</b> given balls left, wickets lost and competition level, adjust for opponent quality, and shrink each player\'s pressure value towards the league mean using empirical-Bayes weights estimated from the data. A player\'s <b>Pressure Rating</b> combines (60%) how good they are under pressure, (20%) how much they hold up against their own usual level, and (20%) how often they deliver in individual pressure innings. Ratings are checked with split-sample predictions, a 120-run bootstrap over matches, and weight and threshold sensitivity tests.')]
s+= [P('Scope: 788 completed matches (357 IPL, 431 T20I), 188,151 deliveries, 40,506 high-pressure deliveries. 85 batters and 116 bowlers met the eligibility thresholds.',S)]
# 1 definition
s+= [P('1. How we define pressure',H1),
 P('Pressure is not one number in a T20 game. A required rate of 11 means something different with 8 wickets in hand than with 3, and something different again with 60 balls left than with 6. The first innings has no target, so the signals differ. We therefore build two context-specific indices on the same 0-100 scale. Each has a <b>batting-side</b> version (pressure on the team that is batting) and a mirrored <b>bowling-side</b> version (pressure on the team defending), because resource terms flip sign between the two sides: a collapse is pressure on batters but relief for bowlers, while set batters with wickets in hand are pressure on bowlers.'),
 P('Supporting models (all fitted with out-of-fold predictions grouped by match to avoid leakage)',H2),
 bl('<b>Expected remaining runs E(balls left, wickets lost, level)</b>: gradient-boosted regression trained on first-innings states, separately aware of IPL vs T20I scoring environments. This is our resource measure, similar in spirit to DLS.'),
 bl('<b>Chase win probability</b>: gradient-boosted classifier on score, wickets, balls left, required rate, current rate and projected score (out-of-fold AUC 0.89 in the second innings).')]
comp=[['','Chase (2nd innings)','First innings'],
 ['Batting-side PI','45% Closeness: 1 - |2p - 1|, p = chase win probability\n30% Difficulty: clip((runs needed / E[remaining] - 0.7) / 0.9, 0, 1)\n25% Lateness: clip((balls bowled - 30) / 90, 0, 1)','30% Collapse: wickets in last ~20 deliveries / 3 (cap 1)\n25% Depletion: (wickets lost - league-typical wickets at this stage) / 3\n25% Lateness (as left)\n20% Par shortfall: (par - projected total) / 40; par = 187 IPL, 173 T20I'],
 ['Bowling-side PI','45% Closeness (as above)\n30% Ease: clip((1.3 - runs needed / E[remaining]) / 0.9, 0, 1)\n25% Lateness','30% Wickets in hand vs typical\n25% Projected total above par\n25% Lateness\n20% No recent collapse (1 - Collapse)']]
cm=[[Paragraph(str(c).replace('\n','<br/>'),ParagraphStyle('c',parent=S,textColor=colors.black if (i>0 and j>0) or j==0 and i>0 else colors.white)) for j,c in enumerate(r)] for i,r in enumerate(comp)]
s+= [tbl(cm,[24*mm,(W-24*mm)/2,(W-24*mm)/2],fs=7.2,zebra=False),Spacer(1,3),
 P('<b>Classes.</b> Within each innings, deliveries in the top 25% of PI are <b>High pressure</b>; the bottom 50% are <b>Low (ordinary)</b>; the rest are Mid. Chase deliveries where the win probability is already below 5% or above 95% are labelled <b>Decided</b> and excluded from both groups, because a decided game is a different situation from a tense one. A player\'s <b>usual</b> level is measured on all non-high-pressure deliveries (Low + Mid).'),
 P('Does the index separate pressure from ordinary situations?',H2),
 P('Yes. On batting-side classes, high-pressure deliveries show clearly higher wicket rates (6.6 vs 4.8 per 100 balls), more wickets already lost (4.4 vs 1.3 on average), fewer balls remaining (44 vs 81) and a harder chase equation (median required rate 10.6 vs 8.2 in the second innings), with a higher dot-ball share in the first innings. Mean PI rises steadily through each innings, as Figure 1 shows.'),
 img('/home/claude/work/f1.png',W*0.92),img('/home/claude/work/f2.png',W*0.92),
 P('Figure 1: Mean batting-side PI by over with the high-pressure cut-off (top). Figure 2: High-pressure vs ordinary deliveries on independent outcomes not used to build PI (wicket rate, dot balls, wickets already lost; legal deliveries).',S)]
# 2 measurement
s+= [P('2. How performance is measured',H1),
 P('Value per delivery (the same expectation framework for batters and bowlers)',H2),
 bl('<b>Batter value</b> = runs off the bat + E(next state) - E(current state). The second term charges a dismissal its true cost in expected runs, which is larger in the middle overs than at the death, so a wicket at ball 100 hurts less than one at ball 40. Wides are excluded; dismissals that were not the striker\'s (non-striker run-outs) are dropped.'),
 bl('<b>Bowler value</b> = -(runs conceded + E(next state) - E(current state)). Wickets are rewarded through the same state change; run-outs are not credited to the bowler. No-balls and wides cost the bowler runs without advancing the over.'),
 bl('<b>Context handling</b>: E is conditioned on balls left, wickets lost and competition level, so a six at the death is not over-credited and a quiet middle-over is not over-penalised. IPL and T20I use separate environments.'),
 bl('<b>Opponent adjustment</b>: each delivery is credited with the shrunk quality of the opponent faced (bowler quality for batters, batter quality for bowlers), so runs against elite bowlers count for more.'),
 P('Player baselines and rating',H2),
 bl('<b>Shrinkage</b>: observed pressure value per ball is shrunk towards the league high-pressure mean, with strength k = noise variance / true-talent variance estimated from the data (k = 244 balls for batters, 151 for bowlers). Small samples are pulled in, large samples are trusted.'),
 bl('<b>Pressure value</b> (press): shrunk value per ball in high-pressure deliveries. <b>Usual value</b>: same on non-high-pressure deliveries. <b>Resilience</b> (delta) = press - usual, i.e. performance under pressure compared with the player\'s own usual level, as the problem statement asks.'),
 bl('<b>Consistency (hit rate)</b>: share of the player\'s high-pressure innings (at least 6 high-pressure balls) that produced positive value, shrunk toward the league average with an 8-innings prior. This rewards performing repeatedly, not once.'),
 bl('<b>Pressure Rating</b> = 0.60 z(press) + 0.20 z(delta) + 0.20 z(hit rate), z-scored among eligible players. The weighting favours absolute performance under pressure, while resilience and consistency separate players who merely inflate numbers in a few matches.')]
# 3 fairness
s+= [P('3. Fair comparisons',H1),
 bl('<b>Eligibility</b>: at least 90 high-pressure deliveries (15 overs equivalent) and at least 10 high-pressure innings, in either role. This gave 85 batters and 116 bowlers.'),
 bl('<b>IPL vs T20I</b>: both are pooled because several of the most-watched players split time between them. Expectations are level-specific, and opponent adjustment recognises that bowlers and batters in T20Is face a wider range of quality. Pressure deliveries are labelled within each innings so neither competition dominates the definition.'),
 bl('<b>Data cleaning</b>: Super Over innings, rain-revised matches (25), no-result matches, and A-team tournaments (Emerging Teams Asia Cup 2024, Asia Cup Rising Stars 2025) are excluded. Overstep and "other" extras are treated as illegal deliveries.'),
 bl('<b>Role balance</b>: batting pressure uses balls faced; bowling pressure uses balls bowled. All-rounders can appear in both lists on the evidence of each role separately.')]
# 4 results
ABB={'Chennai Super Kings':'CSK','Mumbai Indians':'MI','Rajasthan Royals':'RR','Royal Challengers Bengaluru':'RCB','Punjab Kings':'PBKS','Kolkata Knight Riders':'KKR','Sunrisers Hyderabad':'SRH','Delhi Capitals':'DC','Lucknow Super Giants':'LSG','Gujarat Titans':'GT'}
def ab(t):
    for k,v in ABB.items(): t=t.replace(k,v)
    return t
def fmt(x,d=2): return f'{x:.{d}f}'
rb=[['#','Player','Team (IPL / national)','HP balls','HP SR','Usual SR','Out','Press','Delta','Hit %','Rating','P(top10)']]
for i in range(10):
    r=topB.iloc[i]; rb.append([str(i+1),r['name'],ab(r.team),str(int(r.p_balls)),fmt(r.p_sr,1),fmt(r.u_sr,1),str(int(r.p_dis)),fmt(r.press),fmt(r.delta),fmt(r.hit*100,0),fmt(r.rating),fmt(r.p_top10*100,0)+'%'])
rw=[['#','Player','Team (IPL / national)','HP balls','HP econ','Usual econ','Wkts','Balls/wkt','Press','Delta','Hit %','Rating','P(top10)']]
for i in range(10):
    r=topW.iloc[i]; rw.append([str(i+1),r['name'],ab(r.team),str(int(r.p_balls)),fmt(r.p_econ),fmt(r.u_econ),str(int(r.p_wk)),fmt(r.p_bpw,1),fmt(r.press),fmt(r.delta),fmt(r.hit*100,0),fmt(r.rating),fmt(r.p_top10*100,0)+'%'])
s+= [P('4. Evidence for the top 10',H1),P('Top 10 batters',H2),
 tbl(rb,[6*mm,28*mm,43*mm,12*mm,11*mm,12*mm,8*mm,11*mm,10*mm,9*mm,11*mm,13*mm],fs=6.7),
 P('HP = high pressure. SR = runs per 100 bat-faced deliveries. Out = dismissals under pressure. Press / Delta = shrunk value per ball (runs above expectation) and its gap to the player\'s usual level. Hit % = share of pressure innings with positive value. P(top10) = share of bootstrap resamples (in which the player was eligible) that kept them in the top 10.',S),
 img('/home/claude/work/f3.png',W*0.95),
 P('Top 10 bowlers',H2),
 tbl(rw,[6*mm,28*mm,40*mm,11*mm,10*mm,11*mm,8*mm,11*mm,9*mm,9*mm,8*mm,11*mm,13*mm],fs=6.5),
 P('Econ = runs conceded per over including extras. Wkts = bowler-credited wickets under pressure. Other columns as above.',S),
 img('/home/claude/work/f4.png',W*0.95)]
s+= [P('What stands out',H2),
 bl('<b>Suryakumar Yadav</b> leads batters on volume and quality: 504 high-pressure balls at a strike rate of 167.7 against 162.4 usually, only 15 dismissals, and the most stable position across resamples among high-volume players.'),
 bl('<b>Daryl Mitchell, Yashasvi Jaiswal and Fakhar Zaman</b> are the biggest risers from their usual level: each is clearly better under pressure than otherwise (delta 0.15 to 0.21 runs per ball). Their samples are smaller (160-200 balls), so shrinkage holds their ratings back and we treat their positions as less certain than Suryakumar\'s.'),
 bl('<b>Tim David and Heinrich Klaasen</b> are elite finishers whose pressure form is near their (very high) usual level; they rank on level, not on rise.'),
 bl('<b>Sikandar Raza</b> tops the bowlers: economy 6.94 under pressure against 7.45 usually, with 19 wickets in 263 balls. <b>Jasprit Bumrah</b> is the most robust profile (352 balls, pressure value 0.28 per ball, top-10 in 82% of resamples). <b>Mitchell Starc</b> and <b>Ravichandran Ashwin</b> show the largest improvements from their usual level (delta 0.37 and 0.22).'),
 bl('<b>Close calls</b>: for batters, places 8-10 (Russell, Jitesh Sharma, Maxwell) are separated from Ashutosh Sharma and Romario Shepherd by small margins. For bowlers, places 8-10 (Arshdeep Singh, Hasaranga, Roston Chase) are separated from Haris Rauf by a small margin. Arshdeep carries the largest bowling sample (645 pressure balls), so his place is the most evidence-backed in that group.')]
# 5 reliability
s+= [P('5. Reliability and limitations',H1),
 P('Consistency checks',H2),
 bl('<b>Predictive test</b> (30 random match splits): a player\'s shrunk pressure value from one half predicts pressure performance in the other half with r = 0.15 for batters and 0.15-0.17 for bowlers. A player\'s usual level alone predicts it far less well (r = 0.08 batters, 0.12 bowlers), so pressure-specific performance carries genuine information beyond general quality.'),
 bl('<b>Split-half stability</b>: with only half the matches and a lower sample floor (40 balls), rank correlation of ratings between halves is low (batters rho = 0.15, bowlers rho = 0.02). This is the honest expected noise level for pressure-only samples in T20 cricket; it is why we shrink hard, require repeat innings, and report uncertainty.'),
 bl('<b>Bootstrap</b> (120 resamples of matches): median ranks sit inside or near the top 10 for all selected players (Figure 3). Eligibility itself varies in resamples for players near the 10-innings floor (Jaiswal, Starc, Abrar Ahmed), which lowers their P(top10).'),
 bl('<b>Sensitivity</b>: re-ranking with different weights (press-only; equal thirds; 50/25/25) or a stricter 150-ball floor keeps 8 to 10 of 10 batters and 8 to 10 of 10 bowlers in the top 10.'),
 img('/home/claude/work/f5.png',W*0.95),P('Figure 3: Rank stability under bootstrap over matches. Players to the left of the dashed line stay inside the top 10.',S),
 P('Limitations',H2),
 bl('Pressure samples are small even after pooling 2022 onwards data. Individual positions inside the top 10 are less certain than membership of the group.'),
 bl('The first-innings win-probability signal is weak (AUC 0.68), so first-innings pressure uses explicit situational terms rather than a model. Weights in the PI are judgement-based, though the resulting classes show the expected outcome differences.'),
 bl('The data has no fielding quality, pitch, dew or detailed bowler-batter matchups beyond the opponent adjustment; role changes (batting order, impact-sub rules) are not modelled.'),
 bl('Bowler value does not include catches dropped or fielding support, and batter value cannot separate luck from skill at the ball level.'),
 P('Reproducibility',H2),
 P('Pipeline: (1) build match state and clean data; (2) fit E[remaining runs] and chase win probability out-of-fold; (3) compute PI and classes; (4) value per delivery, opponent adjustment, shrinkage and Pressure Rating; (5) reliability checks; (6) tables and charts. Python scripts for every step are included with the submission (step1 to step9, plus the Streamlit explorer).')]
doc.build(s,onFirstPage=foot,onLaterPages=foot)
