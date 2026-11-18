import pandas as pd, numpy as np, streamlit as st, plotly.express as px, plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="SupeRR Selector S2 | Pressure Ratings", page_icon="🏏", layout="wide")
HERE = Path(__file__).parent; DATA = HERE / "data"; IMG = HERE / "images"
PINK, NAVY, SKY, GREY = "#E91E8C", "#1B2A5B", "#7FB3FF", "#8A94A6"

st.markdown(f"""
<style>
.block-container {{padding-top: 1.4rem; max-width: 1250px;}}
.hero {{background: linear-gradient(120deg,{NAVY} 0%,#2B1B5B 60%,{PINK} 160%); padding: 28px 32px; border-radius: 18px; margin-bottom: 18px;}}
.hero h1 {{margin: 0; font-size: 2.1rem;}} .hero p {{margin: 6px 0 0 0; opacity: .85; font-size: 1.02rem;}}
.card {{background: {NAVY}; border-radius: 14px; padding: 12px 14px; border-left: 5px solid {PINK}; margin-bottom: 10px;}}
.card .rk {{font-size: .75rem; color: {PINK}; font-weight: 700; letter-spacing: .08em;}}
.card .nm {{font-size: 1.02rem; font-weight: 700;}} .card .st {{font-size: .78rem; opacity: .8;}}
.av {{display:inline-block; width:34px; height:34px; line-height:34px; text-align:center; border-radius:50%; background:{PINK}; color:white; font-weight:700; margin-right:10px; vertical-align:middle;}}
.kpi {{background:{NAVY}; border-radius:14px; padding:14px 18px; text-align:center;}} .kpi b {{font-size:1.7rem; color:{PINK};}} .kpi span {{display:block; font-size:.8rem; opacity:.8;}}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load():
    bat = pd.read_csv(DATA/"batters.csv", index_col=0); bowl = pd.read_csv(DATA/"bowlers.csv", index_col=0)
    return (bat, bowl, pd.read_csv(DATA/"pi_by_over.csv"), pd.read_csv(DATA/"validation.csv"),
            pd.read_csv(DATA/"chase_heatmap.csv", index_col=0), pd.read_csv(DATA/"sensitivity.csv"))
bat, bowl, pio, val, heat, sens = load()
layout = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#F2F4FA"), margin=dict(l=10, r=10, t=40, b=10))

st.markdown('<div class="hero"><h1>🏏 Who delivers when it matters?</h1><p>Pressure Ratings for IPL + T20I batters and bowlers (2022 onwards) · SupeRR Selector Season 2 · Round 1</p></div>', unsafe_allow_html=True)
k = st.columns(4)
for col, (n, t) in zip(k, [("788", "matches analysed"), ("188,151", "deliveries"), ("40,506", "high-pressure deliveries"), ("85 / 116", "eligible batters / bowlers")]):
    col.markdown(f'<div class="kpi"><b>{n}</b><span>{t}</span></div>', unsafe_allow_html=True)
st.write("")
tabs = st.tabs(["🏆 Top 10", "🎯 Pressure Index", "🏏 Batters", "🎳 Bowlers", "🔬 Reliability", "📘 Method"])

def initials(n): return "".join(w[0] for w in n.split()[:2]).upper()
def cards(df, kind):
    for _, r in df.head(10).iterrows():
        if kind == "bat": line = f"{int(r.hp_balls)} HP balls · SR {r.hp_sr:.0f} (usual {r.usual_sr:.0f}) · {int(r.hp_outs)} outs"
        else: line = f"{int(r.hp_balls)} HP balls · econ {r.hp_econ:.2f} (usual {r.usual_econ:.2f}) · {int(r.hp_wkts)} wkts"
        st.markdown(f'<div class="card"><div class="rk">RANK {int(r["rank"])} · RATING {r.rating:.2f}</div><span class="av">{initials(r["name"])}</span><span class="nm">{r["name"]}</span><div class="st">{line}</div></div>', unsafe_allow_html=True)

with tabs[0]:
    a, b = st.columns(2)
    with a: st.subheader("Top 10 batters"); cards(bat, "bat")
    with b: st.subheader("Top 10 bowlers"); cards(bowl, "bowl")
    st.caption("HP = high pressure. Rating = 60% pressure performance + 20% resilience vs own usual level + 20% consistency across pressure innings.")

with tabs[1]:
    st.subheader("How pressure is defined")
    st.write("A 0–100 **Pressure Index** per delivery. Chase: closeness of contest (45%), equation difficulty vs expected remaining runs (30%), lateness (25%). First innings: recent collapse (30%), wickets lost vs typical (25%), lateness (25%), projected total below par (20%). Bowling-side index mirrors the resource terms. Top 25% of deliveries in each innings = **high pressure**.")
    c1, c2 = st.columns(2)
    with c1:
        f = px.line(pio.assign(innings=pio.innings.map({1: "1st innings", 2: "Chase"})), x="over", y="PI", color="innings", color_discrete_sequence=[SKY, PINK], title="Mean Pressure Index by over")
        f.update_layout(**layout); st.plotly_chart(f, use_container_width=True)
    with c2:
        h = go.Figure(go.Heatmap(z=heat.values, x=list(heat.columns), y=[f"{i} wkts lost" for i in heat.index], colorscale="Magma", colorbar=dict(title="PI")))
        h.update_layout(title="Chase pressure: balls left × wickets lost", **layout); st.plotly_chart(h, use_container_width=True)
    st.subheader("Does it separate pressure from ordinary play?")
    v = val.set_index("pressure").loc[["Low", "Mid", "High"]].reset_index()
    m = st.selectbox("Outcome", ["wicket_rate", "dot_pct", "avg_wkts_lost", "avg_balls_left"], format_func=lambda x: {"wicket_rate": "Wicket rate (per 100 balls)", "dot_pct": "Dot-ball %", "avg_wkts_lost": "Average wickets already lost", "avg_balls_left": "Average balls left"}[x])
    f = px.bar(v, x="pressure", y=m, color="pressure", color_discrete_map={"Low": GREY, "Mid": SKY, "High": PINK}, text_auto=".2f"); f.update_layout(showlegend=False, **layout); st.plotly_chart(f, use_container_width=True)

def role_tab(df, kind):
    mn = st.slider("Minimum high-pressure balls", 90, int(df.hp_balls.max()), 90, 10, key=kind)
    d = df[df.hp_balls >= mn].copy(); d["rank"] = d.rating.rank(ascending=False).astype(int)
    c1, c2 = st.columns([1.05, 1])
    with c1:
        t = d.sort_values("rank").head(10).iloc[::-1]
        f = go.Figure()
        for _, r in t.iterrows(): f.add_trace(go.Scatter(x=[r.usual_value, r.pressure_value], y=[r["name"]] * 2, mode="lines", line=dict(color="#4A5A8F", width=4), showlegend=False, hoverinfo="skip"))
        f.add_trace(go.Scatter(x=t.usual_value, y=t["name"], mode="markers", marker=dict(color=GREY, size=11), name="Usual"))
        f.add_trace(go.Scatter(x=t.pressure_value, y=t["name"], mode="markers", marker=dict(color=PINK, size=13), name="Under pressure"))
        f.update_layout(title="Top 10: value per ball, usual vs under pressure", height=430, **layout); st.plotly_chart(f, use_container_width=True)
    with c2:
        s = px.scatter(d, x="usual_value", y="pressure_value", size="hp_balls", color="rating", hover_name="name", color_continuous_scale="Magma", title="Everyone: pressure vs usual (above line = rises under pressure)")
        lo, hi = float(min(d.usual_value.min(), d.pressure_value.min())), float(max(d.usual_value.max(), d.pressure_value.max()))
        s.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color=GREY)); s.update_layout(height=430, **layout); st.plotly_chart(s, use_container_width=True)
    cols = ["rank", "name", "hp_balls", "hp_sr" if kind == "bat" else "hp_econ", "usual_sr" if kind == "bat" else "usual_econ", "hp_outs" if kind == "bat" else "hp_wkts", "pressure_value", "usual_value", "delta", "hit_rate", "rating", "p_top10"]
    st.dataframe(d.sort_values("rank")[cols].round(3), hide_index=True, use_container_width=True, height=330)
    st.subheader("Player profile")
    p = st.selectbox("Choose a player", d.sort_values("rank")["name"], key="p" + kind); r = d[d["name"] == p].iloc[0]
    q = st.columns(5); q[0].metric("Rank", int(r["rank"])); q[1].metric("Rating", f"{r.rating:.2f}"); q[2].metric("Pressure value/ball", f"{r.pressure_value:+.3f}"); q[3].metric("Resilience", f"{r.delta:+.3f}"); q[4].metric("Hit rate", f"{r.hit_rate:.0%}")
    rad = go.Figure(go.Scatterpolar(r=[np.clip((d[c] <= r[c]).mean(), 0, 1) for c in ["pressure_value", "delta", "hit_rate", "hp_balls", "usual_value"]], theta=["Pressure value", "Resilience", "Consistency", "Sample size", "Usual level"], fill="toself", line=dict(color=PINK)))
    rad.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)), title="Percentile profile among eligible players", height=380, **layout); st.plotly_chart(rad, use_container_width=True)

with tabs[2]: role_tab(bat, "bat")
with tabs[3]: role_tab(bowl, "bowl")

with tabs[4]:
    st.subheader("How sure are we?")
    st.write("120 bootstrap resamples of matches: dot = median rank, bar = 10th–90th percentile. Left of the dashed line = inside the top 10.")
    cc = st.columns(2)
    for col, df, t in zip(cc, [bat, bowl], ["Batters", "Bowlers"]):
        d = df.head(10).iloc[::-1]; f = go.Figure()
        for _, r in d.iterrows(): f.add_trace(go.Scatter(x=[r.p10, r.p90], y=[r["name"]] * 2, mode="lines", line=dict(color="#4A5A8F", width=6), showlegend=False, hoverinfo="skip"))
        f.add_trace(go.Scatter(x=d.med_rank, y=d["name"], mode="markers", marker=dict(color=PINK, size=11), showlegend=False)); f.add_vline(x=10.5, line_dash="dash", line_color=GREY)
        f.update_layout(title=t, xaxis=dict(range=[0, 45], title="Rank"), height=400, **layout); col.plotly_chart(f, use_container_width=True)
    st.subheader("Sensitivity of the top 10")
    st.dataframe(sens.rename(columns={"overlap_with_final_top10": "names kept (of 10)"}), hide_index=True, use_container_width=True)
    st.info("Honest caveat: pressure samples are small. Split-half rank stability is low (batters ρ≈0.15, bowlers ρ≈0.02), but shrunk pressure values predict held-out pressure performance (r≈0.15) better than usual level alone (r≈0.08 batters). Membership of the top 10 is more reliable than exact order.")

with tabs[5]:
    st.subheader("Method in one page")
    st.markdown("""
1. **Clean & scope** – IPL + T20I from 2022; drop Super Overs, rain-revised, no-result and A-team tournaments.
2. **Match state** – score, wickets, balls left, target, required/current rate for every delivery.
3. **Expected remaining runs** – gradient-boosted model of balls left × wickets lost × level. **Chase win probability** – out-of-fold, grouped by match (AUC 0.89).
4. **Pressure Index** – batting-side and mirrored bowling-side; top 25% per innings = high pressure; decided chases excluded.
5. **Value per ball** – runs above / saved vs expectation (wickets cost their expected-runs value); opponent-quality adjusted.
6. **Shrinkage** – empirical-Bayes toward league mean (k = 244 balls batters, 151 bowlers).
7. **Pressure Rating** – 0.60 × pressure value + 0.20 × resilience (vs own usual) + 0.20 × consistency (share of pressure innings with positive value), z-scored.
""")
    for f, cap in [("f1.png", "Pressure Index by over"), ("f2.png", "High-pressure vs ordinary outcomes")]:
        if (IMG / f).exists(): st.image(str(IMG / f), caption=cap, use_container_width=True)
st.caption("Built for SupeRR Selector Season 2 · data: supplied IPL/T20I ball-by-ball pack · full methodology in the PDF report")
