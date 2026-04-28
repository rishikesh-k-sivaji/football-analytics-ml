"""
FOOTBALL ANALYTICS — STREAMLIT APP
4 Pages: Home | Forward Role Classifier | Player Similarity | Player Search & xG
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib, warnings, os
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Football Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; }
h1, h2, h3, p, label              { color: #e6edf3 !important; }
.stMetric label                    { color: #8b949e !important; }
.block-container                   { padding-top: 1.5rem; }
div[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor":   "#30363d", "axes.labelcolor": "#e6edf3",
    "xtick.color":      "#8b949e", "ytick.color":     "#8b949e",
    "text.color":       "#e6edf3", "grid.color":      "#21262d",
    "font.family":      "DejaVu Sans",
})

LEAGUE_COLORS = {
    "Premier League": "#a855f7", "La Liga":    "#f97316",
    "Bundesliga":     "#ef4444", "Serie A":    "#3b82f6",
    "Ligue 1":        "#22c55e",
}
ROLE_COLORS = {
    "Poacher":          "#ef4444",
    "Winger":           "#f97316",
    "Target Man":       "#3b82f6",
    "Pressing Forward": "#22c55e",
}

# ── Load data and models ──────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned.csv")

@st.cache_resource
def load_models():
    roles_bundle = joblib.load("models/forward_roles.pkl")
    sim_bundle   = joblib.load("models/player_similarity.pkl")
    return roles_bundle, sim_bundle

df                       = load_data()
roles_bundle, sim_bundle = load_models()
player_agg               = sim_bundle['player_agg']
cluster_sim_matrices     = sim_bundle['cluster_sim_matrices']
sim_features             = sim_bundle['features']
role_features            = roles_bundle['features']
season_order             = ["2017-18","2018-19","2019-20","2020-21",
                            "2021-22","2022-23","2023-24","2024-25"]

# ── Sidebar ───────────────────────────────────────────
st.sidebar.title("Football Analytics")
st.sidebar.caption("Top 5 European Leagues  |  2017 - 2025")
st.sidebar.divider()

page = st.sidebar.radio("Navigation", [
    "Home",
    "Forward Role Classifier",
    "Player Similarity",
    "Player Search",
])

st.sidebar.divider()
st.sidebar.caption(f"{len(df):,} player-seasons  |  5 leagues  |  8 seasons")

# ════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════
if page == "Home":
    st.title("Football Analytics Dashboard")
    st.caption("Top 5 European Leagues — Premier League, La Liga, Bundesliga, Serie A, Ligue 1")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Player-Seasons",  f"{len(df):,}")
    c2.metric("Unique Players",  f"{df['player'].nunique():,}")
    c3.metric("Leagues",         "5")
    c4.metric("Seasons",         "8  (2017 to 2025)")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("About this project")
        st.markdown("""
**Forward Role Classifier**

Identifies the tactical role of any forward — Poacher, Winger,
Target Man, or Pressing Forward — using KMeans clustering to
discover natural role groups, followed by a Random Forest
classifier trained on those labels.

**Player Similarity**

Finds the most statistically similar players to any player
in the dataset. Uses KMeans to group all 5,766 players into
style clusters, then ranks within each cluster using cosine
similarity. Includes an age filter to find younger alternatives.

**Player Search**

Career stats and season-by-season breakdown for any player.
Includes an xG vs actual goals chart to show whether a player
consistently outperforms or underperforms their expected output.
        """)

    with col2:
        st.subheader("All-Time Leaders")
        leaders = df.groupby('player').agg(
            Goals=('Performance_Gls',   'sum'),
            Assists=('Performance_Ast', 'sum'),
            xG=('Expected_xG',          'sum'),
        )
        top_g  = leaders['Goals'].idxmax()
        top_a  = leaders['Assists'].idxmax()
        top_xg = leaders['xG'].idxmax()

        st.metric("Top Goal Scorer",
                  top_g, f"{int(leaders.loc[top_g,'Goals'])} goals")
        st.metric("Most Assists",
                  top_a, f"{int(leaders.loc[top_a,'Assists'])} assists")
        st.metric("Highest Career xG",
                  top_xg, f"{leaders.loc[top_xg,'xG']:.1f} xG")

    st.markdown("---")
    st.subheader("Goals per Season by League")

    lg_season = (
        df.groupby(['league', 'season_label'])['Performance_Gls']
          .sum().reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    for league, grp in lg_season.groupby('league'):
        grp2 = grp.set_index('season_label').reindex(season_order).reset_index()
        ax.plot(grp2['season_label'], grp2['Performance_Gls'],
                marker='o', lw=2.5,
                color=LEAGUE_COLORS.get(league, '#888'), label=league)
        ax.fill_between(grp2['season_label'], grp2['Performance_Gls'],
                        alpha=0.06, color=LEAGUE_COLORS.get(league, '#888'))
    ax.set_xlabel('Season')
    ax.set_ylabel('Total Goals')
    ax.legend(framealpha=0.3, fontsize=9)
    ax.grid(True, axis='y', alpha=0.4)
    plt.xticks(rotation=25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════
# PAGE 2 — FORWARD ROLE CLASSIFIER
# ════════════════════════════════════════════════════════
elif page == "Forward Role Classifier":
    st.title("Forward Role Classifier")
    st.caption("KMeans clustering discovers natural role groups — Random Forest classifies each forward")
    st.markdown("---")

    role_df = pd.read_csv("data/forward_roles.csv")

    # Role descriptions
    st.subheader("Role Profiles")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**Poacher**\nHigh shot conversion, penalty area presence, low dribbles")
    c2.markdown("**Winger**\nHigh dribbles, progressive carries, wide touches, crosses")
    c3.markdown("**Target Man**\nHigh aerial win rate, strong xG, hold-up play")
    c4.markdown("**Pressing Forward**\nHigh defensive work rate, tackles, pressing aggression")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Browse Players", "Model Evaluation", "Predict Role"])

    # ── Tab 1: Browse ─────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)
        sel_role   = col1.selectbox("Role", ["Poacher","Winger","Target Man","Pressing Forward"])
        sel_league = col2.multiselect("League", df['league'].unique().tolist(),
                                      default=df['league'].unique().tolist())

        player_meta = df[df['primary_pos']=='FW'].groupby('player').agg(
            league=('league','last'), team=('team','last'),
            avg_age=('age_','mean'), career_goals=('Performance_Gls','sum'),
            career_assists=('Performance_Ast','sum'), avg_xG=('Expected_xG','mean'),
        ).reset_index()
        player_meta = player_meta.merge(role_df[['player','role']], on='player', how='inner')

        filtered = player_meta[
            (player_meta['role']   == sel_role) &
            (player_meta['league'].isin(sel_league))
        ].sort_values('career_goals', ascending=False)

        filtered['avg_age'] = filtered['avg_age'].round(1)
        filtered['avg_xG']  = filtered['avg_xG'].round(2)

        st.markdown(f"**{sel_role} — {len(filtered)} players found**")
        st.dataframe(
            filtered[['player','league','team','avg_age',
                       'career_goals','career_assists','avg_xG']]
            .rename(columns={'avg_age':'Age','career_goals':'Goals',
                             'career_assists':'Assists','avg_xG':'Avg xG'})
            .reset_index(drop=True),
            use_container_width=True
        )

        # Role distribution bar chart
        st.markdown("---")
        st.subheader("Role Distribution")
        role_counts = role_df['role'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors  = [ROLE_COLORS[r] for r in role_counts.index]
        bars    = ax.barh(role_counts.index, role_counts.values,
                          color=colors, edgecolor='none', height=0.5)
        for bar, val in zip(bars, role_counts.values):
            ax.text(val + 2, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', fontsize=10)
        ax.set_xlabel('Number of Players')
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_title('Forward Role Distribution — All Players')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 2: Model Evaluation ────────────────────────
    with tab2:
        cv_acc   = roles_bundle.get('cv_accuracy',   75.0)
        test_acc = roles_bundle.get('test_accuracy', 76.6)

        c1, c2, c3 = st.columns(3)
        c1.metric("Algorithm",          "Random Forest")
        c2.metric("Cross-Val Accuracy", f"{cv_acc}%")
        c3.metric("Test Accuracy",      f"{test_acc}%")

        st.markdown("""
**How the model was built:**

1. KMeans clustering grouped forwards into 4 natural style clusters based on 13 features
2. Cluster centres were interpreted using domain knowledge to assign role labels
3. A Random Forest classifier was trained on those labels using per-season rows
4. The classifier learns patterns like — high shot conversion + high penalty touches = Poacher

**Why 77% accuracy is reasonable:**

The labels themselves came from clustering, not manually verified ground truth.
Some players genuinely sit between roles — an attacking midfielder playing as a forward
will have mixed characteristics. The Winger class achieves 85% precision because
dribbling and crossing stats are very distinctive.
        """)

        col1, col2 = st.columns(2)
        with col1:
            cm_path = "plots/role_confusion_matrix.png"
            if os.path.exists(cm_path):
                st.subheader("Confusion Matrix")
                st.image(cm_path, use_container_width=True)
        with col2:
            fi_path = "plots/role_feature_importance.png"
            if os.path.exists(fi_path):
                st.subheader("Feature Importance")
                st.image(fi_path, use_container_width=True)

    # ── Tab 3: Live Prediction ─────────────────────────
    with tab3:
        st.subheader("Predict the role of a new forward")
        st.caption("Adjust stats below and click Predict")

        display_names = {
            'shot_conversion':        'Shot Conversion (0 to 1)',
            'Touches_Att Pen':        'Touches in Penalty Area',
            'Take-Ons_Succ':          'Successful Dribbles',
            'Progression_PrgC':       'Progressive Carries',
            'Pass Types_Crs':         'Crosses',
            'aerial_dominance':       'Aerial Duel Win Rate (0 to 1)',
            'Aerial Duels_Won':       'Aerial Duels Won',
            'Expected_xG':            'Expected Goals (xG)',
            'defensive_actions_p90':  'Defensive Actions per 90',
            'Tackles_TklW':           'Tackles Won',
            'Performance_Fls':        'Fouls Committed',
            'goal_contrib_p90':       'Goal Contribution per 90',
            'SCA_SCA':                'Shot Creating Actions',
        }

        inp_cols = st.columns(3)
        inputs   = {}
        for i, feat in enumerate(role_features):
            label = display_names.get(feat, feat)
            mn    = float(df[feat].min())  if feat in df.columns else 0.0
            mx    = float(df[feat].max())  if feat in df.columns else 10.0
            med   = float(df[feat].median()) if feat in df.columns else 0.0
            inputs[feat] = inp_cols[i % 3].number_input(
                label, min_value=round(mn, 2),
                max_value=round(mx, 2), value=round(med, 2),
                key=f"pred_{feat}"
            )

        if st.button("Predict Role"):
            X_in  = pd.DataFrame([inputs])[role_features]
            pred  = roles_bundle['classifier'].predict(X_in)[0]
            probs = roles_bundle['classifier'].predict_proba(X_in)[0]
            classes = roles_bundle['classifier'].classes_

            st.success(f"Predicted Role:  {pred}")
            prob_df = pd.DataFrame({
                'Role': classes,
                'Probability': (probs * 100).round(1)
            }).sort_values('Probability', ascending=False)
            prob_df['Probability'] = prob_df['Probability'].astype(str) + '%'
            st.dataframe(prob_df.reset_index(drop=True),
                         use_container_width=False, hide_index=True)

# ════════════════════════════════════════════════════════
# PAGE 3 — PLAYER SIMILARITY
# ════════════════════════════════════════════════════════
elif page == "Player Similarity":
    st.title("Player Similarity")
    st.caption("KMeans groups all players into style clusters — cosine similarity ranks matches within each cluster")
    st.markdown("---")

    all_players = sorted(player_agg['player'].tolist())
    default_idx = all_players.index("Kevin De Bruyne") if "Kevin De Bruyne" in all_players else 0

    col1, col2, col3 = st.columns(3)
    selected_player = col1.selectbox("Player", all_players, index=default_idx)
    same_pos        = col2.checkbox("Same position only", value=True)
    max_age         = col3.slider("Max age (0 = no limit)", 0, 35, 0)

    if st.button("Find Similar Players"):
        if selected_player not in player_agg['player'].values:
            st.error("Player not found.")
        else:
            player_row = player_agg[player_agg['player'] == selected_player].iloc[0]
            cluster_id = int(player_row['style_cluster'])

            if cluster_id not in cluster_sim_matrices:
                st.error("No similarity data available for this player.")
            else:
                sim_mat = cluster_sim_matrices[cluster_id]

                if selected_player not in sim_mat.index:
                    st.error("Player not in similarity matrix.")
                else:
                    scores = sim_mat[selected_player].drop(selected_player)\
                                                     .sort_values(ascending=False)
                    result = player_agg[player_agg['player'].isin(scores.index)].copy()
                    result['similarity'] = result['player'].map(scores)

                    if same_pos:
                        result = result[result['primary_pos'] == player_row['primary_pos']]
                    if max_age > 0:
                        result = result[result['avg_age'] <= max_age]

                    result = result.sort_values('similarity', ascending=False).head(5)

                    # Reference player
                    st.markdown("---")
                    st.subheader(f"Reference Player: {selected_player}")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Position",     player_row['primary_pos'])
                    r2.metric("League",       player_row['league'])
                    r3.metric("Career Goals", int(player_row['career_goals']))
                    r4.metric("Style Cluster", f"Cluster {cluster_id}")

                    st.markdown("---")
                    st.subheader("Top 5 Similar Players")

                    display = result[['player','league','team','primary_pos',
                                      'avg_age','career_goals','career_assists',
                                      'similarity']].copy()
                    display['similarity'] = (display['similarity'] * 100).round(1)\
                                                                          .astype(str) + '%'
                    display['avg_age']    = display['avg_age'].round(1)
                    display.columns       = ['Player','League','Team','Position',
                                             'Age','Goals','Assists','Similarity']
                    st.dataframe(display.reset_index(drop=True), use_container_width=True)

                    # Comparison chart
                    st.markdown("---")
                    st.subheader("Stat Comparison")

                    compare_stats  = ['goal_contrib_p90','progressive_impact_p90',
                                      'defensive_actions_p90','aerial_dominance',
                                      'shot_conversion']
                    compare_stats  = [c for c in compare_stats if c in player_agg.columns]
                    stat_labels    = ['G+A per 90','Progression per 90',
                                      'Defensive per 90','Aerial Win Rate',
                                      'Shot Conversion'][:len(compare_stats)]

                    all_compare = pd.concat([
                        player_agg[player_agg['player'] == selected_player],
                        result
                    ]).drop_duplicates('player')

                    x     = np.arange(len(compare_stats))
                    width = 0.8 / len(all_compare)
                    fig, ax = plt.subplots(figsize=(12, 5))
                    colors  = ['#f0c030','#3b82f6','#ef4444','#22c55e','#a855f7','#f97316']

                    for i, (_, row) in enumerate(all_compare.iterrows()):
                        vals   = [row[c] if not pd.isna(row[c]) else 0 for c in compare_stats]
                        offset = (i - len(all_compare)/2 + 0.5) * width
                        ax.bar(x + offset, vals, width * 0.9,
                               label=row['player'], color=colors[i % len(colors)], alpha=0.85)

                    ax.set_xticks(x)
                    ax.set_xticklabels(stat_labels)
                    ax.legend(fontsize=8, framealpha=0.3, loc='upper right')
                    ax.grid(True, axis='y', alpha=0.3)
                    ax.set_title(f"Stat Comparison — {selected_player} vs Similar Players")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

# ════════════════════════════════════════════════════════
# PAGE 4 — PLAYER SEARCH
# ════════════════════════════════════════════════════════
elif page == "Player Search":
    st.title("Player Search")
    st.caption("Career stats and xG analysis for any player in the dataset")
    st.markdown("---")

    all_players = sorted(df['player'].unique().tolist())
    default_idx = all_players.index("Lionel Messi") if "Lionel Messi" in all_players else 0
    selected    = st.selectbox("Search Player", all_players, index=default_idx)

    # One row per season — deduplicated at page level
    pdf = (
        df[df['player'] == selected]
          .drop_duplicates(subset=['season_label'])
          .sort_values('season_label')
    )

    if len(pdf) == 0:
        st.warning("No data found for this player.")
    else:
        lat = pdf.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Position",  lat['primary_pos'])
        c2.metric("Last Club", lat['team'])
        c3.metric("League",    lat['league'])
        c4.metric("Seasons",   pdf['season_label'].nunique())

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Career Totals")
            total_over = (pdf['Performance_Gls'] - pdf['Expected_xG']).sum()
            career = pd.DataFrame({
                "Metric": ["Goals","Assists","G + A","xG","xAG","Goals minus xG"],
                "Value":  [
                    int(pdf['Performance_Gls'].sum()),
                    int(pdf['Performance_Ast'].sum()),
                    int(pdf['Performance_Gls'].sum() + pdf['Performance_Ast'].sum()),
                    round(pdf['Expected_xG'].sum(), 1),
                    round(pdf['Expected_xAG'].sum(), 1),
                    round(total_over, 2),
                ]
            })
            st.dataframe(career, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Season by Season")
            sv = pdf[['season_label','team','league',
                       'Performance_Gls','Performance_Ast',
                       'Expected_xG','Playing Time_MP']].copy()
            sv.columns = ['Season','Team','League','Goals','Assists','xG','Matches']
            sv['xG']   = sv['xG'].round(1)
            st.dataframe(sv.reset_index(drop=True), use_container_width=True)

        st.markdown("---")
        st.subheader("Goals and Assists per Season")
        fig, ax = plt.subplots(figsize=(11, 4))
        x = np.arange(len(pdf))
        ax.bar(x - 0.2, pdf['Performance_Gls'].values, 0.38,
               color='#2ecc71', label='Goals',   alpha=0.9)
        ax.bar(x + 0.2, pdf['Performance_Ast'].values, 0.38,
               color='#3498db', label='Assists',  alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(pdf['season_label'].values, rotation=25)
        ax.legend(framealpha=0.3)
        ax.grid(True, axis='y', alpha=0.4)
        ax.set_title(f"{selected} — Goals and Assists per Season")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.subheader("xG vs Actual Goals")
        st.caption("Green area = outperforming xG     Red area = underperforming xG")

        fig, ax    = plt.subplots(figsize=(11, 4))
        seasons    = pdf['season_label'].values
        actual     = pdf['Performance_Gls'].values.astype(float)
        expected   = pdf['Expected_xG'].values.astype(float)

        ax.plot(seasons, actual,   marker='o', color='#2ecc71', lw=2.5, label='Actual Goals')
        ax.plot(seasons, expected, marker='s', color='#e74c3c', lw=2,   label='xG', ls='--')
        ax.fill_between(seasons, actual, expected,
                        where=(actual >= expected),
                        alpha=0.15, color='#2ecc71', label='Above xG')
        ax.fill_between(seasons, actual, expected,
                        where=(actual < expected),
                        alpha=0.15, color='#e74c3c', label='Below xG')
        ax.legend(framealpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{selected} — Actual Goals vs xG per Season")
        plt.xticks(rotation=25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Summary line
        st.markdown("---")
        total_over = round((pdf['Performance_Gls'] - pdf['Expected_xG']).sum(), 1)
        if total_over > 0:
            st.success(f"{selected} scored {total_over} goals more than xG predicted — consistent overperformer.")
        elif total_over < 0:
            st.warning(f"{selected} scored {abs(total_over)} goals fewer than xG predicted — underperformed expected output.")
        else:
            st.info(f"{selected} scored almost exactly what xG predicted.")
