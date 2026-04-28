"""
PLAYER SIMILARITY ENGINE
Step 1 — Unit normalise vectors so KMeans uses cosine-equivalent distance
Step 2 — KMeans clusters all players into 15 style groups (ML)
Step 3 — Cosine similarity within each cluster to rank matches

Feature selection rationale:
  - Removed redundant: Progression_PrgC, Progression_PrgP (already in progressive_impact_p90)
  - Removed weak: Total_Cmp% (less meaningful than specific creation metrics)
  - Removed: Take-Ons_Succ volume (replaced by rate + attempts separately)
  - Added: Standard_Dist, PPA_, CrsPA_, Take-Ons_Succ%, Take-Ons_Att,
           Performance_Int, Carries_Dis, 1/3_, Performance_Fls, Tackles_TklW

Why 15 clusters:
  4 positions x ~4 styles each = 15 to 16 natural groups

Why unit normalise before KMeans:
  KMeans uses Euclidean distance internally.
  When vectors are unit normalised (length = 1),
  Euclidean distance and cosine distance become equivalent.
  This ensures KMeans groups by playing style direction not volume.
"""
import pandas as pd
import numpy as np
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs('models', exist_ok=True)
df = pd.read_csv('data/cleaned.csv')

# ── Features ──────────────────────────────────────────
sim_features = [
    # Attacking output
    'goal_contrib_p90',        # goals + assists per 90
    'shot_conversion',         # goals per shot — clinical finishing
    'Expected_xG',             # quality of chances for themselves
    'Expected_xAG',            # quality of chances created for teammates
    'Standard_Dist',           # average shot distance — box striker vs long range

    # Chance creation style
    'SCA_SCA',                 # shot creating actions
    'GCA_GCA',                 # goal creating actions
    'KP_',                     # key passes
    'PPA_',                    # passes into penalty area — direct danger
    'CrsPA_',                  # crosses into penalty area — crossing style
    '1/3_',                    # passes into final third — build-up involvement

    # Progression
    'progressive_impact_p90',  # combined progressive carries + passes per 90
    'Touches_Att Pen',         # penalty area presence

    # Dribbling
    'Take-Ons_Att',            # dribble attempts — how often they try
    'Take-Ons_Succ%',          # dribble success rate — quality of attempts

    # Defensive contribution
    'defensive_actions_p90',   # combined tackles + interceptions + blocks per 90
    'aerial_dominance',        # aerial duel win rate
    'Performance_Int',         # interceptions — reading of the game
    'Tackles_TklW',            # tackles won — active defending
    'Performance_Fls',         # fouls committed — pressing aggression

    # Ball retention
    'Carries_Dis',             # times dispossessed — ball retention under pressure
]
sim_features = [f for f in sim_features if f in df.columns]
print(f"Total features: {len(sim_features)}")

# ── Aggregate per player — career average ─────────────
feat_agg = df.groupby('player')[sim_features].mean()
meta_agg = df.groupby('player').agg(
    primary_pos=('primary_pos',      'last'),
    league=('league',                'last'),
    team=('team',                    'last'),
    avg_age=('age_',                 'mean'),
    career_goals=('Performance_Gls', 'sum'),
    career_assists=('Performance_Ast','sum'),
    seasons=('season_label',         'nunique'),
).reset_index()

player_agg = meta_agg.merge(feat_agg.reset_index(), on='player')
X = player_agg[sim_features].fillna(0)

# ── Step 1: Standardise ───────────────────────────────
# Mean 0, std 1 — puts all features on equal scale
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Step 2: Unit normalise ────────────────────────────
# Normalise each player vector to length 1
# This makes Euclidean distance = cosine distance
# So KMeans clusters by style direction not volume
X_unit = normalize(X_scaled, norm='l2')
print("\nVector lengths after unit normalisation (should all be 1.0):")
lengths = np.linalg.norm(X_unit, axis=1)
print(f"  Min: {lengths.min():.4f}  Max: {lengths.max():.4f}  Mean: {lengths.mean():.4f}")

# ── Step 3: KMeans with 15 clusters ───────────────────
# 15 clusters aligns with real football structure:
#   GK: ~2 styles, DF: ~4, MF: ~4, FW: ~4 = 14 to 16
print("\nRunning KMeans (15 clusters) on unit-normalised vectors...")
kmeans = KMeans(n_clusters=15, random_state=42, n_init=20)
player_agg['style_cluster'] = kmeans.fit_predict(X_unit)

# ── Cluster summary ───────────────────────────────────
print("\nCluster sizes and position mix:")
for cluster in range(15):
    cluster_players = player_agg[player_agg['style_cluster'] == cluster]
    pos_mix = cluster_players['primary_pos'].value_counts().head(3).to_dict()
    print(f"  Cluster {cluster:2d} ({len(cluster_players):3d} players): {pos_mix}")

# ── Step 4: Cosine similarity within each cluster ─────
print("\nComputing cosine similarity within clusters...")
cluster_sim_matrices = {}
for cluster in range(15):
    mask            = player_agg['style_cluster'] == cluster
    cluster_players = player_agg[mask]
    if len(cluster_players) < 2:
        continue
    cluster_X = X_unit[mask]
    sim_mat   = cosine_similarity(cluster_X)
    cluster_sim_matrices[cluster] = pd.DataFrame(
        sim_mat,
        index=cluster_players['player'].values,
        columns=cluster_players['player'].values
    )
print(f"  Built {len(cluster_sim_matrices)} cluster similarity matrices")

# ── Save ──────────────────────────────────────────────
joblib.dump({
    'kmeans':               kmeans,
    'scaler':               scaler,
    'player_agg':           player_agg,
    'cluster_sim_matrices': cluster_sim_matrices,
    'features':             sim_features,
    'n_clusters':           15,
}, 'models/player_similarity.pkl')
print("Saved -> models/player_similarity.pkl")

# ── Helper function ───────────────────────────────────
def find_similar(name, top_n=5, same_pos=True, max_age=None):
    if name not in player_agg['player'].values:
        return f"Player not found: {name}"
    player_row = player_agg[player_agg['player'] == name].iloc[0]
    cluster_id = int(player_row['style_cluster'])
    if cluster_id not in cluster_sim_matrices:
        return "No similarity data for this cluster"
    sim_mat = cluster_sim_matrices[cluster_id]
    if name not in sim_mat.index:
        return "Player not in similarity matrix"
    scores = sim_mat[name].drop(name).sort_values(ascending=False)
    result = player_agg[player_agg['player'].isin(scores.index)].copy()
    result['similarity'] = result['player'].map(scores)
    if same_pos:
        result = result[result['primary_pos'] == player_row['primary_pos']]
    if max_age and max_age > 0:
        result = result[result['avg_age'] <= max_age]
    return result.sort_values('similarity', ascending=False).head(top_n)[
        ['player','league','team','primary_pos','avg_age','career_goals','similarity']
    ]

# ── Test ──────────────────────────────────────────────
print("\nSimilar to Kevin De Bruyne:")
print(find_similar('Kevin De Bruyne').to_string(index=False))

print("\nSimilar to Lionel Messi (under 28):")
print(find_similar('Lionel Messi', max_age=28).to_string(index=False))

print("\nSimilar to Virgil van Dijk:")
print(find_similar('Virgil van Dijk').to_string(index=False))

print("\nSimilar to Alisson:")
print(find_similar('Alisson').to_string(index=False))

print("\nSimilar to Erling Haaland:")
print(find_similar('Erling Haaland').to_string(index=False))
