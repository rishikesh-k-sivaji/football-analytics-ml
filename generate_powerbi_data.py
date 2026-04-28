"""
GENERATE POWER BI DATA
Creates two clean focused CSVs for Power BI report:
  1. powerbi_players.csv  — main fact table (one row per player per season)
  2. powerbi_forwards.csv — forward roles table (one row per player)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df    = pd.read_csv('data/cleaned.csv')
roles = pd.read_csv('data/forward_roles.csv')

# ══════════════════════════════════════════════════════
# FILE 1 — powerbi_players.csv
# One row per player per season
# Only the columns Power BI actually needs
# ══════════════════════════════════════════════════════
keep_cols = [
    # Identity
    'player', 'league', 'season_label', 'team',
    'primary_pos', 'age_',

    # Playing time
    'Playing Time_MP', 'Playing Time_Min', '90s_',

    # Core attacking
    'Performance_Gls', 'Performance_Ast',
    'Performance_G+A', 'Expected_xG', 'Expected_xAG',
    'xG_overperformance',

    # Per 90 metrics
    'goal_contrib_p90', 'progressive_impact_p90',
    'defensive_actions_p90',

    # Shooting
    'Standard_Sh', 'Standard_SoT', 'Standard_SoT%',
    'Standard_Dist', 'shot_conversion',

    # Chance creation
    'SCA_SCA', 'GCA_GCA', 'KP_', 'PPA_', 'CrsPA_',

    # Progression
    'Progression_PrgC', 'Progression_PrgP',

    # Dribbling
    'Take-Ons_Att', 'Take-Ons_Succ', 'Take-Ons_Succ%',

    # Defensive
    'Tackles_TklW', 'Performance_Int',
    'Blocks_Blocks', 'Clr_',

    # Aerial
    'Aerial Duels_Won', 'Aerial Duels_Lost', 'aerial_dominance',

    # Discipline
    'Performance_CrdY', 'Performance_CrdR',
]

keep_cols = [c for c in keep_cols if c in df.columns]
players_bi = df[keep_cols].copy()

# Clean column names — replace spaces and special chars for Power BI
players_bi.columns = (
    players_bi.columns
    .str.replace(' ', '_')
    .str.replace('+', 'plus')
    .str.replace('-', 'minus')
    .str.replace('%', '_pct')
    .str.replace('/', '_')
    .str.replace('(', '')
    .str.replace(')', '')
)

players_bi.to_csv('data/powerbi_players.csv', index=False)
print(f"Saved powerbi_players.csv — {players_bi.shape[0]:,} rows x {players_bi.shape[1]} cols")
print(f"Columns: {list(players_bi.columns)}")

# ══════════════════════════════════════════════════════
# FILE 2 — powerbi_forwards.csv
# One row per forward player — career averages + role
# ══════════════════════════════════════════════════════

# Get career averages from main data for forwards
fwd_main = df[df['primary_pos'] == 'FW'].groupby('player').agg(
    league=('league',               'last'),
    team=('team',                   'last'),
    avg_age=('age_',                'mean'),
    seasons=('season_label',        'nunique'),
    career_goals=('Performance_Gls','sum'),
    career_assists=('Performance_Ast','sum'),
    career_xG=('Expected_xG',       'sum'),
    avg_xG_over=('xG_overperformance','mean'),
    avg_goal_p90=('goal_contrib_p90','mean'),
    avg_prog_p90=('progressive_impact_p90','mean'),
    avg_def_p90=('defensive_actions_p90','mean'),
    avg_shot_conv=('shot_conversion','mean'),
    avg_aerial=('aerial_dominance', 'mean'),
    avg_SCA=('SCA_SCA',             'mean'),
    avg_dribbles=('Take-Ons_Succ',  'mean'),
    avg_shots=('Standard_Sh',       'mean'),
    avg_shot_dist=('Standard_Dist', 'mean'),
).reset_index()

# Merge role labels
fwd_main = fwd_main.merge(
    roles[['player', 'role', 'cluster']],
    on='player', how='left'
)

# Round for cleanliness
num_cols = fwd_main.select_dtypes(include=np.number).columns
fwd_main[num_cols] = fwd_main[num_cols].round(2)

fwd_main.to_csv('data/powerbi_forwards.csv', index=False)
print(f"\nSaved powerbi_forwards.csv — {fwd_main.shape[0]:,} rows x {fwd_main.shape[1]} cols")
print(f"Columns: {list(fwd_main.columns)}")

# ══════════════════════════════════════════════════════
# QUICK SUMMARY
# ══════════════════════════════════════════════════════
print("\n--- POWER BI FILES READY ---")
print("Load both files into Power BI")
print("Create relationship: powerbi_players[player] → powerbi_forwards[player]")
print("\nSuggested Pages:")
print("  Page 1 — League Overview  (use powerbi_players.csv)")
print("  Page 2 — Forward Roles   (use powerbi_forwards.csv)")
print("  Page 3 — Player Profile  (use powerbi_players.csv with slicers)")
