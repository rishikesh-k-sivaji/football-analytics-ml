"""
DATA PREPARATION
- Load raw data
- Deduplicate (one row per player per season per team)
- Clean decimals, positions, leagues
- Engineer features
- Save cleaned file
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_clean(path="data/Top5_League_Players_2017to2024_dataset.csv"):

    # ── Load ──────────────────────────────────────────
    df = pd.read_csv(path, sep=';', on_bad_lines='skip', decimal=',')
    print(f"Raw shape: {df.shape}")

    # ── Deduplicate ───────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=['player', 'season', 'team'])
    print(f"Removed {before - len(df)} duplicate rows → {len(df)} rows remaining")

    # ── Clean league names ────────────────────────────
    league_map = {
        'ENG-Premier League': 'Premier League',
        'ESP-La Liga':        'La Liga',
        'FRA-Ligue 1':        'Ligue 1',
        'GER-Bundesliga':     'Bundesliga',
        'ITA-Serie A':        'Serie A',
    }
    df['league'] = df['league'].map(league_map).fillna(df['league'])

    # ── Season labels ─────────────────────────────────
    season_map = {
        1718:'2017-18', 1819:'2018-19', 1920:'2019-20',
        2021:'2020-21', 2122:'2021-22', 2223:'2022-23',
        2324:'2023-24', 2425:'2024-25',
    }
    df['season_label'] = df['season'].map(season_map)

    # ── Primary position ──────────────────────────────
    def primary_pos(p):
        if pd.isna(p): return 'Unknown'
        pos = str(p).split(',')[0].strip()
        return pos if pos in ['GK','DF','MF','FW'] else 'Unknown'
    df['primary_pos'] = df['pos_'].apply(primary_pos)

    # ── Filter min 5 matches ──────────────────────────
    df = df[df['Playing Time_MP'] >= 5].reset_index(drop=True)

    # ── Fill nulls ────────────────────────────────────
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(0)

    # ── Feature engineering ───────────────────────────
    mins90 = df['90s_'].replace(0, np.nan)
    df['goal_contrib_p90']      = (df['Performance_Gls'] + df['Performance_Ast']) / mins90
    df['xG_overperformance']    = df['Performance_Gls'] - df['Expected_xG']
    df['progressive_impact_p90']= (df['Progression_PrgC'] + df['Progression_PrgP']) / mins90
    df['defensive_actions_p90'] = (df['Tackles_TklW'] + df['Performance_Int'] + df['Blocks_Blocks']) / mins90
    total_aerials = df['Aerial Duels_Won'] + df['Aerial Duels_Lost']
    df['aerial_dominance']      = df['Aerial Duels_Won'] / total_aerials.replace(0, np.nan)
    df['shot_conversion']       = df['Performance_Gls'] / df['Standard_Sh'].replace(0, np.nan)
    df[['goal_contrib_p90','progressive_impact_p90','defensive_actions_p90',
        'aerial_dominance','shot_conversion']] = \
        df[['goal_contrib_p90','progressive_impact_p90','defensive_actions_p90',
            'aerial_dominance','shot_conversion']].fillna(0)

    print(f"Final shape: {df.shape}")
    return df

if __name__ == '__main__':
    df = load_and_clean()
    df.to_csv('data/cleaned.csv', index=False)
    print("Saved → data/cleaned.csv")
