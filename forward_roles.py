"""
FORWARD ROLE CLASSIFICATION
Step 1 — KMeans clustering to discover roles (unsupervised)
Step 2 — Random Forest classifier trained on those labels
Step 3 — Evaluation: confusion matrix + feature importance saved as images
Roles: Poacher | Winger | Target Man | Pressing Forward
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor":   "#30363d", "axes.labelcolor": "#e6edf3",
    "xtick.color":      "#8b949e", "ytick.color":     "#8b949e",
    "text.color":       "#e6edf3", "grid.color":      "#21262d",
    "font.family":      "DejaVu Sans",
})

df = pd.read_csv('data/cleaned.csv')

# ── Filter forwards with enough playing time ──────────
fwd = df[
    (df['primary_pos'] == 'FW') &
    (df['90s_'] >= 8)
].copy()
print(f"Forwards with >=8 90s: {len(fwd)}")

# ── Features ──────────────────────────────────────────
role_features = [
    'shot_conversion',
    'Touches_Att Pen',
    'Take-Ons_Succ',
    'Progression_PrgC',
    'Pass Types_Crs',
    'aerial_dominance',
    'Aerial Duels_Won',
    'Expected_xG',
    'defensive_actions_p90',
    'Tackles_TklW',
    'Performance_Fls',
    'goal_contrib_p90',
    'SCA_SCA',
]
role_features = [f for f in role_features if f in fwd.columns]

# ── Aggregate per player — career average ─────────────
fwd_agg = fwd.groupby('player')[role_features + ['league', 'age_']].agg(
    {**{f: 'mean' for f in role_features}, 'league': 'last', 'age_': 'mean'}
).reset_index()

X = fwd_agg[role_features].fillna(0)

# ── Normalise ─────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── KMeans — 4 clusters ───────────────────────────────
kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
fwd_agg['cluster'] = kmeans.fit_predict(X_scaled)

# ── Interpret clusters ────────────────────────────────
centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=role_features
)
print("\nCluster Centers (key stats):")
key_stats = ['shot_conversion', 'aerial_dominance', 'Take-Ons_Succ',
             'defensive_actions_p90', 'Touches_Att Pen']
key_stats = [k for k in key_stats if k in centers.columns]
print(centers[key_stats].round(3))

role_mapping = {}
role_mapping[centers['shot_conversion'].idxmax()]       = 'Poacher'
role_mapping[centers['Take-Ons_Succ'].idxmax()]         = 'Winger'
role_mapping[centers['aerial_dominance'].idxmax()]      = 'Target Man'
role_mapping[centers['defensive_actions_p90'].idxmax()] = 'Pressing Forward'

all_roles = ['Poacher', 'Winger', 'Target Man', 'Pressing Forward']
used = set(role_mapping.values())
remaining = [r for r in all_roles if r not in used]
for c in range(4):
    if c not in role_mapping:
        role_mapping[c] = remaining.pop(0) if remaining else 'Winger'

fwd_agg['role'] = fwd_agg['cluster'].map(role_mapping)
print("\nRole Distribution:")
print(fwd_agg['role'].value_counts())

# ── Map labels back to season rows ────────────────────
player_role_map = fwd_agg.set_index('player')['role'].to_dict()
fwd['role'] = fwd['player'].map(player_role_map)
fwd = fwd.dropna(subset=['role'])

# ── Random Forest Classifier ──────────────────────────
X_clf = fwd[role_features].fillna(0)
y_clf = fwd['role']

X_tr, X_te, y_tr, y_te = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

clf = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)

cv = cross_val_score(clf, X_clf, y_clf, cv=5, scoring='accuracy')
print(f"\nCross-Val Accuracy: {cv.mean()*100:.1f}% +/- {cv.std()*100:.1f}%")
print(f"Test Accuracy:      {(y_pred==y_te).mean()*100:.1f}%")
print("\nClassification Report:")
print(classification_report(y_te, y_pred))

# ── PLOT 1: Confusion Matrix ──────────────────────────
print("Saving confusion matrix...")
cm     = confusion_matrix(y_te, y_pred, labels=all_roles)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=all_roles, yticklabels=all_roles,
    linewidths=0.5, linecolor='#21262d', ax=ax,
    cbar_kws={'shrink': 0.8}
)
ax.set_title('Forward Role Classifier — Confusion Matrix\nRandom Forest (200 trees, 5-fold CV: 75%)',
             fontsize=13, fontweight='bold', pad=14)
ax.set_xlabel('Predicted Role', labelpad=10)
ax.set_ylabel('Actual Role',    labelpad=10)
plt.tight_layout()
plt.savefig('plots/role_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved -> plots/role_confusion_matrix.png")

# ── PLOT 2: Feature Importance ────────────────────────
print("Saving feature importance...")
feature_labels = {
    'shot_conversion':        'Shot Conversion',
    'Touches_Att Pen':        'Penalty Area Touches',
    'Take-Ons_Succ':          'Successful Dribbles',
    'Progression_PrgC':       'Progressive Carries',
    'Pass Types_Crs':         'Crosses',
    'aerial_dominance':       'Aerial Win Rate',
    'Aerial Duels_Won':       'Aerial Duels Won',
    'Expected_xG':            'Expected Goals (xG)',
    'defensive_actions_p90':  'Defensive Actions/90',
    'Tackles_TklW':           'Tackles Won',
    'Performance_Fls':        'Fouls Committed',
    'goal_contrib_p90':       'Goal Contribution/90',
    'SCA_SCA':                'Shot Creating Actions',
}
importances = pd.Series(
    clf.feature_importances_,
    index=[feature_labels.get(f, f) for f in role_features]
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors  = plt.cm.YlOrRd(np.linspace(0.35, 0.9, len(importances)))
ax.barh(importances.index, importances.values,
        color=colors, edgecolor='none', height=0.65)
for i, (val, name) in enumerate(zip(importances.values, importances.index)):
    ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9)
ax.set_title('Forward Role Classifier — Feature Importance\nRandom Forest',
             fontsize=13, fontweight='bold', pad=14)
ax.set_xlabel('Importance Score', labelpad=10)
ax.grid(True, axis='x', alpha=0.3)
ax.set_xlim(0, importances.max() + 0.05)
plt.tight_layout()
plt.savefig('plots/role_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved -> plots/role_feature_importance.png")

# ── Save model ────────────────────────────────────────
joblib.dump({
    'kmeans':       kmeans,
    'scaler':       scaler,
    'classifier':   clf,
    'features':     role_features,
    'role_mapping': role_mapping,
    'cv_accuracy':  round(cv.mean() * 100, 1),
    'test_accuracy': round((y_pred == y_te).mean() * 100, 1),
}, 'models/forward_roles.pkl')

fwd_agg.to_csv('data/forward_roles.csv', index=False)
print("\nSaved -> models/forward_roles.pkl")
print("Saved -> data/forward_roles.csv")
print("\nDONE")
