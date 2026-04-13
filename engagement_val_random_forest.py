import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve()

while not (PROJECT_ROOT / "outputs").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

print("Detected project root:", PROJECT_ROOT)

MODEL_NAME = "random_forest"


ML_FILE = PROJECT_ROOT / "outputs" / MODEL_NAME / "random_forest_predictions.csv"
TREND_FILE = PROJECT_ROOT / "Data" / "archive" / "youtube_shorts_tiktok_trends_2025.csv_ML.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / MODEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / f"{MODEL_NAME}_engagement_validation.txt"
QUANTILE_PLOT = OUTPUT_DIR / f"{MODEL_NAME}_engagement_quantile.png"


ml_df = pd.read_csv(ML_FILE)
trend_df = pd.read_csv(TREND_FILE)


needed_cols = [
    "like_rate",
    "comment_rate",
    "share_rate",
    "rel_like",
    "rel_share"
]

available_cols = [c for c in needed_cols if c in trend_df.columns]

if len(available_cols) < 2:
    raise ValueError("At least 2 engagement features required.")

trend_df = trend_df[available_cols].copy()

for col in trend_df.columns:
    trend_df[col] = pd.to_numeric(trend_df[col], errors="coerce")

trend_df = trend_df.replace([np.inf, -np.inf], np.nan).dropna().copy()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(trend_df[available_cols])

pca = PCA(n_components=1)
pc1_scores = pca.fit_transform(X_scaled).flatten()
loadings = pca.components_[0].copy()

if loadings.sum() < 0:
    loadings = -loadings
    pc1_scores = -pc1_scores

trend_df["behavior_score"] = pc1_scores

explained_variance = pca.explained_variance_ratio_[0]
pca_loadings = dict(zip(available_cols, loadings))

abs_loadings = np.abs(loadings)
contribution_pct = (abs_loadings / abs_loadings.sum()) * 100
pca_contributions = dict(zip(available_cols, contribution_pct))

equation_terms = [f"({loading:.4f} * z({feature}))" for feature, loading in pca_loadings.items()]
pca_equation = "behavior_score = " + " + ".join(equation_terms)


score_candidates = [c for c in ml_df.columns if "score" in c.lower()]
if not score_candidates:
    raise ValueError("No ML score column found.")

ml_score_col = score_candidates[0]
ml_scores = pd.to_numeric(ml_df[ml_score_col], errors="coerce")
ml_scores = ml_scores.replace([np.inf, -np.inf], np.nan).dropna()


def zscore(series):
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


ml_z = zscore(ml_scores.reset_index(drop=True))
eng_z = zscore(trend_df["behavior_score"].reset_index(drop=True))


q = np.linspace(0.05, 0.95, 19)
ml_q = np.quantile(ml_z, q)
eng_q = np.quantile(eng_z, q)

spearman_corr, spearman_p = spearmanr(ml_q, eng_q)


plt.figure(figsize=(8, 6))
plt.plot(q, ml_q, marker="o", label="ML Score")
plt.plot(q, eng_q, marker="o", label="PCA Behavior Score")
plt.xlabel("Quantile")
plt.ylabel("Standardized Value")
plt.legend()
plt.tight_layout()
plt.savefig(QUANTILE_PLOT, dpi=300)
plt.close()


with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    f.write(f"ML file: {ML_FILE}\n")
    f.write(f"Trend file: {TREND_FILE}\n")
    f.write(f"ML score column used: {ml_score_col}\n")
    f.write(f"Engagement columns used: {available_cols}\n\n")

    f.write(f"Explained variance (PC1): {explained_variance:.4f}\n\n")

    f.write("PCA loadings:\n")
    for feature, loading in pca_loadings.items():
        f.write(f"{feature}: {loading:.4f}\n")

    f.write("\nContribution (%):\n")
    for feature, pct in pca_contributions.items():
        f.write(f"{feature}: {pct:.2f}%\n")

    f.write("\nEquation:\n")
    f.write(pca_equation + "\n\n")

    f.write(f"Spearman correlation: {spearman_corr:.4f}\n")
    f.write(f"P-value: {spearman_p:.6f}\n")


print("Validation complete.")
print("Summary saved to:", SUMMARY_FILE)
print("Quantile plot saved to:", QUANTILE_PLOT)