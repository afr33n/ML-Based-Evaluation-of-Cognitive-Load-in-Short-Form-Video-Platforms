import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# =========================
# BASE PROJECT PATH
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =========================
# FILE PATHS
# =========================
PREDICTIONS_FILE = PROJECT_ROOT / "outputs" / "gradient_boost" / "gradient_boosting_predictions.csv"
CLI_FILE = PROJECT_ROOT / "outputs" / "cli_3fps.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradient_boost"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MERGED_FILE = OUTPUT_DIR / "gradient_boosting_cli_validation.csv"
METRICS_FILE = OUTPUT_DIR / "gradient_boosting_cli_metrics.txt"
SCATTER_PLOT = OUTPUT_DIR / "gradient_boosting_cli_scatter.png"
BOX_PLOT = OUTPUT_DIR / "gradient_boosting_cli_boxplot.png"
CLASS_SUMMARY_FILE = OUTPUT_DIR / "gradient_boosting_cli_class_summary.csv"
CROSSTAB_FILE = OUTPUT_DIR / "gradient_boosting_cli_crosstab.csv"


def merge_with_cli(pred_df: pd.DataFrame, cli_df: pd.DataFrame) -> pd.DataFrame:
    if "CLI" in cli_df.columns:
        cli_col = "CLI"
    elif "cli" in cli_df.columns:
        cli_col = "cli"
    else:
        raise ValueError("CLI file must contain a column named 'CLI' or 'cli'.")

    if "video_id" in pred_df.columns and "video_id" in cli_df.columns:
        merged = pred_df.merge(cli_df[["video_id", cli_col]], on="video_id", how="inner")
    elif "id" in pred_df.columns and "id" in cli_df.columns:
        merged = pred_df.merge(cli_df[["id", cli_col]], on="id", how="inner")
    else:
        raise ValueError("Need a common merge key: 'video_id' or 'id' in both files.")

    merged = merged.rename(columns={cli_col: "CLI"})
    return merged


def validate_against_cli(merged: pd.DataFrame) -> None:
    pearson_corr, pearson_p = pearsonr(merged["ml_risk_score"], merged["CLI"])
    spearman_corr, spearman_p = spearmanr(merged["ml_risk_score"], merged["CLI"])

    merged["cli_class"] = pd.qcut(merged["CLI"], 3, labels=["Low", "Medium", "High"])

    crosstab = pd.crosstab(merged["predicted_label"], merged["cli_class"])
    classwise_cli = merged.groupby("predicted_label")["CLI"].agg(["mean", "median", "count"])

    classwise_cli.to_csv(CLASS_SUMMARY_FILE)
    crosstab.to_csv(CROSSTAB_FILE)

    with open(METRICS_FILE, "w") as f:
        f.write("Gradient Boosting vs CLI Validation\n")
        f.write("==================================\n\n")
        f.write(f"Pearson correlation: {pearson_corr:.4f}\n")
        f.write(f"Pearson p-value: {pearson_p:.6f}\n\n")
        f.write(f"Spearman correlation: {spearman_corr:.4f}\n")
        f.write(f"Spearman p-value: {spearman_p:.6f}\n\n")
        f.write("CLI summary by predicted label:\n")
        f.write(classwise_cli.to_string())
        f.write("\n\nPredicted label vs CLI class:\n")
        f.write(crosstab.to_string())

    plt.figure(figsize=(7, 6))
    plt.scatter(merged["CLI"], merged["ml_risk_score"])
    plt.xlabel("CLI")
    plt.ylabel("Gradient Boosting ML Risk Score")
    plt.title("CLI vs Gradient Boosting ML Risk Score")
    plt.tight_layout()
    plt.savefig(SCATTER_PLOT, dpi=200)
    plt.close()

    order = ["Low", "Medium", "High"]
    data = [merged.loc[merged["predicted_label"] == cls, "CLI"] for cls in order]

    plt.figure(figsize=(7, 6))
    plt.boxplot(data, tick_labels=order)
    plt.xlabel("Predicted Label")
    plt.ylabel("CLI")
    plt.title("CLI Distribution by Gradient Boosting Predicted Label")
    plt.tight_layout()
    plt.savefig(BOX_PLOT, dpi=200)
    plt.close()

    merged.to_csv(MERGED_FILE, index=False)

    print("\n================ CLI VALIDATION ================\n")
    print(f"Pearson correlation: {pearson_corr:.4f}")
    print(f"Pearson p-value: {pearson_p:.6f}")
    print(f"Spearman correlation: {spearman_corr:.4f}")
    print(f"Spearman p-value: {spearman_p:.6f}")

    print("\nCLI summary by predicted label:\n")
    print(classwise_cli)

    print("\nPredicted label vs CLI class:\n")
    print(crosstab)

    print("\nSaved files:")
    print(f"- Merged validation file: {MERGED_FILE}")
    print(f"- Metrics text: {METRICS_FILE}")
    print(f"- Scatter plot: {SCATTER_PLOT}")
    print(f"- Box plot: {BOX_PLOT}")
    print(f"- Class summary CSV: {CLASS_SUMMARY_FILE}")
    print(f"- Crosstab CSV: {CROSSTAB_FILE}")


def main():
    pred_df = pd.read_csv(PREDICTIONS_FILE)
    cli_df = pd.read_csv(CLI_FILE)

    merged = merge_with_cli(pred_df, cli_df)
    validate_against_cli(merged)


if __name__ == "__main__":
    main()
