from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -------- CONFIG --------

CSV_1FPS = Path("outputs/features1.csv")   # 1 FPS
CSV_3FPS = Path("outputs/features.csv")    # 3 FPS

OUT_DIR = Path("outputs/plots/stability")

ID_COL = "id"

FEATURES = [
    "shot_rate",
    "motion_mean",
    "motion_std",
    "edge_density",
    "luminance_change",
]

# ------------------------


def spearman(x, y):
    """Compute Spearman correlation"""

    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()

    return np.corrcoef(rx, ry)[0, 1]


def plot_feature(df, col1, col3, name):

    x = df[col1].to_numpy()
    y = df[col3].to_numpy()

    # Remove NaN / inf
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) == 0:
        print(f"Skipping {name}: no valid data")
        return

    rho = spearman(x, y)

    plt.figure()

    plt.scatter(x, y, alpha=0.7)

    # y = x reference line
    mn = min(x.min(), y.min())
    mx = max(x.max(), y.max())

    plt.plot([mn, mx], [mn, mx])

    plt.xlabel("1 FPS")
    plt.ylabel("3 FPS")

    plt.title(f"Stability: {name}")

    plt.text(
        0.02, 0.98,
        f"n = {len(x)}\nρ = {rho:.3f}",
        transform=plt.gca().transAxes,
        va="top"
    )

    plt.tight_layout()

    plt.savefig(OUT_DIR / f"stability_{name}.png", dpi=200)

    plt.close()


def main():

    # Create output folder
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load CSV files
    df1 = pd.read_csv(CSV_1FPS)
    df3 = pd.read_csv(CSV_3FPS)

    # Keep only needed columns
    cols = [ID_COL] + FEATURES

    df1 = df1[cols]
    df3 = df3[cols]

    # Merge on video ID
    df = df1.merge(df3, on=ID_COL, suffixes=("_1fps", "_3fps"))

    # Plot each feature
    for feat in FEATURES:

        plot_feature(
            df,
            f"{feat}_1fps",
            f"{feat}_3fps",
            feat
        )

    print("Stability plots saved in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
