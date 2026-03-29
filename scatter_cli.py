from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -------- CONFIG --------

CLI_1FPS = Path("outputs/cli_1fps.csv")
CLI_3FPS = Path("outputs/cli_3fps.csv")

OUT_PLOT = Path("outputs/plots/cli_stability.png")

ID_COL = "id"
CLI_COL = "cli"

# ------------------------


def spearman(x, y):

    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()

    return np.corrcoef(rx, ry)[0, 1]


def main():

    df1 = pd.read_csv(CLI_1FPS)
    df3 = pd.read_csv(CLI_3FPS)

    # Merge on id
    df = df1[[ID_COL, CLI_COL]].merge(
        df3[[ID_COL, CLI_COL]],
        on=ID_COL,
        suffixes=("_1fps", "_3fps")
    )

    x = df["cli_1fps"].to_numpy()
    y = df["cli_3fps"].to_numpy()

    # Remove NaN/inf
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    rho = spearman(x, y)

    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    plt.scatter(x, y, alpha=0.7)

    mn = min(x.min(), y.min())
    mx = max(x.max(), y.max())

    plt.plot([mn, mx], [mn, mx])

    plt.xlabel("CLI @ 1 FPS")
    plt.ylabel("CLI @ 3 FPS")

    plt.title(f"CLI Stability (ρ = {rho:.3f})")

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=200)
    plt.close()

    print("Saved:", OUT_PLOT)


if __name__ == "__main__":
    main()
