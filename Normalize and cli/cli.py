from pathlib import Path
import pandas as pd


IN_1FPS = Path("outputs/features1_norm.csv")
IN_3FPS = Path("outputs/features_norm.csv")

OUT_1FPS = Path("outputs/cli_1fps.csv")
OUT_3FPS = Path("outputs/cli_3fps.csv")

ID_COL = "id"

FEATURES = [
    "shot_rate",
    "motion_mean",
    "motion_std",
    "edge_density",
    "luminance_change",
]


def compute_cli(df):

    out = df[[ID_COL] + FEATURES].copy()
    out["cli"] = out[FEATURES].mean(axis=1)
    return out


def process(in_file, out_file):
    df = pd.read_csv(in_file)
    cli_df = compute_cli(df)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cli_df.to_csv(out_file, index=False)
    print("Saved:", out_file)


def main():
    process(IN_1FPS, OUT_1FPS)
    process(IN_3FPS, OUT_3FPS)


if __name__ == "__main__":
    main()
