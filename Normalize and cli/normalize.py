from pathlib import Path
import pandas as pd

CSV_1FPS = Path("outputs/features1.csv")
CSV_3FPS = Path("outputs/features.csv")

OUT_1FPS = Path("outputs/features1_norm.csv")
OUT_3FPS = Path("outputs/features_norm.csv")

ID_COL = "id"

FEATURES = [
    "shot_rate",
    "motion_mean",
    "motion_std",
    "edge_density",
    "luminance_change",
]

FIT_ON = "3fps"

def main():

    df1 = pd.read_csv(CSV_1FPS)
    df3 = pd.read_csv(CSV_3FPS)

    fit_df = df3 if FIT_ON == "3fps" else df1

    params = {}

    for col in FEATURES:
        mean = fit_df[col].mean()
        std = fit_df[col].std(ddof=0)
        if std == 0 or pd.isna(std):
            std = 1.0
        params[col] = (mean, std)

    def normalize(df):
        out = df.copy()
        for col in FEATURES:
            mean, std = params[col]
            out[col] = (out[col] - mean) / std
        return out


    df1_norm = normalize(df1)
    df3_norm = normalize(df3)

    OUT_1FPS.parent.mkdir(parents=True, exist_ok=True)

    df1_norm.to_csv(OUT_1FPS, index=False)
    df3_norm.to_csv(OUT_3FPS, index=False)


if __name__ == "__main__":
    main()
