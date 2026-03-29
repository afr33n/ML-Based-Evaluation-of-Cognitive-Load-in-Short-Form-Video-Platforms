import pandas as pd
from pathlib import Path

# Input file
INPUT_FILE = Path("/home/diya/Downloads/capstone/outputs/features1_norm.csv")

# Output file in the same folder
OUTPUT_FILE = INPUT_FILE.parent / "features1_ml_ready.csv"

# Read data
df = pd.read_csv(INPUT_FILE)

# Create simple sequential ID
df.insert(0, "video_id", range(1, len(df) + 1))

# Keep only useful columns
df_ml = df[
    [
        "video_id",
        "id",
        "duration",
        "shot_rate",
        "motion_mean",
        "motion_std",
        "edge_density",
        "luminance_change",
    ]
].copy()

# Save
df_ml.to_csv(OUTPUT_FILE, index=False)

print(f"Created: {OUTPUT_FILE}")
print(df_ml.head())
print("\nColumns kept:")
print(df_ml.columns.tolist())