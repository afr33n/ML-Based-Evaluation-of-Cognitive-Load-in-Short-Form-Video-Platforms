from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from src.config import Config
from src.io_utils import ensure_dir, read_json
from src.metadata import load_id_duration_from_json, find_video_path
from src.video_sampling import sample_video_frames
from src.features_visual import compute_features


def extract_all(cfg: Config) -> pd.DataFrame:

    ensure_dir(cfg.outputs_dir)



    data = read_json(cfg.json_path)

    meta = load_id_duration_from_json(data)


    meta = meta[
        (meta["duration_sec_json"].notna()) &
        (meta["duration_sec_json"] >= 15)
    ]


    results = []


    for _, row in tqdm(meta.iterrows(), total=len(meta)):

        vid = row["id"]
        duration = row["duration_sec_json"]


        path = find_video_path(cfg.videos_dir, vid)

        if path is None:
            continue


        try:

            sampled = sample_video_frames(
                str(path),
                cfg.target_fps
            )


            feats = compute_features(
                sampled.frames,
                sampled.sampled_fps,
                cfg.shot_hist_thresh
            )


            # Store result
            results.append({

                "id": vid,
                "duration": duration,
                "video_path": str(path),

                "original_fps": sampled.original_fps,
                "sampled_fps": sampled.sampled_fps,
                "num_frames": len(sampled.frames),

                "shot_rate": feats.shot_rate,
                "motion_mean": feats.motion_mean,
                "motion_std": feats.motion_std,
                "edge_density": feats.edge_density,
                "luminance_change": feats.luminance_change,

            })


        except Exception as e:
            print(f"Error in {vid}: {e}")



    df = pd.DataFrame(results)

    out_file = cfg.outputs_dir / "features1.csv"

    df.to_csv(out_file, index=False)

    print("Saved to:", out_file)

    return df
