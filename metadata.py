from pathlib import Path
import pandas as pd

def load_id_duration_from_json(data: dict) -> pd.DataFrame:

    rows = []
    for item in data.get("collector", []):
        vid = str(item.get("id", ""))
        if not vid:
            continue

        video_meta = item.get("videoMeta") or {}
        duration = video_meta.get("duration", None)

        rows.append({
            "id": vid,
            "duration_sec_json": duration,
        })

    df = pd.DataFrame(rows)
    df = df[df["id"].str.len() > 0].copy()
    return df

def find_video_path(video_dir: Path, video_id: str) -> Path | None:
    p = video_dir / f"{video_id}.mp4"
    if p.exists():
        return p

    return None
