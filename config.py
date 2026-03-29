from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]

    data_dir: Path = project_root / "Data"
    json_path: Path = data_dir / "trending.json"
    videos_dir: Path = data_dir / "videos"

    outputs_dir: Path = project_root / "outputs"
    logs_dir: Path = outputs_dir / "logs"

    target_fps: float = 3.0

    shot_hist_thresh: float = 0.50


