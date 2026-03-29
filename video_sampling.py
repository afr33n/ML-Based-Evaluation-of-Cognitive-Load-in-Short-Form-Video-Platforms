import cv2
from dataclasses import dataclass


@dataclass
class SampledFrames:
    frames: list
    original_fps: float
    sampled_fps: float


def sample_video_frames(video_path, target_fps):

    cap = cv2.VideoCapture(video_path)

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    step = int(original_fps / target_fps)
    if step < 1:
        step = 1

    sampled_fps = original_fps / step

    frames = []
    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % step == 0:
            frames.append(frame)

        frame_idx += 1

    cap.release()

    return SampledFrames(
        frames=frames,
        original_fps=original_fps,
        sampled_fps=sampled_fps
    )
