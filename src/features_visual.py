import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class VisualFeatures:
    shot_rate: float
    motion_mean: float
    motion_std: float
    edge_density: float
    luminance_change: float


def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def hsv_hist(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv], [0, 1], None,
        [16, 16], [0, 180, 0, 256]
    )

    hist = cv2.normalize(hist, hist).flatten()

    return hist

# avg shot length
def compute_features(frames, fps, shot_thresh=0.50):

    n = len(frames)
    cuts = 0
    prev_hist = hsv_hist(frames[0])
    for i in range(1, n):
        curr_hist = hsv_hist(frames[i])
        diff = cv2.compareHist(
            prev_hist.astype(np.float32),
            curr_hist.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA
        )
        if diff > shot_thresh:
            cuts += 1
        prev_hist = curr_hist
    duration = n / fps
    shot_rate = cuts / duration

#optical flow gunnar farneback
    motions = []
    prev_gray = to_gray(frames[0])
    for i in range(1, n):
        gray = to_gray(frames[i])
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            0.5,   # pyr_scale
            3,     # levels
            15,    # winsize
            3,     # iterations
            5,     # poly_n
            1.2,   # poly_sigma
            0      # flags
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motions.append(np.mean(mag))
        prev_gray = gray
    motion_mean = float(np.mean(motions))
    motion_std = float(np.std(motions))


#edge density 
    edges_list = []

    for frame in frames:

        gray = to_gray(frame)

        edges = cv2.Canny(gray, 100, 200)

        edges_list.append(np.mean(edges > 0))


    edge_density = float(np.mean(edges_list))



#Luminance Change
    brightness = []

    for frame in frames:

        gray = to_gray(frame)

        brightness.append(np.mean(gray))


    brightness = np.array(brightness)

    diffs = np.abs(np.diff(brightness))

    luminance_change = float(np.mean(diffs) * fps)



    return VisualFeatures(
        shot_rate,
        motion_mean,
        motion_std,
        edge_density,
        luminance_change
    )
