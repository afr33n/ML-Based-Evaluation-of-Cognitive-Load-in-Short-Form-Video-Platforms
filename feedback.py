import pandas as pd
import numpy as np
import shap
import joblib
from pathlib import Path

# =========================
# PATH
# =========================
PROJECT_ROOT = Path(__file__).resolve()

while not (PROJECT_ROOT / "outputs").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

# =========================
# FILES
# =========================
FEATURE_FILE = PROJECT_ROOT / "outputs" / "features_norm.csv"
MODEL_FILE = PROJECT_ROOT / "outputs" / "random_forest" / "random_forest_model.pkl"

# =========================
# LOAD DATA + MODEL
# =========================
df = pd.read_csv(FEATURE_FILE)
model = joblib.load(MODEL_FILE)

# =========================
# FEATURES
# =========================
feature_cols = [
    "shot_rate",
    "motion_mean",
    "motion_std",
    "edge_density",
    "luminance_change"
]

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

X = df[feature_cols].copy()

# =========================
# SHAP
# =========================
explainer = shap.TreeExplainer(model)
classes = list(model.classes_)

# =========================
# MAPS
# =========================
reason_map = {
    "shot_rate": "high cut rate with very frequent scene changes",
    "motion_mean": "high motion intensity",
    "motion_std": "unstable or inconsistent motion",
    "edge_density": "high visual clutter",
    "luminance_change": "abrupt brightness changes"
}

suggestion_map = {
    "shot_rate": "Reduce frequent scene cuts and keep shots slightly longer",
    "motion_mean": "Reduce excessive camera or object movement",
    "motion_std": "Keep motion more consistent across scenes",
    "edge_density": "Simplify crowded frames and reduce background clutter",
    "luminance_change": "Avoid sudden brightness or flashing transitions"
}

# =========================
# FEEDBACK FUNCTION
# =========================
def get_video_feedback(video_row):
    video_df = pd.DataFrame([video_row[feature_cols].to_dict()])

    predicted_risk = model.predict(video_df)[0]

    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(video_df)[0]))
    else:
        confidence = None

    shap_values = explainer.shap_values(video_df)
    shap_arr = np.array(shap_values)
    class_idx = classes.index(predicted_risk)

    # Handle common SHAP output formats
    if isinstance(shap_values, list):
        row_shap = shap_values[class_idx][0]
    elif shap_arr.ndim == 3:
        # common case: (n_samples, n_features, n_classes)
        if shap_arr.shape[0] == 1 and shap_arr.shape[1] == len(feature_cols):
            row_shap = shap_arr[0, :, class_idx]
        # alternate case: (n_classes, n_samples, n_features)
        elif shap_arr.shape[0] == len(classes):
            row_shap = shap_arr[class_idx, 0, :]
        else:
            raise ValueError(f"Unexpected SHAP shape: {shap_arr.shape}")
    elif shap_arr.ndim == 2:
        row_shap = shap_arr[0]
    else:
        raise ValueError(f"Unexpected SHAP shape: {shap_arr.shape}")

    abs_shap = np.abs(row_shap)
    max_contrib = abs_shap.max()

    # keep only features with meaningful effect
    threshold = 0.25 * max_contrib
    selected_idx = np.where(abs_shap >= threshold)[0]

    if len(selected_idx) == 0:
        selected_idx = [int(np.argmax(abs_shap))]

    selected_idx = np.array(selected_idx)[np.argsort(abs_shap[selected_idx])[::-1]]

    selected_features = [feature_cols[j] for j in selected_idx]

    # project goal: only act when risk is high
    if predicted_risk == "High":
        reasons = [reason_map[f] for f in selected_features]
        suggestions = [suggestion_map[f] for f in selected_features]
    elif predicted_risk == "Medium":
        reasons = ["visual features are within a moderate range"]
        suggestions = ["Cognitive load appears moderate and manageable. No corrective action is required."]
    else:  # Low
        reasons = ["visual features indicate low cognitive load"]
        suggestions = ["Cognitive load appears low and does not require corrective action."]

    return {
        "id": video_row["id"] if "id" in video_row else "N/A",
        "predicted_risk": predicted_risk,
        "confidence": confidence,
        "selected_features": selected_features,
        "reasons": reasons,
        "suggestions": suggestions
    }

# =========================
# SAMPLE VIDEOS AND SHOW FEEDBACK
# =========================
sample_df = df.iloc[16:21]

for _, row in sample_df.iterrows():
    result = get_video_feedback(row)

    print("=" * 70)
    print("Video ID:", result["id"])
    print("Predicted Risk:", result["predicted_risk"])

    if result["confidence"] is not None:
        print("Confidence:", round(result["confidence"], 4))

    print("Important Features:", ", ".join(result["selected_features"]))

    print("Reasons:")
    for r in result["reasons"]:
        print(" -", r)

    print("Suggestions:")
    for s in result["suggestions"]:
        print(" -", s)