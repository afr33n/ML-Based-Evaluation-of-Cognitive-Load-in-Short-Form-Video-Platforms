import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# =========================
# FILE PATHS
# =========================
INPUT_FILE = Path("/home/diya/Downloads/capstone/outputs/features1_ml_ready.csv")
OUTPUT_DIR = Path("/home/diya/Downloads/capstone/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = OUTPUT_DIR / "gradient_boosting_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "gradient_boosting_model_metrics.txt"
CONFUSION_MATRIX_PLOT = OUTPUT_DIR / "gradient_boosting_confusion_matrix.png"
CLASSIFICATION_REPORT_CSV = OUTPUT_DIR / "gradient_boosting_classification_report.csv"

FEATURES = [
    "shot_rate",
    "motion_mean",
    "motion_std",
    "edge_density",
    "luminance_change",
]

HIGHER_SHOTRATE_MEANS_HIGHER_LOAD = True


# =========================
# WEAK LABELS
# =========================
def compute_thresholds(df: pd.DataFrame) -> dict:
    thresholds = {}
    for col in FEATURES:
        thresholds[col] = {
            "low": df[col].quantile(0.30),
            "high": df[col].quantile(0.70),
        }
    return thresholds


def assign_label(row: pd.Series, thresholds: dict) -> str:
    high_score = 0
    low_score = 0

    shot_low = thresholds["shot_rate"]["low"]
    shot_high = thresholds["shot_rate"]["high"]

    if HIGHER_SHOTRATE_MEANS_HIGHER_LOAD:
        if row["shot_rate"] >= shot_high:
            high_score += 1
        elif row["shot_rate"] <= shot_low:
            low_score += 1
    else:
        if row["shot_rate"] <= shot_low:
            high_score += 1
        elif row["shot_rate"] >= shot_high:
            low_score += 1

    if row["motion_mean"] >= thresholds["motion_mean"]["high"]:
        high_score += 1
    elif row["motion_mean"] <= thresholds["motion_mean"]["low"]:
        low_score += 1

    if row["motion_std"] >= thresholds["motion_std"]["high"]:
        high_score += 1
    elif row["motion_std"] <= thresholds["motion_std"]["low"]:
        low_score += 1

    if row["edge_density"] >= thresholds["edge_density"]["high"]:
        high_score += 1
    elif row["edge_density"] <= thresholds["edge_density"]["low"]:
        low_score += 1

    if row["luminance_change"] >= thresholds["luminance_change"]["high"]:
        high_score += 1
    elif row["luminance_change"] <= thresholds["luminance_change"]["low"]:
        low_score += 1

    if high_score >= 3 and high_score > low_score:
        return "High"
    elif low_score >= 3 and low_score > high_score:
        return "Low"
    else:
        return "Medium"


# =========================
# TRAIN MODEL ONLY
# =========================
def main():
    df = pd.read_csv(INPUT_FILE)

    if "video_id" not in df.columns:
        df.insert(0, "video_id", range(1, len(df) + 1))

    thresholds = compute_thresholds(df)
    df = df.copy()
    df["risk_label"] = df.apply(assign_label, axis=1, thresholds=thresholds)

    X = df[FEATURES]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)

    # Cross-validation on full dataset
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    # Train on train split
    model.fit(X_train, y_train)

    # Test predictions
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    report_text = classification_report(y_test, y_test_pred)
    report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_test_pred)
    class_names = list(model.classes_)

    # Save classification report as CSV
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(CLASSIFICATION_REPORT_CSV)

    # Save confusion matrix plot
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Gradient Boosting Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PLOT, dpi=200)
    plt.close()

    # Predict on full dataset and save for separate CLI validation
    probs = model.predict_proba(X)
    preds = model.predict(X)

    cols = ["video_id"] if "video_id" in df.columns else []
    for c in ["id", "duration"]:
        if c in df.columns:
            cols.append(c)

    pred_df = df[cols].copy()
    pred_df["actual_weak_label"] = df["risk_label"]
    pred_df["predicted_label"] = preds

    for i, cls in enumerate(class_names):
        pred_df[f"prob_{cls.lower()}"] = probs[:, i]

    class_to_score = {"Low": 0, "Medium": 1, "High": 2}
    pred_df["ml_risk_score"] = sum(
        pred_df[f"prob_{cls.lower()}"] * class_to_score[cls]
        for cls in class_names
    )

    pred_df.to_csv(PREDICTIONS_FILE, index=False)

    # Save text metrics
    with open(METRICS_FILE, "w") as f:
        f.write("Gradient Boosting Model Evaluation\n")
        f.write("=================================\n\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
        f.write("5-Fold Cross-Validation Accuracy Scores:\n")
        f.write(", ".join(f"{score:.4f}" for score in cv_scores))
        f.write(f"\nMean CV Accuracy: {cv_scores.mean():.4f}\n")
        f.write(f"Std CV Accuracy: {cv_scores.std():.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    # Print directly
    print("\n================ MODEL PERFORMANCE ================\n")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("5-Fold CV Accuracy Scores:", ", ".join(f"{score:.4f}" for score in cv_scores))
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
    print(f"Std CV Accuracy: {cv_scores.std():.4f}")

    print("\nClassification Report:\n")
    print(report_text)

    print("\nConfusion Matrix:\n")
    print(cm)

    print("\nSaved files:")
    print(f"- Predictions: {PREDICTIONS_FILE}")
    print(f"- Metrics text: {METRICS_FILE}")
    print(f"- Classification report CSV: {CLASSIFICATION_REPORT_CSV}")
    print(f"- Confusion matrix plot: {CONFUSION_MATRIX_PLOT}")


if __name__ == "__main__":
    main()