import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression

INPUT_FILE = Path("/home/diya/Downloads/capstone/outputs/features1_ml_ready.csv")
OUTPUT_DIR = Path("/home/diya/Downloads/capstone/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "shot_rate",
    "motion_mean",
    "motion_std",
    "edge_density",
    "luminance_change",
]

HIGHER_SHOTRATE_MEANS_HIGHER_LOAD = True


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


def prepare_data():
    df = pd.read_csv(INPUT_FILE)

    if "video_id" not in df.columns:
        df.insert(0, "video_id", range(1, len(df) + 1))

    thresholds = compute_thresholds(df)
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

    return df, X, X_train, X_test, y_train, y_test


def save_metrics_and_confusion_matrix(model_name, y_test, y_pred, output_prefix):
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    labels = ["Low", "Medium", "High"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    metrics_path = OUTPUT_DIR / f"{output_prefix}_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Macro: {f1_macro:.4f}\n")
        f.write(f"F1 Weighted: {f1_weighted:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(pd.DataFrame(cm, index=labels, columns=labels).to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax)
    ax.set_title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    cm_path = OUTPUT_DIR / f"{output_prefix}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)

    return metrics_path, cm_path, acc, f1_macro, f1_weighted


def save_predictions(df, X, model, output_prefix):
    cols = ["video_id"]
    for c in ["id", "duration"]:
        if c in df.columns:
            cols.append(c)

    pred_df = df[cols].copy()
    pred_df["actual_weak_label"] = df["risk_label"]
    pred_df["predicted_label"] = model.predict(X)

    probs = model.predict_proba(X)
    class_order = list(model.classes_)

    for i, cls in enumerate(class_order):
        pred_df[f"prob_{cls.lower()}"] = probs[:, i]

    class_to_score = {"Low": 0, "Medium": 1, "High": 2}
    pred_df["ml_risk_score"] = sum(
        pred_df[f"prob_{cls.lower()}"] * class_to_score[cls]
        for cls in class_order
    )

    pred_path = OUTPUT_DIR / f"{output_prefix}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    return pred_path


def main():
    df, X, X_train, X_test, y_train, y_test = prepare_data()

    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics_path, cm_path, acc, f1m, f1w = save_metrics_and_confusion_matrix(
        "Logistic Regression", y_test, y_pred, "logistic_regression"
    )
    pred_path = save_predictions(df, X, model, "logistic_regression")

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Macro: {f1m:.4f}")
    print(f"F1 Weighted: {f1w:.4f}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved confusion matrix image: {cm_path}")
    print(f"Saved predictions: {pred_path}")


if __name__ == "__main__":
    main()