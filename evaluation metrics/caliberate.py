# calibrate.py
# -----------------------------------------
# Calibrates SEMANTIC weight using Mohler ASAG
# -----------------------------------------

import json
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr

from trial3 import (
    embeddingManager,
    semantic_similarity_score,
)

# -----------------------------------------
# Feature extraction (SEMANTIC ONLY)
# -----------------------------------------
def extract_semantic_feature(
    student_answer: str,
    model_answer: str,
    embedding_manager
):
    semantic = semantic_similarity_score(
        student_answer,
        model_answer,
        embedding_manager
    )
    return [semantic]


# -----------------------------------------
# Main calibration routine
# -----------------------------------------
def calibrate_semantic_weight():

    print("🔹 Initializing embedding model...")
    embedding_manager = embeddingManager()

    print("🔹 Loading Mohler ASAG dataset...")
    dataset = load_dataset("nkazi/MohlerASAG", split="open_ended")
    dataset = dataset.shuffle(seed=42)

    print("Dataset columns:", dataset.column_names)

    train_size = int(0.7 * len(dataset))
    train_data = dataset.select(range(train_size))
    test_data = dataset.select(range(train_size, len(dataset)))

    X_train, y_train = [], []
    X_test, y_test = [], []

    print("🔹 Extracting training features...")
    for row in train_data:
        X_train.append(
            extract_semantic_feature(
                row["student_answer"],
                row["instructor_answer"],
                embedding_manager
            )
        )
        y_train.append(row["score_avg"])

    print("🔹 Extracting test features...")
    for row in test_data:
        X_test.append(
            extract_semantic_feature(
                row["student_answer"],
                row["instructor_answer"],
                embedding_manager
            )
        )
        y_test.append(row["score_avg"])

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("🔹 Training regression model...")
    reg = LinearRegression(positive=True)
    reg.fit(X_train, y_train)

    print("🔹 Evaluating model...")
    y_pred = reg.predict(X_test)

    pearson = pearsonr(y_test, y_pred)[0]

    y_test_round = np.round(y_test).astype(int)
    y_pred_round = np.round(y_pred).astype(int)

    qwk = cohen_kappa_score(
        y_test_round,
        y_pred_round,
        weights="quadratic"
    )

    print("\n===== EVALUATION RESULTS =====")
    print(f"Pearson Correlation: {pearson:.4f}")
    print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")

    # -----------------------------------------
    # Save evaluation plot
    # -----------------------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Human Score")
    plt.ylabel("Predicted Score")
    plt.title("Semantic Similarity Calibration (Mohler ASAG)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("semantic_calibration_plot.png")
    plt.close()

    # -----------------------------------------
    # Save learned semantic weight
    # -----------------------------------------
    semantic_weight = float(reg.coef_[0])
    bias = float(reg.intercept_)

    weights = {
        "semantic": semantic_weight,
        "bias": bias,
        "note": "Calibrated on Mohler ASAG (semantic-only)"
    }

    with open("semantic_weight.json", "w") as f:
        json.dump(weights, f, indent=2)

    print("\n===== LEARNED SEMANTIC WEIGHT =====")
    print(f"semantic: {semantic_weight:.4f}")
    print(f"bias: {bias:.4f}")

    print("\n✅ Calibration complete")
    print("📁 Saved: semantic_weight.json")
    print("📊 Saved: semantic_calibration_plot.png")


# -----------------------------------------
# Entry point
# -----------------------------------------
if __name__ == "__main__":
    calibrate_semantic_weight()
