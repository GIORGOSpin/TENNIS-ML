import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.calibration import calibration_curve

# optional plotting
do_plot = True
try:
    import matplotlib.pyplot as plt
except Exception:
    do_plot = False

def load_trained_model():
    if os.path.exists("trained_model.json"):
        with open("trained_model.json", "r", encoding="utf-8") as f:
            md = json.load(f)
        weights = np.array(md["weights"]).reshape(-1)
        bias = float(md.get("bias", 0.0))
        mean = np.array(md["mean"]) if "mean" in md else None
        std = np.array(md["std"]) if "std" in md else None
        feature_names = md.get("feature_names", None)
        return {"weights": weights, "bias": bias, "mean": mean, "std": std, "feature_names": feature_names}

    if os.path.exists("logreg_weights.npy") and os.path.exists("logreg_bias.npy"):
        weights = np.load("logreg_weights.npy")
        bias = float(np.load("logreg_bias.npy"))
        mean = np.load("logreg_mean.npy") if os.path.exists("logreg_mean.npy") else None
        std = np.load("logreg_std.npy") if os.path.exists("logreg_std.npy") else None
        feature_names = None
        if os.path.exists("feature_names.json"):
            with open("feature_names.json","r",encoding="utf-8") as f:
                feature_names = json.load(f)
        return {"weights": weights.reshape(-1), "bias": bias, "mean": mean, "std": std, "feature_names": feature_names}

    raise FileNotFoundError("Δεν βρέθηκε αποθηκευμένο μοντέλο (trained_model.json ή logreg_weights.npy).")

def sigmoid(z):
    z = np.array(z, dtype=np.float64)       # Ensure numpy array
    z = np.clip(z, -500, 500)               # Optional: avoid overflow
    return 1.0 / (1.0 + np.exp(-z))

def evaluate(input_csv="features_selected.csv", date_col="tourney_date", date_cutoff=20240925):
    df = pd.read_csv(input_csv)
    if "target" not in df.columns:
        raise ValueError("Το αρχείο δεν περιέχει 'target' στήλη.")
    feature_cols = [c for c in df.columns if c != "target"]

    # Διαχωρισμός με βάση ημερομηνία ή τυχαία
    if date_col in df.columns:
        try:
            df[date_col] = pd.to_numeric(df[date_col], errors="coerce")
        except Exception:
            pass
        train_df = df[df[date_col] < date_cutoff].copy()
        test_df = df[df[date_col] >= date_cutoff].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            print("Warning: το date-based split δημιούργησε άδειο train/test. Χρησιμοποιείται random split.")
            df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
            cut = int(0.8 * len(df_shuffled))
            train_df = df_shuffled.iloc[:cut]
            test_df = df_shuffled.iloc[cut:]
    else:
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        cut = int(0.8 * len(df_shuffled))
        train_df = df_shuffled.iloc[:cut]
        test_df = df_shuffled.iloc[cut:]

    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    model = load_trained_model()
    weights = model["weights"]
    bias = model["bias"]
    mean = model["mean"]
    std = model["std"]
    feature_names_saved = model.get("feature_names", None)

    if feature_names_saved is not None:
        if set(feature_names_saved) != set(feature_cols):
            missing = [f for f in feature_names_saved if f not in feature_cols]
            if missing:
                raise ValueError(f"Feature names saved do not match current features. Missing: {missing}")
            X_test = test_df[feature_names_saved].values
            feature_cols = feature_names_saved
    else:
        if X_test.shape[1] != weights.shape[0]:
            raise ValueError(f"Mismatch features ({X_test.shape[1]}) vs weights ({weights.shape[0]}).")

    if mean is not None and std is not None:
        # Χρήση αποθηκευμένου mean/std
        X_test_norm = (X_test - mean) / (std + 1e-9)
    else:
        X_test_norm = (X_test - X_test.mean(axis=0)) / (X_test.std(axis=0) + 1e-9)
        print("Προειδοποίηση: Δεν βρέθηκε mean/std από το training. Χρησιμοποιήθηκε scaling του test ως fallback (μπορεί να αλλοιώσει το αποτέλεσμα).")

    logits = np.dot(X_test_norm, weights) + bias
    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(int).reshape(-1)

    # Μετρικές
    ll = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = float("nan")
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)

    from collections import Counter
    c = Counter(y_test)
    maj_class = c.most_common(1)[0][0]
    baseline_acc = sum(y_test == maj_class) / len(y_test)

    print("\n=== Evaluation results ===")
    print(f"LogLoss: {ll:.5f}")
    print(f"Brier score: {brier:.5f}")
    print(f"ROC AUC: {auc if not np.isnan(auc) else 'N/A'}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Baseline (majority) accuracy: {baseline_acc:.4f}")
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
    print("Confusion matrix:")
    print(cm)

    try:
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
        print("\nCalibration (per-bin):")
        for i in range(len(prob_true)):
            print(f" bin {i+1}: pred_avg={prob_pred[i]:.3f}, true_frac={prob_true[i]:.3f}")
        if do_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,6))
            plt.plot(prob_pred, prob_true, marker='o')
            plt.plot([0,1],[0,1], linestyle='--', color='gray')
            plt.xlabel("Average predicted probability (bin)")
            plt.ylabel("Fraction of positives (true)")
            plt.title("Calibration plot")
            plt.grid(True)
            plt.savefig("calibration_plot.png", dpi=150)
            print("Calibration plot saved to calibration_plot.png")
    except Exception as e:
        print("Calibration curve failed:", e)

    summary = {
        "logloss": float(ll),
        "brier": float(brier),
        "auc": float(auc) if not np.isnan(auc) else None,
        "accuracy": float(acc),
        "baseline_accuracy": float(baseline_acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm.tolist()
    }
    return summary

if __name__ == "__main__":
    res = evaluate()
    print("\nDone.")
