import subprocess
import sys
import time

# Βοηθητική συνάρτηση για να τρέχουμε κάθε script
def run_step(step_name, filename):
    print("\n" + "="*60)
    print(f"➡️  Τρέχει: {step_name}  ({filename}.py)")
    print("="*60)

    start = time.time()
    result = subprocess.run([sys.executable, f"{filename}.py"])

    end = time.time()
    print(f"⏳ Χρόνος εκτέλεσης: {end - start:.2f} sec")

    if result.returncode != 0:
        print(f"❌ Σφάλμα στο {step_name}! Σταματά η διαδικασία.")
        sys.exit(1)

    print(f"✅ Ολοκληρώθηκε: {step_name}")


if __name__ == "__main__":
import pandas as pd
import numpy as np


def safe_pearson(x, y):
    """Pearson correlation that cannot crash.
    Returns 0 if column is constant or invalid."""
    if x.std() == 0 or y.std() == 0:
        return 0
    try:
        return np.corrcoef(x, y)[0, 1]
    except:
        return 0


def feature_selection(
    input_csv='features_two_vector.csv',
    output_csv='features_selected.csv',
    target='target',
    threshold=0.03  # relaxed threshold to avoid dropping weak-but-useful features
):
    print("Loading engineered features...")
    df = pd.read_csv(input_csv)

    X = df.drop(columns=[target])
    y = df[target]

    # One-hot encoded categorical columns — always keep
    one_hot_cols = [
        col for col in X.columns
        if col.startswith('surface_') or col.startswith('tourney_level_')
    ]

    # Label-encoded categorical columns — always keep
    label_encoded_cols = [
        col for col in ['round_enc', 'tourney_name_enc', 'tourney_id_enc']
        if col in X.columns
    ]

    # Engineered difference features — always keep regardless of correlation score
    always_keep = [
        col for col in [
            'rank_diff',
            'rank_points_diff',
            'age_diff',
            'height_diff',
            'seed_diff',
            'experience_diff',
            'p1_is_seeded',
            'p2_is_seeded'
        ] if col in X.columns
    ]

    # Numerical columns — filter by Pearson correlation threshold
    exclude_from_filter = set(one_hot_cols + label_encoded_cols + always_keep)
    numeric_cols = [
        col for col in X.columns
        if np.issubdtype(X[col].dtype, np.number)
        and col not in exclude_from_filter
    ]

    correlations = {}
    for col in numeric_cols:
        corr = abs(safe_pearson(X[col], y))
        correlations[col] = corr

    selected_numeric = [col for col, corr in correlations.items() if corr >= threshold]

    # Final feature set
    final_features = sorted(list(set(
        selected_numeric +
        one_hot_cols +
        label_encoded_cols +
        always_keep
    )))

    print("\n=== SELECTED FEATURES ===")
    for f in final_features:
        print(f)

    df_out = df[final_features + [target]]
    df_out.to_csv(output_csv, index=False)

    print(f"\nTotal features selected: {len(final_features)}")
    print(f"Saved to {output_csv}")


if __name__ == "__main__":
    feature_selection()
    print("\n🚀 Ξεκινάει η πλήρης διαδικασία pipeline για ATP matches...\n")

    run_step("1. Load & Clean Data", "load_clean_data")
    run_step("2. Feature Engineering (Two-Vector)", "feature_engineering")
    run_step("3. Feature Selection", "feature_selection")
    run_step("4. Train Model (LogReg)", "train_model")
    run_step("5. Evaluate Model", "evaluate_model")

    print("\n🎉 ΟΛΟΚΛΗΡΟ ΤΟ PIPELINE ΕΚΤΕΛΕΣΤΗΚΕ ΕΠΙΤΥΧΩΣ!")
