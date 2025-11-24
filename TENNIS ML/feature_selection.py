import pandas as pd
import numpy as np

def feature_selection(input_csv='features_two_vector.csv', output_csv='features_selected.csv', target='target', threshold=0.05):
    # Load data
    df = pd.read_csv(input_csv)

    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Identify one-hot encoded categorical columns
    one_hot_cols = [col for col in X.columns if col.startswith('surface_') or col.startswith('tourney_level_')]

    # Label encoded categorical columns to always keep
    label_encoded_cols = ['round_enc', 'tourney_name_enc', 'tourney_id_enc']

    # Filter to those label encoded columns that actually exist in the dataframe
    label_encoded_cols = [col for col in label_encoded_cols if col in X.columns]

    # Compute Pearson correlation for numerical features excluding one-hot and label encoded
    correlations = {}
    for col in X.columns:
        if col in one_hot_cols or col in label_encoded_cols:
            continue
        if np.issubdtype(X[col].dtype, np.number):
            corr = np.corrcoef(X[col], y)[0,1]
            correlations[col] = abs(corr)

    # Select numerical features passing threshold
    selected_features = [col for col, corr in correlations.items() if corr >= threshold]

    # Add all one-hot encoded columns and label encoded categorical columns
    selected_features += one_hot_cols + label_encoded_cols

    print(f"Selected features (numerical passing threshold + all categorical):")
    print(selected_features)

    # Create new dataframe with selected features and target
    df_selected = df[selected_features + [target]]

    # Save to CSV
    df_selected.to_csv(output_csv, index=False)
    print(f"Saved selected features to {output_csv}")

if __name__ == "__main__":
    feature_selection()
