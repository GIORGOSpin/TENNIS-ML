import pandas as pd
import numpy as np

def feature_selection(input_csv='features_two_vector.csv', output_csv='features_selected.csv', target='target', threshold=0.05):
    # Φόρτωση δεδομένων με χαρακτηριστικά και στόχο
    df = pd.read_csv(input_csv)

    # Διαχωρισμός χαρακτηριστικών και στόχου
    X = df.drop(columns=[target])
    y = df[target]

    correlations = {}

    # Υπολογισμός απόλυτης συσχέτισης Pearson για κάθε αριθμητικό χαρακτηριστικό
    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.number):
            corr = np.corrcoef(X[col], y)[0,1]
            correlations[col] = abs(corr)

    # Επιλογή χαρακτηριστικών με συσχέτιση >= threshold
    selected_features = [col for col, corr in correlations.items() if corr >= threshold]

    print(f"Χαρακτηριστικά που επιλέχθηκαν με συσχέτιση >= {threshold}:")
    print(selected_features)

    # Δημιουργία νέου DataFrame με τα επιλεγμένα χαρακτηριστικά και τον στόχο
    df_selected = df[selected_features + [target]]

    # Αποθήκευση του νέου dataset
    df_selected.to_csv(output_csv, index=False)
    print(f"Αποθηκεύτηκαν τα επιλεγμένα χαρακτηριστικά στο αρχείο: {output_csv}")

if __name__ == "__main__":
    feature_selection()
