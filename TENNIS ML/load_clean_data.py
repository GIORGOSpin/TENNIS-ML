import pandas as pd
import numpy as np

# Φόρτωση ακατέργαστων δεδομένων
df = pd.read_csv("atp_matches_2024.csv")

# Έλεγχος ελλιπών τιμών πριν τον καθαρισμό
print("Απουσίες πριν τον καθαρισμό:")
print(df.isnull().sum())

# Κωδικοποίηση όλων των στήλων τύπου string σε αριθμούς
def encode_all_string_columns(df):
    mappings = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].unique()
            val_to_id = {val: i for i, val in enumerate(unique_vals)}
            df[col + "_encoded"] = df[col].map(val_to_id)
            mappings[col] = val_to_id
    return df, mappings

df, encoding_maps = encode_all_string_columns(df)

# Διαχείριση ελλιπών τιμών: αριθμητικά με τη διάμεσο, κατηγορικά με "Unknown"
def handle_missing(df):
    for col in df.columns:
        if df[col].isnull().all():
            continue

        if np.issubdtype(df[col].dtype, np.number):
            if df[col].isnull().any():
                df[col + "_missing"] = df[col].isnull().astype(int)
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        elif df[col].dtype == "object":
            if df[col].isnull().any():
                df[col] = df[col].fillna("Unknown")
    return df

df = handle_missing(df)

# Έλεγχος ελλιπών τιμών μετά τον καθαρισμό
print("Απουσίες μετά τον καθαρισμό:")
print(df.isnull().sum())

# Αποθήκευση καθαρών δεδομένων για το επόμενο βήμα
df.to_csv("cleaned_atp_matches.csv", index=False)
print("Καθαρά δεδομένα αποθηκεύτηκαν στο cleaned_atp_matches.csv")
