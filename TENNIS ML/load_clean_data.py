import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv("atp_matches_2024.csv")

print("Missing values before cleaning:")
print(df.isnull().sum())

# Handle missing data smartly
def handle_missing(df):
    for col in df.columns:
        if df[col].isnull().all():
            continue
        
        if np.issubdtype(df[col].dtype, np.number):
            if df[col].isnull().any():
                # Add missing indicator column
                df[col + "_missing"] = df[col].isnull().astype(int)
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Filled numeric '{col}' missing with median: {median_val}")
        else:
            if df[col].isnull().any():
                df[col].fillna("Unknown", inplace=True)
                print(f"Filled categorical '{col}' missing with 'Unknown'")
    return df

df = handle_missing(df)

# Now encode string columns after missing values are filled
def encode_all_string_columns(df):
    mappings = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].unique()
            val_to_id = {val: i for i, val in enumerate(unique_vals)}
            df[col + "_encoded"] = df[col].map(val_to_id)
            mappings[col] = val_to_id
            print(f"Encoded column '{col}' with {len(val_to_id)} unique values.")
    return df, mappings

df, encoding_maps = encode_all_string_columns(df)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Save cleaned and encoded dataset
df.to_csv("cleaned_atp_matches.csv", index=False)
print("\nCleaned data saved to cleaned_atp_matches.csv")
