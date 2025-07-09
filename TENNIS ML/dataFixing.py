import pandas as pd
import numpy as np

df = pd.read_csv("atp_matches_2024.csv")

'''Check original missing counts before'''
print("Missing counts BEFORE:")
print(df.isnull().sum())

'''make every string into a number'''
def encode_all_string_columns(df):
    mappings = {}
    
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].unique()
            val_to_id = {val: i for i, val in enumerate(unique_vals)}
            df[col + "_encoded"] = df[col].map(val_to_id)
            mappings[col] = val_to_id  # Save the mapping if you want to reverse it later
    '''            
    print(df.columns)
    print(df[["tourney_name", "tourney_name_encoded"]].head(3000))
    print(df[["surface", "surface_encoded"]].head(3000))
    print(df["tourney_name"].unique())
    '''

    return df, mappings

df, encoding_maps = encode_all_string_columns(df)


'''fix missing data problems'''
'''fill numeric columns with median'''
'''fill categorical columns with 'unknown' '''
def handle_missing(df):
    import numpy as np

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

''' Check if new missing indicator columns are created for numeric columns
for col in df.columns:
    if col.endswith('_missing'):
        print(f"Checking missing indicator column: {col}")
        print(df[col].value_counts())'''

'''Verify missing values are filled in numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    print(f"{col} missing values count: {df[col].isnull().sum()}")
'''

'''Verify missing values in categorical (object) columns are replaced with "Unknown"
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    print(f"{col} unique values:")
    print(df[col].unique())
    print(f"Missing values count in {col}: {df[col].isnull().sum()}")'''

'''Check original missing counts after'''
print("Missing counts AFTER:")
print(df.isnull().sum())

