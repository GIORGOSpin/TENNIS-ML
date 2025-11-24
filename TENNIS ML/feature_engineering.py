import pandas as pd
import numpy as np
import json

def label_encode_series(s):
    """Simple label encoder for a pandas Series."""
    s = s.fillna("Unknown").astype(str)
    vals = s.unique()
    mapping = {v: i for i, v in enumerate(vals)}
    return s.map(mapping), mapping


def build_two_vector_features(
    input_csv='cleaned_atp_matches.csv',
    output_csv='features_two_vector.csv',
    encodings_out='feature_encodings.json',
    include_swapped=True
):
    print("Loading cleaned data...")
    df = pd.read_csv(input_csv)

    # Only keep **pre-match** features (no leakage)
    prematch_p1 = [
        'winner_id', 'winner_seed', 'winner_entry',
        'winner_name', 'winner_hand', 'winner_ht',
        'winner_ioc', 'winner_age', 'winner_rank', 'winner_rank_points'
    ]

    prematch_p2 = [
        'loser_id', 'loser_seed', 'loser_entry',
        'loser_name', 'loser_hand', 'loser_ht',
        'loser_ioc', 'loser_age', 'loser_rank', 'loser_rank_points'
    ]

    match_fields = [
        'surface', 'tourney_level', 'tourney_date', 'round',
        'tourney_name', 'tourney_id', 'draw_size', 'best_of'
    ]

    # Remove fields that do NOT exist in this dataset
    prematch_p1 = [c for c in prematch_p1 if c in df.columns]
    prematch_p2 = [c for c in prematch_p2 if c in df.columns]
    match_fields = [c for c in match_fields if c in df.columns]

    # Fill simple missing values
    df.fillna({
        'winner_entry': "Unknown",
        'loser_entry': "Unknown",
        'winner_hand': "Unknown",
        'loser_hand': "Unknown",
        'surface': "Unknown",
        'tourney_level': "Unknown",
        'round': "Unknown"
    }, inplace=True)

    # Encode categorical match-level fields
    onehot_cols = ['surface', 'tourney_level']
    df = pd.get_dummies(df, columns=onehot_cols, prefix=onehot_cols)

    # Encode remaining large categories
    encodings = {}
    for c in ['round', 'tourney_name', 'tourney_id']:
        if c in df.columns:
            df[c + '_enc'], mapping = label_encode_series(df[c])
            encodings[c] = mapping

    rows = []

    print("Building two-vector rows...")
    for _, row in df.iterrows():
        p1 = {}
        p2 = {}

        # Map winner → p1
        for c in prematch_p1:
            key = "p1_" + c.replace("winner_", "")
            p1[key] = row[c]

        # Map loser → p2
        for c in prematch_p2:
            key = "p2_" + c.replace("loser_", "")
            p2[key] = row[c]

        # Match-level fields
        m = {}
        for c in df.columns:
            if c.startswith('surface_') or c.startswith('tourney_level_'):
                m[c] = row[c]
        for c in ['round_enc', 'tourney_name_enc', 'tourney_id_enc', 'draw_size', 'best_of', 'tourney_date']:
            if c in df.columns:
                m[c] = row[c]

        # Primary row (winner = p1 → target=1)
        combined = {**p1, **p2, **m, 'target': 1}
        rows.append(combined)

        # Swapped row (loser becomes p1 → target=0)
        if include_swapped:
            sw1 = {("p1_" + k[3:]): v for k, v in p2.items()}
            sw2 = {("p2_" + k[3:]): v for k, v in p1.items()}
            swapped = {**sw1, **sw2, **m, 'target': 0}
            rows.append(swapped)

    final_df = pd.DataFrame(rows)

    # Encode player names at the end
    print("Encoding player names...")
    names = pd.concat([
        final_df['p1_name'].astype(str),
        final_df['p2_name'].astype(str)
    ], ignore_index=True)

    name_encoded, name_map = label_encode_series(names)

    # Split back correctly
    half = len(name_encoded) // 2
    final_df['p1_name_encoded'] = name_encoded[:half].values
    final_df['p2_name_encoded'] = name_encoded[half:].values
    encodings['player_name'] = name_map

    # Drop name strings
    final_df.drop(columns=['p1_name', 'p2_name'], inplace=True)

    # Save
    print(f"Saving output to {output_csv}...")
    final_df.to_csv(output_csv, index=False)

    with open(encodings_out, 'w', encoding='utf-8') as f:
        json.dump(encodings, f, ensure_ascii=False, indent=2)

    print("DONE.")
    print("Generated rows:", len(final_df))


if __name__ == "__main__":
    build_two_vector_features()
