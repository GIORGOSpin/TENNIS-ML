

import pandas as pd
import numpy as np
import json

def label_encode_series(s):
    """Απλό label encoder για μία pandas Series. Επιστρέφει series_encoded, mapping dict."""
    vals = s.fillna("Unknown").astype(str).unique()
    mapping = {v: i for i, v in enumerate(vals)}
    return s.fillna("Unknown").astype(str).map(mapping), mapping

def build_two_vector_features(
    input_csv='cleaned_atp_matches.csv',
    output_csv='features_two_vector.csv',
    encodings_out='feature_encodings.json',
    include_swapped=True
):
    """
    Διαβάζει cleaned_atp_matches.csv και παράγει features_two_vector.csv
    include_swapped: αν True, για κάθε αγώνα θα δημιουργηθεί και η αντίστροφη σειρά (p1=loser) με target=0
    Επίσης αποθηκεύει τα mappings (label encodings) σε JSON για μελλοντική χρήση.
    """

    df = pd.read_csv(input_csv)
    df = df.copy()  # εργαστούμε σε αντίγραφο

    # --- 1) Βασικός καθαρισμός / συμπλήρωση αν χρειάζεται ---
    df.fillna({"winner_hand": "Unknown", "loser_hand": "Unknown", "surface": "Unknown", "tourney_level": "Unknown"}, inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if df[c].isnull().any():
            df[c].fillna(df[c].median(), inplace=True)

    # --- 2) One-hot για μικρές κατηγορίες (surface, tourney_level) ---
    categorical_onehot = ['surface', 'tourney_level']
    for col in categorical_onehot:
        if col in df.columns:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col)
            df = pd.concat([df, dummies], axis=1)

    # --- 3) Binary / one-hot για χέρι κάθε παίκτη ---
    df['winner_hand'] = df['winner_hand'].fillna("Unknown").astype(str)
    df['loser_hand'] = df['loser_hand'].fillna("Unknown").astype(str)
    hands = sorted(list(set(df['winner_hand'].unique().tolist() + df['loser_hand'].unique().tolist())))
    for h in hands:
        df[f'p1_hand_{h}'] = (df['winner_hand'] == h).astype(int)
        df[f'p2_hand_{h}'] = (df['loser_hand'] == h).astype(int)

    # --- 4) Label-encode για μεγάλες κατηγορίες που δεν θέλουμε one-hot ---
    encodings = {}
    for col in ['tourney_name', 'round', 'tourney_id', 'winner_ioc', 'loser_ioc', 'winner_entry', 'loser_entry']:
        if col in df.columns:
            encoded_series, mapping = label_encode_series(df[col])
            df[col + '_encoded'] = encoded_series
            encodings[col] = mapping

    # --- 5) Συλλογή πεδίων για p1 και p2 ---
    p1_source_fields = [
        'winner_id','winner_seed','winner_entry','winner_name','winner_hand','winner_ht','winner_ioc',
        'winner_age','winner_rank','winner_rank_points',
        'w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon','w_SvGms','w_bpSaved','w_bpFaced'
    ]
    p2_source_fields = [
        'loser_id','loser_seed','loser_entry','loser_name','loser_hand','loser_ht','loser_ioc',
        'loser_age','loser_rank','loser_rank_points',
        'l_ace','l_df','l_svpt','l_1stIn','l_1stWon','l_2ndWon','l_SvGms','l_bpSaved','l_bpFaced'
    ]
    p1_source_fields = [c for c in p1_source_fields if c in df.columns]
    p2_source_fields = [c for c in p2_source_fields if c in df.columns]

    # --- 6) Match-level χαρακτηριστικά ---
    match_fields = ['tourney_date', 'draw_size', 'best_of', 'minutes', 'match_num', 'score']
    match_fields = [c for c in match_fields if c in df.columns]

    surface_cols = [c for c in df.columns if c.startswith('surface_')]
    tourney_level_cols = [c for c in df.columns if c.startswith('tourney_level_')]

    # --- 7) Δημιουργία rows TWO-VECTOR ---
    rows = []

    for _, row in df.iterrows():
        p1 = {}
        p2 = {}

        for c in p1_source_fields:
            key = 'p1_' + c.replace('winner_', '')
            if c + '_encoded' in df.columns:
                val = row.get(c + '_encoded', row.get(c, np.nan))
            else:
                val = row.get(c, np.nan)
            p1[key] = val

        for c in p2_source_fields:
            key = 'p2_' + c.replace('loser_', '')
            if c + '_encoded' in df.columns:
                val = row.get(c + '_encoded', row.get(c, np.nan))
            else:
                val = row.get(c, np.nan)
            p2[key] = val

        match_feats = {}
        for c in surface_cols + tourney_level_cols:
            match_feats[c] = row.get(c, 0)
        for enc_col in ['round_encoded', 'tourney_name_encoded', 'tourney_id_encoded']:
            if enc_col in df.columns:
                match_feats[enc_col] = row.get(enc_col, np.nan)
        for c in match_fields:
            match_feats[c] = row.get(c, np.nan)

        combined = {}
        combined.update(p1)
        combined.update(p2)
        combined.update(match_feats)
        combined['target'] = 1  # p1 (winner) κέρδισε

        rows.append(combined)

        if include_swapped:
            swapped = {}
            for k, v in p2.items():
                swapped['p1_' + k[len('p2_'):]] = v
            for k, v in p1.items():
                swapped['p2_' + k[len('p1_'):]] = v
            swapped.update(match_feats)
            swapped['target'] = 0
            rows.append(swapped)

    final_df = pd.DataFrame(rows)

    # --- 7.5) Label encode για p1_name και p2_name ---
    final_df['p1_name_encoded'], mapping_p1_name = label_encode_series(final_df['p1_name'])
    final_df['p2_name_encoded'], mapping_p2_name = label_encode_series(final_df['p2_name'])
    encodings['p1_name'] = mapping_p1_name
    encodings['p2_name'] = mapping_p2_name

    # Αν θέλεις, αφαιρούμε τα string ονόματα (προτείνεται)
    final_df.drop(columns=['p1_name', 'p2_name'], inplace=True)

    # --- 8) Συμπλήρωση missing ---
    for c in final_df.columns:
        if final_df[c].dtype == object:
            final_df[c] = final_df[c].fillna("Unknown")
        else:
            if final_df[c].isnull().any():
                try:
                    med = final_df[c].median()
                    final_df[c] = final_df[c].fillna(med)
                except Exception:
                    final_df[c] = final_df[c].fillna(0)

    # --- 9) Αποθήκευση ---
    with open(encodings_out, 'w', encoding='utf-8') as f:
        json.dump(encodings, f, ensure_ascii=False, indent=2)

    final_df.to_csv(output_csv, index=False)
    print(f"Saved TWO-VECTOR features to: {output_csv}")
    print(f"Saved label-encodings (mappings) to: {encodings_out}")
    print(f"Rows produced: {len(final_df)} (include_swapped={include_swapped})")
    print("Columns in final dataset:", final_df.columns.tolist())

if __name__ == "__main__":
    build_two_vector_features()



