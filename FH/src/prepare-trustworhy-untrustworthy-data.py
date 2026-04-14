import numpy as np
import pandas as pd
import os
import pickle
import argparse

# --- 1. Configuration for FakeHealth (FH) Dataset ---
BASE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/FH" # Base path for FH
PROCESSED_FILE_DIR = os.path.join(BASE_DIR, "processed-files")
DATA_FILE_DIR = os.path.join(BASE_DIR, "data")

# IMPORTANT: Please verify these filenames match your actual FH data files
KEY_FILE_PATH = os.path.join(DATA_FILE_DIR, "key_healthstory.csv")
TRAIN_FILE_PATH = os.path.join(DATA_FILE_DIR, "train_ratings.csv")


# --- 2. Data Preparation Function ---

def prepare_trust_data(threshold):
    """
    Calculates user reputation from FH training data and saves lists of
    trustworthy and untrustworthy users.
    """
    print(f"--- Starting Data Preparation for FH Dataset (Threshold: {threshold}) ---")
    os.makedirs(PROCESSED_FILE_DIR, exist_ok=True)
    
    # Load and process training data
    if not os.path.exists(TRAIN_FILE_PATH) or not os.path.exists(KEY_FILE_PATH):
        print(f"Error: Training file or Key file not found. Please check paths.")
        print(f"Looked for: {TRAIN_FILE_PATH}")
        print(f"Looked for: {KEY_FILE_PATH}")
        return

    train = pd.read_csv(TRAIN_FILE_PATH).iloc[:, [1, 2, 3]]
    train.drop_duplicates(subset=['user', 'item'], inplace=True)

    # --- Reputation Calculation Logic for FH Dataset ---
    print("Calculating user reputations...")
    key = pd.read_csv(KEY_FILE_PATH)
    key_news_id = key[["item", "label"]].drop_duplicates(subset=["item"], keep="last").reset_index(drop=True)
    key_news_id["fake_real_labels"] = key_news_id["label"].apply(lambda x: 0 if x == "fake" else 1)
    
    train_with_labels = pd.merge(train, key_news_id, on="item", how='inner')
    
    interaction_counts = train_with_labels.groupby(['user', 'label']).size().unstack(fill_value=0)
    if 'fake' not in interaction_counts.columns:
        interaction_counts['fake'] = 0
    if 'real' not in interaction_counts.columns:
        interaction_counts['real'] = 0
        
    interaction_counts.rename(columns={'fake': 'fake_news_count', 'real': 'real_news_count'}, inplace=True)
    interaction_counts = interaction_counts.reset_index()
    
    interaction_counts['reputation'] = interaction_counts['real_news_count'] / (interaction_counts['real_news_count'] + interaction_counts['fake_news_count'])
    interaction_counts.fillna(0, inplace=True) # Handle division by zero if a user has no interactions
    # --- End of Reputation Logic ---
    
    # Identify and save trustworthy/untrustworthy user lists
    print(f"Identifying user segments with reputation threshold: {threshold}...")
    trustworthy_users = interaction_counts[interaction_counts['reputation'] > threshold]['user'].tolist()
    untrustworthy_users = interaction_counts[interaction_counts['reputation'] <= threshold]['user'].tolist()

    with open(os.path.join(PROCESSED_FILE_DIR, f'trustworthy_users_{threshold}_threshold.pickle'), 'wb') as f:
        pickle.dump(trustworthy_users, f)
    print(f"✓ Saved {len(trustworthy_users)} trustworthy users to {PROCESSED_FILE_DIR}")
    
    with open(os.path.join(PROCESSED_FILE_DIR, f'untrustworthy_users_{threshold}_threshold.pickle'), 'wb') as f:
        pickle.dump(untrustworthy_users, f)
    print(f"✓ Saved {len(untrustworthy_users)} untrustworthy users to {PROCESSED_FILE_DIR}")
        
    print(f"\n--- FH Data preparation complete. ---")


# --- 3. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preparation for FakeHealth (FH) dataset.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Reputation threshold for identifying trustworthy users.')
    args = parser.parse_args()
    prepare_trust_data(args.threshold)
