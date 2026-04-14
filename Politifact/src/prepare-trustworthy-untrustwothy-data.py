import numpy as np
import pandas as pd
import os
import pickle
from itertools import product
from collections import defaultdict
import argparse

# --- 1. Configuration and File Paths ---
PROCESSED_FILE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/processed-files"
DATA_FILE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/data"
KEY_FILE_PATH = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/data/politifact_Shu_fake_news_keyforSOCIALMF.csv"
ORIGINAL_PICKLE_PATH = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/processed-files/TrustWorthy_Politifact_MAGrec.pickle"

# --- 2. Data Preparation Functions ---
def calculate_cosine_similarity(user1, user2, user_item_sets):
    """Calculates cosine similarity between two users based on item interactions."""
    items_user1 = user_item_sets.get(user1, set())
    items_user2 = user_item_sets.get(user2, set())
    intersection = len(items_user1.intersection(items_user2))
    magnitude_user1 = len(items_user1)
    magnitude_user2 = len(items_user2)
    if magnitude_user1 == 0 or magnitude_user2 == 0:
        return 0.0
    return intersection / (magnitude_user1 * magnitude_user2) ** 0.5

def prepare_trust_data(threshold):
    """
    Performs all data preparation steps based on a dynamic threshold.
    Saves the main data pickle file and the user segment lists.
    """
    print(f"--- Starting Data Preparation for threshold: {threshold} ---")
    os.makedirs(PROCESSED_FILE_DIR, exist_ok=True)
    
    # Load and process training data
    train = pd.read_csv(f"{DATA_FILE_DIR}/train_ratings.csv").iloc[:, [1, 2, 3]]
    train.drop_duplicates(subset=['user', 'item'], inplace=True)
    user_item_sets = train.groupby('user')['item'].apply(set).to_dict()
    unique_users = train['user'].unique()

    # Calculate cosine similarity matrix
    print("Calculating cosine similarity...")
    cosine_similarities = [
        {'User1': u1, 'User2': u2, 'Similarity': calculate_cosine_similarity(u1, u2, user_item_sets)}
        for u1, u2 in product(unique_users, repeat=2) if u1 != u2
    ]
    cosine_df = pd.DataFrame(cosine_similarities)
    cosine_matrix = cosine_df.pivot(index='User1', columns='User2', values='Similarity')
    idx = range(len(cosine_matrix))
    cosine_matrix.values[idx, idx] = 1.0

    # Calculate user reputation
    print("Calculating user reputation...")
    key = pd.read_csv(KEY_FILE_PATH)
    key_news_id = key[["news_id_new", "news_label"]].drop_duplicates(subset=["news_id_new"], keep="last").reset_index(drop=True)
    key_news_id["fake_real_labels"]=key_news_id["news_label"].apply(lambda x:0 if x=="fake" else 1)
    train_labeled = pd.merge(train, key_news_id, left_on="item", right_on="news_id_new", how='inner')
    
    interaction_counts = train_labeled.groupby(['user', 'news_label']).size().unstack(fill_value=0)
    interaction_counts.rename(columns={'fake': 'fake_news_count', 'real': 'real_news_count'}, inplace=True)
    interaction_counts.reset_index(inplace=True)
    interaction_counts['reputation'] = interaction_counts['real_news_count'] / (interaction_counts['real_news_count'] + interaction_counts['fake_news_count'])
    
    # Identify and save trustworthy/untrustworthy user lists
    print(f"Identifying user segments with reputation threshold: {threshold}...")
    trustworthy_users = interaction_counts[interaction_counts['reputation'] > threshold]['user'].tolist()
    untrustworthy_users = interaction_counts[interaction_counts['reputation'] <= threshold]['user'].tolist()

    with open(os.path.join(PROCESSED_FILE_DIR, f'trustworthy_users_{threshold}_threshold.pickle'), 'wb') as f:
        pickle.dump(trustworthy_users, f)
    print(f"✓ Saved {len(trustworthy_users)} trustworthy users.")
    
    with open(os.path.join(PROCESSED_FILE_DIR, f'untrustworthy_users_{threshold}_threshold.pickle'), 'wb') as f:
        pickle.dump(untrustworthy_users, f)
    print(f"✓ Saved {len(untrustworthy_users)} untrustworthy users.")

    # # Identify trustworthy neighbors for the model
    # print(f"Identifying trustworthy neighbors with reputation > {threshold}...")
    # top_trustworthy_similar_users = defaultdict(set)
    # interaction_counts.set_index('user', inplace=True)
    # all_users_in_matrix = cosine_matrix.index
    # interaction_counts = interaction_counts.reindex(all_users_in_matrix, fill_value=0)

    # for user in cosine_matrix.index:
    #     user_similarity = cosine_matrix.loc[user]
    #     similar_trustworthy_users = user_similarity[
    #         (user_similarity > 0) &
    #         (interaction_counts.loc[user_similarity.index, 'reputation'] > threshold) &
    #         (user_similarity.index != user)
    #     ]
    #     top_trustworthy_similar_users[user] = set(similar_trustworthy_users.sort_values(ascending=False).index[:10].tolist())

    # # Load original data and replace social_adj_lists
    # print("Updating social adjacency lists in main pickle file...")
    # with open(ORIGINAL_PICKLE_PATH, 'rb') as file:
    #     loaded_data = pickle.load(file)
    
    # history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, _, ratings_list = loaded_data
    
    # data_to_save = (history_u_lists, history_ur_lists, history_v_lists, history_vr_lists,
    #                 train_u, train_v, train_r, test_u, test_v, test_r, top_trustworthy_similar_users, ratings_list)

    # # Use threshold in the final pickle filename
    # final_pickle_path = os.path.join(PROCESSED_FILE_DIR, f"D1_Trust_Neighbor_Update_{threshold}_threshold.pickle")
    # with open(final_pickle_path, 'wb') as file:
    #     pickle.dump(data_to_save, file)
        
    # print(f"\n--- Data preparation complete. ---")
    # print(f"✓ Final data saved to: {final_pickle_path}")


# --- 3. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preparation for GraphRec model.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Reputation threshold for identifying trustworthy users.')
    args = parser.parse_args()
    prepare_trust_data(args.threshold)
