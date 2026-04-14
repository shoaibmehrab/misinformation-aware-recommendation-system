import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
from lenskit import topn
import os
import argparse
from tqdm import tqdm

# --- 1. Configuration and File Paths for Politifact ---
PROCESSED_FILE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/processed-files"
DATA_FILE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/data"
RESULTS_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/results"
KEY_FILE_PATH = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/data/politifact_Shu_fake_news_keyforSOCIALMF.csv"
SEEDS_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/seeds"
TRAIN_FILE_PATH = os.path.join(DATA_FILE_DIR, "train_ratings.csv")
TEST_FILE_PATH = os.path.join(DATA_FILE_DIR, "politifact_test.csv")


# --- 2. Recommendation Generation ---
def generate_popularity_recommendations(args):
    """
    Generates and saves recommendations based on item popularity for Politifact.
    """
    print("--- Starting Popularity Recommender for Politifact ---")
    
    # Load data
    try:
        train_df = pd.read_csv(TRAIN_FILE_PATH)
        test_df = pd.read_csv(TEST_FILE_PATH)
    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}")
        return

    # 1. Calculate Popularity
    item_popularity = train_df['item'].value_counts().reset_index()
    item_popularity.columns = ['item', 'popularity']
    popular_items = item_popularity['item'].tolist()
    print(f"Calculated popularity for {len(popular_items)} unique items.")

    # 2. Prepare for recommendation generation
    user_history = train_df.groupby('user')['item'].apply(set).to_dict()
    test_users = test_df['user'].unique()
    print(f"Generating recommendations for {len(test_users)} users.")

    # 3. Generate recommendations for each user
    all_recs = []
    for user in tqdm(test_users, desc="Generating recommendations"):
        seen_items = user_history.get(user, set())
        recs = []
        rank = 1
        for item in popular_items:
            if item not in seen_items:
                recs.append({'user': user, 'rank': rank, 'item': item})
                rank += 1
            if rank > max(args.top_k):
                break
        all_recs.extend(recs)

    all_recs_df = pd.DataFrame(all_recs)

    # 4. Save recommendations to files for each k
    print("\nSaving recommendation files...")
    for k in args.top_k:
        recs_to_save = all_recs_df[all_recs_df['rank'] <= k]
        filename = f'{DATA_FILE_DIR}/Politifact_Popularity_top_{k}.csv'
        recs_to_save[['user', 'rank', 'item']].to_csv(filename, index=False)
        print(f"✓ Saved {len(recs_to_save)} recommendations to {filename}")


# --- 3. Evaluation Functions ---
def mmc(dataframe, top_n):
    """Calculates the Mean Misinformation Count (MC)."""
    dataframe = dataframe[dataframe['rank'] <= top_n]
    sorted_df = dataframe.sort_values(by=['user', 'rank'])
    misinformation_df = sorted_df[sorted_df['news_label'] == 'fake']
    user_mc = misinformation_df.groupby('user').size().reset_index(name='M_item')
    if not user_mc.empty:
        user_mc['MC'] = user_mc['M_item'] / top_n
        return user_mc['MC'].mean()
    return 0.0

def evaluate_model(args):
    """
    Calculates and saves MRR and MC metrics for the Politifact popularity model.
    """
    print("\n--- Starting Evaluation ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    try:
        test = pd.read_csv(TEST_FILE_PATH).iloc[:, [1, 2, 3]]
        key = pd.read_csv(KEY_FILE_PATH)
    except FileNotFoundError as e:
        print(f"Error: Evaluation file not found. {e}")
        return

    # Politifact-specific data cleaning
    test.drop(index=test[test['user'] == 1028].index, inplace=True, errors='ignore')

    # MRR Calculation
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.recip_rank)
    mrr_results = {}
    for k in args.top_k:
        recs_filename = f"{DATA_FILE_DIR}/Politifact_Popularity_top_{k}.csv"
        recs_df = pd.read_csv(recs_filename)
        results = rla.compute(recs_df, test)
        mrr_results[k] = results.mean()['recip_rank']
    
    mrr_summary = pd.DataFrame(list(mrr_results.items()), columns=['Top-K', 'MRR'])
    mrr_summary_path = os.path.join(RESULTS_DIR, 'politifact_popularity_mrr_summary.csv')
    mrr_summary.to_csv(mrr_summary_path, index=False)
    print(f"MRR Results:\n{mrr_summary}")
    print(f"✓ MRR summary saved to {mrr_summary_path}")

    # MC Calculation
    key_news_id = key[["news_id_new", "news_label"]].drop_duplicates(subset=["news_id_new"], keep="last")
    mc_results = {}
    for k in args.top_k:
        recs_filename = f"{DATA_FILE_DIR}/Politifact_Popularity_top_{k}.csv"
        recs_df = pd.read_csv(recs_filename)
        # Merge using the correct key column for Politifact
        final_recs_mc = pd.merge(recs_df, key_news_id, left_on="item", right_on="news_id_new", how='inner')
        mc_results[k] = mmc(final_recs_mc, k)
        
    mc_summary = pd.DataFrame(list(mc_results.items()), columns=['Top-K', 'MC'])
    mc_summary_path = os.path.join(RESULTS_DIR, 'politifact_popularity_mc_summary.csv')
    mc_summary.to_csv(mc_summary_path, index=False)
    print(f"\nMC Results:\n{mc_summary}")
    print(f"✓ MC summary saved to {mc_summary_path}")


# --- 4. Seed File Creation ---
def create_seed_files(args):
    """
    Creates seed files for the Politifact popularity model's recommendations.
    """
    print("\n--- Creating Seed Files ---")
    
    try:
        test = pd.read_csv(TEST_FILE_PATH).iloc[:, [1, 2, 3]]
        key = pd.read_csv(KEY_FILE_PATH)
    except FileNotFoundError as e:
        print(f"Error: Data file not found for seed creation. {e}")
        return

    # Politifact-specific data cleaning
    test.drop(index=test[test['user'] == 1028].index, inplace=True, errors='ignore')

    key_news_id = key[["news_id_new", "news_label"]].drop_duplicates(subset=["news_id_new"], keep="last")
    key_news_id["fake_real_labels"] = key_news_id["news_label"].apply(lambda x: 0 if x == "fake" else 1)
    
    test['label'] = test['item'].map(key_news_id.set_index('news_id_new')['fake_real_labels'])
    all_fake_items_in_test = set(test[test['label'] == 0]['item'])

    for k in args.top_k:
        seed_dir = os.path.join(SEEDS_DIR, 'popularity', f'threshold_{args.threshold}', f'top_{k}')
        os.makedirs(seed_dir, exist_ok=True)
        print(f"Created directory for seeds: {seed_dir}")

        recs_filename = f"{DATA_FILE_DIR}/Politifact_Popularity_top_{k}.csv"
        if not os.path.exists(recs_filename):
            print(f"Warning: Recommendation file not found, skipping seed creation for k={k}. Path: {recs_filename}")
            continue
        
        recs_df = pd.read_csv(recs_filename)
        recommended_fake_items = set(recs_df['item']).intersection(all_fake_items_in_test)
        print(f"Found {len(recommended_fake_items)} recommended fake news items for k={k}.")

        for item_id in recommended_fake_items:
            seed_users = recs_df[recs_df['item'] == item_id]['user'].unique().tolist()
            if seed_users:
                seed_file_path = os.path.join(seed_dir, f"{item_id}GraphRec.txt")
                with open(seed_file_path, "w") as f:
                    for user_id in seed_users:
                        f.write(f"{user_id}\n")
        
        print(f"Finished creating seed files for k={k}.")


# --- 5. Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(description='Popularity Recommender for Politifact dataset.')
    parser.add_argument('--top_k', nargs='+', type=int, default=[5, 10, 15], help='List of top-k values to evaluate.')
    args = parser.parse_args()

    generate_popularity_recommendations(args)
    evaluate_model(args)
    create_seed_files(args)
    
    print("\n--- Script finished successfully! ---")

if __name__ == "__main__":
    main()
# ```

# ### How to Run

# 1.  Save the code above as `popularity_recommender_politifact.py` in your `/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/src/` directory.
# 2.  Open your terminal, navigate to that directory, and run the script:

#     ```bash
#     python popularity_recommender_politifact.py
#     ```

# This will run the entire pipeline for the Politifact dataset, creating the recommendation files, evaluation summaries (MRR and MC), and the necessary seed files for your spread analysis, all with filenames and directory structures specific to this experiment.# filepath: /home/shoaib/recommender-system/royal/journal-revision/New/Politifact/src/popularity_recommender_politifact.py
