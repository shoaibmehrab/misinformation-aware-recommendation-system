import pandas as pd
import os
import pickle
import argparse
from lenskit import topn

# --- 1. Configuration for FakeHealth (FH) Dataset ---
BASE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/FH"
PROCESSED_FILE_DIR = os.path.join(BASE_DIR, "processed-files")
DATA_FILE_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# File paths specific to the FH dataset
KEY_FILE_PATH = os.path.join(DATA_FILE_DIR, "key_healthstory.csv")
TEST_FILE_PATH = os.path.join(DATA_FILE_DIR, "test_ratings_final_before_train.csv")


# --- 2. Helper Function (adapted for FH) ---
def mmc(dataframe, top_n):
    """Calculates Mean Misinformation Consumption."""
    dataframe = dataframe[dataframe['rank'] <= top_n]
    sorted_df = dataframe.sort_values(by=['user', 'rank'])
    # Use 'label' column for FH dataset
    misinformation_df = sorted_df[sorted_df['label'] == 'fake']
    user_mc = misinformation_df.groupby('user').size().reset_index(name='M_item')
    if not user_mc.empty:
        user_mc['MC'] = user_mc['M_item'] / top_n
        return user_mc['MC'].mean()
    return 0.0

# --- 3. Main Evaluation Logic ---
def evaluate_segments(threshold, experiment_type):
    """
    Calculates and saves MRR and MC metrics for trustworthy and untrustworthy user segments for the FH dataset.
    """
    print(f"--- Starting Segmented Evaluation for FH Dataset ---")
    print(f"Experiment: '{experiment_type}', Threshold: {threshold}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load user segments from the FH processed-files directory
    try:
        with open(os.path.join(PROCESSED_FILE_DIR, f'trustworthy_users_{threshold}_threshold.pickle'), 'rb') as f:
            trustworthy_users = pickle.load(f)
        with open(os.path.join(PROCESSED_FILE_DIR, f'untrustworthy_users_{threshold}_threshold.pickle'), 'rb') as f:
            untrustworthy_users = pickle.load(f)
        print(f"Loaded {len(trustworthy_users)} trustworthy and {len(untrustworthy_users)} untrustworthy users.")
    except FileNotFoundError:
        print(f"Error: User segment files for threshold {threshold} not found in {PROCESSED_FILE_DIR}.")
        print("Please ensure the 'prepare-data-FH.py' script has been run first.")
        return

    # Load test data and news key
    if not os.path.exists(TEST_FILE_PATH) or not os.path.exists(KEY_FILE_PATH):
        print(f"Error: Test file or Key file not found. Please check paths.")
        return
        
    test = pd.read_csv(TEST_FILE_PATH).iloc[:, [1, 2, 3]]
    if 5406 in test.index:
        test.drop(index=5406, inplace=True)
        
    key = pd.read_csv(KEY_FILE_PATH)
    # Use 'item' and 'label' columns for FH dataset
    key_news_id = key[["item", "label"]].drop_duplicates(subset=["item"], keep="last")

    # --- Count fake news in test set ---
    test_labeled = pd.merge(test, key_news_id, on="item", how='inner')
    fake_news_count = test_labeled[test_labeled['label'] == 'fake']['item'].nunique()
    print(f"Found {fake_news_count} unique fake news items in the test dataset.")

    top_k_values = [5, 10, 15]
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.recip_rank)

    results = []

    for k in top_k_values:
        print(f"\nProcessing Top-{k} recommendations...")
        recs_filename = f"{DATA_FILE_DIR}/FH_{experiment_type}_{threshold}_threshold_top_{k}.csv"
        try:
            recs_df = pd.read_csv(recs_filename)
        except FileNotFoundError:
            print(f"  Warning: Recommendation file not found, skipping for k={k}. Path: {recs_filename}")
            continue

        # Filter recommendations for each segment
        recs_trustworthy = recs_df[recs_df['user'].isin(trustworthy_users)]
        recs_untrustworthy = recs_df[recs_df['user'].isin(untrustworthy_users)]
        print(f"  Found {len(recs_trustworthy)} recommendations for trustworthy users.")
        print(f"  Found {len(recs_untrustworthy)} recommendations for untrustworthy users.")

        # --- Trustworthy Segment ---
        mrr_trustworthy = rla.compute(recs_trustworthy, test).mean()['recip_rank']
        mc_df_trustworthy = pd.merge(recs_trustworthy, key_news_id, on="item", how='inner')
        mc_trustworthy = mmc(mc_df_trustworthy, k)
        results.append({'Top-K': k, 'Segment': 'Trustworthy', 'MRR': mrr_trustworthy, 'MC': mc_trustworthy})

        # --- Untrustworthy Segment ---
        mrr_untrustworthy = rla.compute(recs_untrustworthy, test).mean()['recip_rank']
        mc_df_untrustworthy = pd.merge(recs_untrustworthy, key_news_id, on="item", how='inner')
        mc_untrustworthy = mmc(mc_df_untrustworthy, k)
        results.append({'Top-K': k, 'Segment': 'Untrustworthy', 'MRR': mrr_untrustworthy, 'MC': mc_untrustworthy})

    # Save combined results
    if not results:
        print("No results were generated. Exiting.")
        return
        
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(RESULTS_DIR, f'fh_segmented_metrics_summary_{experiment_type}_{threshold}_threshold.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print("\n--- Segmented Evaluation Complete ---")
    print(f"Results:\n{summary_df}")
    print(f"\n✓ Combined summary saved to {summary_path}")


# --- 4. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate segmented metrics for the FakeHealth (FH) dataset.')
    parser.add_argument('--experiment_type', type=str, required=True, help='A name for the experiment being evaluated (e.g., Trustworthy_Neighbor).')
    parser.add_argument('--threshold', type=float, default=0.5, help='Reputation threshold used during training.')
    args = parser.parse_args()

    evaluate_segments(args.threshold, args.experiment_type)
