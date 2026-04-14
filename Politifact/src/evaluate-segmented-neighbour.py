import pandas as pd
import os
import pickle
import argparse
from lenskit import topn

# --- 1. Configuration and File Paths ---
PROCESSED_FILE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/processed-files"
DATA_FILE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/data"
RESULTS_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/results"
KEY_FILE_PATH = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/data/politifact_Shu_fake_news_keyforSOCIALMF.csv"

# --- 2. Helper Function ---
def mmc(dataframe, top_n):
    """Calculates Mean Misinformation Consumption."""
    dataframe = dataframe[dataframe['rank'] <= top_n]
    sorted_df = dataframe.sort_values(by=['user', 'rank'])
    misinformation_df = sorted_df[sorted_df['news_label'] == 'fake']
    user_mc = misinformation_df.groupby('user').size().reset_index(name='M_item')
    if not user_mc.empty:
        user_mc['MC'] = user_mc['M_item'] / top_n
        return user_mc['MC'].mean()
    return 0.0

# --- 3. Main Evaluation Logic ---
def evaluate_segments(threshold):
    """
    Calculates and saves MRR and MC metrics for trustworthy and untrustworthy user segments.
    """
    print(f"--- Starting Segmented Evaluation for threshold: {threshold} ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load user segments
    try:
        with open(os.path.join(PROCESSED_FILE_DIR, f'trustworthy_users_{threshold}_threshold.pickle'), 'rb') as f:
            trustworthy_users = pickle.load(f)
        with open(os.path.join(PROCESSED_FILE_DIR, f'untrustworthy_users_{threshold}_threshold.pickle'), 'rb') as f:
            untrustworthy_users = pickle.load(f)
        print(f"Loaded {len(trustworthy_users)} trustworthy and {len(untrustworthy_users)} untrustworthy users.")
    except FileNotFoundError:
        print(f"Error: User segment files for threshold {threshold} not found.")
        print("Please run the main 'trustworthy-neighbour.py' script first to generate them.")
        return

    # Load test data and news key
    test = pd.read_csv(f"{DATA_FILE_DIR}/politifact_test.csv").iloc[:, [1, 2, 3]]
    test.drop(index=test[test['user'] == 1028].index, inplace=True, errors='ignore')
    key = pd.read_csv(KEY_FILE_PATH)
    key_news_id = key[["news_id_new", "news_label"]].drop_duplicates(subset=["news_id_new"], keep="last")

    # --- Count fake news in test set ---
    test_labeled = pd.merge(test, key_news_id, left_on="item", right_on="news_id_new", how='inner')
    fake_news_count = test_labeled[test_labeled['news_label'] == 'fake']['item'].nunique()
    print(f"Found {fake_news_count} unique fake news items in the test dataset.")

    top_k_values = [5, 10, 15]
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.recip_rank)

    results = []

    for k in top_k_values:
        print(f"\nProcessing Top-{k} recommendations...")
        recs_filename = f"{DATA_FILE_DIR}/Politifact_Trustworthy_Social_Union_Neighbor_{threshold}_threshold_top_{k}.csv"
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
        mc_df_trustworthy = pd.merge(recs_trustworthy, key_news_id, left_on="item", right_on="news_id_new", how='inner')
        mc_trustworthy = mmc(mc_df_trustworthy, k)
        results.append({'Top-K': k, 'Segment': 'Trustworthy', 'MRR': mrr_trustworthy, 'MC': mc_trustworthy})

        # --- Untrustworthy Segment ---
        mrr_untrustworthy = rla.compute(recs_untrustworthy, test).mean()['recip_rank']
        mc_df_untrustworthy = pd.merge(recs_untrustworthy, key_news_id, left_on="item", right_on="news_id_new", how='inner')
        mc_untrustworthy = mmc(mc_df_untrustworthy, k)
        results.append({'Top-K': k, 'Segment': 'Untrustworthy', 'MRR': mrr_untrustworthy, 'MC': mc_untrustworthy})

    # Save combined results
    if not results:
        print("No results were generated. Exiting.")
        return
        
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(RESULTS_DIR, f'segmented_metrics_summary_trustwothy_Social_union_neighbor_{threshold}_threshold.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print("\n--- Segmented Evaluation Complete ---")
    print(f"Results:\n{summary_df}")
    print(f"\n✓ Combined summary saved to {summary_path}")


# --- 4. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate segmented metrics for the GraphRec model.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Reputation threshold used during training.')
    args = parser.parse_args()

    evaluate_segments(args.threshold)
