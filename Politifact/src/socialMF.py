import numpy as np
import pandas as pd
from collections import defaultdict
import os
import argparse
from tqdm import tqdm
from lenskit import topn

# --- 1. Configuration and File Paths for Politifact ---
PROCESSED_FILE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/processed-files"
DATA_FILE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/data"
RESULTS_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/results"
KEY_FILE_PATH = os.path.join(DATA_FILE_DIR, "politifact_Shu_fake_news_keyforSOCIALMF.csv")
SEEDS_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/seeds"
TRAIN_FILE_PATH = os.path.join(DATA_FILE_DIR, "train_ratings.csv")
TEST_FILE_PATH = os.path.join(DATA_FILE_DIR, "politifact_test.csv")
# Assuming the social network is stored in 'network.txt' as a tab-separated file
NETWORK_FILE_PATH = os.path.join(DATA_FILE_DIR, "networkN.txt")


# --- 2. SocialMF Model Implementation ---
class SocialMF:
    def __init__(self, num_users, num_items, embed_dim=64, social_reg=0.1, lr=0.001, reg=0.01):
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.social_reg = social_reg
        self.lr = lr
        self.reg = reg

        # Initialize user and item latent factor matrices
        self.P = np.random.normal(0, 0.1, (num_users, embed_dim))
        self.Q = np.random.normal(0, 0.1, (num_items, embed_dim))

    def train(self, train_data, social_graph, epochs=100):
        print("--- Starting SocialMF Training ---")
        for epoch in range(epochs):
            total_loss = 0
            # Shuffle training data for each epoch
            shuffled_data = train_data.sample(frac=1)
            
            for _, row in tqdm(shuffled_data.iterrows(), total=len(shuffled_data), desc=f"Epoch {epoch+1}/{epochs}"):
                u, i, r = int(row['user']), int(row['item']), float(row['ratings'])
                
                # Prediction and error
                pred = np.dot(self.P[u], self.Q[i])
                error = r - pred
                
                # Social influence term
                social_influence = np.zeros(self.embed_dim)
                if u in social_graph and social_graph[u]:
                    trusted_friends = social_graph[u]
                    # Average influence of trusted friends' factors
                    social_influence = np.mean(self.P[trusted_friends], axis=0)

                # Update user and item factors
                p_u_old = self.P[u].copy()
                q_i_old = self.Q[i].copy()
                
                self.P[u] += self.lr * (error * q_i_old - self.reg * p_u_old - self.social_reg * (p_u_old - social_influence))
                self.Q[i] += self.lr * (error * p_u_old - self.reg * q_i_old)
                
                total_loss += error ** 2

            print(f"Epoch {epoch+1}/{epochs}, Loss: {np.sqrt(total_loss / len(shuffled_data)):.4f}")
        print("--- Training Complete ---")

    def predict_all(self):
        """Predict ratings for all user-item pairs."""
        return np.dot(self.P, self.Q.T)

# --- 3. Recommendation Generation ---
def generate_socialmf_recommendations(args):
    """
    Trains SocialMF and generates recommendations.
    """
    # Load data
    try:
        train_df = pd.read_csv(TRAIN_FILE_PATH)
        test_df = pd.read_csv(TEST_FILE_PATH)
        
        # Load social network
        social_graph = defaultdict(list)
        with open(NETWORK_FILE_PATH, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    u1, u2 = int(parts[0]), int(parts[1])
                    social_graph[u1].append(u2)
                    social_graph[u2].append(u1) # Assuming undirected graph

    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}")
        return

    # Get number of unique users and items
    num_users = int(max(train_df['user'].max(), test_df['user'].max())) + 1
    num_items = int(max(train_df['item'].max(), test_df['item'].max())) + 1
    print(f"Dataset info: {num_users} users, {num_items} items.")

    # Initialize and train the model
    model = SocialMF(num_users, num_items, embed_dim=args.embed_dim, social_reg=args.social_reg, lr=args.lr, reg=args.reg)
    model.train(train_df, social_graph, epochs=args.epochs)

    # Generate scores for all users and items
    all_scores = model.predict_all()

    # Generate recommendations
    user_history = train_df.groupby('user')['item'].apply(set).to_dict()
    test_users = test_df['user'].unique()
    
    all_recs = []
    for user in tqdm(test_users, desc="Generating recommendations"):
        if user >= num_users: continue # Skip users not in training
        
        scores = all_scores[user]
        seen_items = user_history.get(user, set())
        
        # Create a list of (item, score) tuples for unseen items
        unseen_item_scores = [(i, scores[i]) for i in range(num_items) if i not in seen_items]
        
        # Sort by score and get top recommendations
        unseen_item_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (item, score) in enumerate(unseen_item_scores[:max(args.top_k)], 1):
            all_recs.append({'user': user, 'rank': rank, 'item': item})

    all_recs_df = pd.DataFrame(all_recs)

    # Save recommendations to files
    print("\nSaving recommendation files...")
    for k in args.top_k:
        recs_to_save = all_recs_df[all_recs_df['rank'] <= k]
        filename = f'{DATA_FILE_DIR}/Politifact_SocialMF_top_{k}.csv'
        recs_to_save[['user', 'rank', 'item']].to_csv(filename, index=False)
        print(f"✓ Saved {len(recs_to_save)} recommendations to {filename}")


# --- 4. Evaluation and Seed File Functions (Identical to previous scripts) ---
def mmc(dataframe, top_n):
    dataframe = dataframe[dataframe['rank'] <= top_n]
    sorted_df = dataframe.sort_values(by=['user', 'rank'])
    misinformation_df = sorted_df[sorted_df['news_label'] == 'fake']
    user_mc = misinformation_df.groupby('user').size().reset_index(name='M_item')
    if not user_mc.empty:
        user_mc['MC'] = user_mc['M_item'] / top_n
        return user_mc['MC'].mean()
    return 0.0

def evaluate_model(args):
    print("\n--- Starting Evaluation ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    test = pd.read_csv(TEST_FILE_PATH).iloc[:, [1, 2, 3]]
    key = pd.read_csv(KEY_FILE_PATH)
    test.drop(index=test[test['user'] == 1028].index, inplace=True, errors='ignore')

    rla = topn.RecListAnalysis()
    rla.add_metric(topn.recip_rank)
    mrr_results = {}
    for k in args.top_k:
        recs_filename = f"{DATA_FILE_DIR}/Politifact_SocialMF_top_{k}.csv"
        recs_df = pd.read_csv(recs_filename)
        results = rla.compute(recs_df, test)
        mrr_results[k] = results.mean()['recip_rank']
    
    mrr_summary = pd.DataFrame(list(mrr_results.items()), columns=['Top-K', 'MRR'])
    mrr_summary_path = os.path.join(RESULTS_DIR, 'politifact_socialmf_mrr_summary.csv')
    mrr_summary.to_csv(mrr_summary_path, index=False)
    print(f"MRR Results:\n{mrr_summary}")

    key_news_id = key[["news_id_new", "news_label"]].drop_duplicates(subset=["news_id_new"], keep="last")
    mc_results = {}
    for k in args.top_k:
        recs_filename = f"{DATA_FILE_DIR}/Politifact_SocialMF_top_{k}.csv"
        recs_df = pd.read_csv(recs_filename)
        final_recs_mc = pd.merge(recs_df, key_news_id, left_on="item", right_on="news_id_new", how='inner')
        mc_results[k] = mmc(final_recs_mc, k)
        
    mc_summary = pd.DataFrame(list(mc_results.items()), columns=['Top-K', 'MC'])
    mc_summary_path = os.path.join(RESULTS_DIR, 'politifact_socialmf_mc_summary.csv')
    mc_summary.to_csv(mc_summary_path, index=False)
    print(f"\nMC Results:\n{mc_summary}")

def create_seed_files(args):
    print("\n--- Creating Seed Files ---")
    test = pd.read_csv(TEST_FILE_PATH).iloc[:, [1, 2, 3]]
    key = pd.read_csv(KEY_FILE_PATH)
    test.drop(index=test[test['user'] == 1028].index, inplace=True, errors='ignore')
    key_news_id = key[["news_id_new", "news_label"]].drop_duplicates(subset=["news_id_new"], keep="last")
    key_news_id["fake_real_labels"] = key_news_id["news_label"].apply(lambda x: 0 if x == "fake" else 1)
    test['label'] = test['item'].map(key_news_id.set_index('news_id_new')['fake_real_labels'])
    all_fake_items_in_test = set(test[test['label'] == 0]['item'])

    for k in args.top_k:
        seed_dir = os.path.join(SEEDS_DIR, 'social_mf', f'top_{k}')
        os.makedirs(seed_dir, exist_ok=True)
        print(f"Created directory for seeds: {seed_dir}")
        recs_filename = f"{DATA_FILE_DIR}/Politifact_SocialMF_top_{k}.csv"
        if not os.path.exists(recs_filename): continue
        recs_df = pd.read_csv(recs_filename)
        recommended_fake_items = set(recs_df['item']).intersection(all_fake_items_in_test)
        print(f"Found {len(recommended_fake_items)} recommended fake news items for k={k}.")
        for item_id in recommended_fake_items:
            seed_users = recs_df[recs_df['item'] == item_id]['user'].unique().tolist()
            if seed_users:
                seed_file_path = os.path.join(seed_dir, f"{item_id}GraphRec.txt")
                with open(seed_file_path, "w") as f:
                    for user_id in seed_users: f.write(f"{user_id}\n")
        print(f"Finished creating seed files for k={k}.")

# --- 5. Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(description='SocialMF Recommender for Politifact dataset.')
    parser.add_argument('--top_k', nargs='+', type=int, default=[5, 10, 15], help='List of top-k values to evaluate.')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension for latent factors.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for SGD.')
    parser.add_argument('--reg', type=float, default=0.01, help='Regularization parameter for user/item factors.')
    parser.add_argument('--social_reg', type=float, default=0.1, help='Social regularization parameter.')
    args = parser.parse_args()

    generate_socialmf_recommendations(args)
    evaluate_model(args)
    create_seed_files(args)
    
    print("\n--- Script finished successfully! ---")

if __name__ == "__main__":
    main()
# ```

# ### How to Run

# 1.  Save the code above as `social_mf_politifact.py` in your `/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/src/` directory.
# 2.  Make sure you have a `network.txt` file in your `/data` directory with the social connections.
# 3.  Open your terminal, navigate to the `src` directory, and run the script:

#     ```bash
#     python social_mf_politifact.py --epochs 50
#     ```

# You can adjust the hyperparameters as needed:
# *   `--epochs`: Number of training iterations.
# *   `--lr`: Learning rate.
# *   `--reg`: Standard regularization for factors.
# *   `--social_reg`: Controls the strength of the social influence.

# This will execute the full pipeline, creating recommendation files (`Politifact_SocialMF_...`), evaluation summaries (`politifact_socialmf_...`), and seed files in a new `social_mf` subdirectory.# filepath: /home/shoaib/recommender-system/royal/journal-revision/New/Politifact/src/social_mf_politifact.py
