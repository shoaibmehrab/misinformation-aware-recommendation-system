import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict
import pickle
from lenskit import topn
import os
import torch
import torch.nn as nn
import argparse

# Assuming these custom modules are in the same directory or in the python path
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator

# --- 1. Configuration and File Paths ---
PROCESSED_FILE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/FH/processed-files"
DATA_FILE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/FH/data"
RESULTS_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/FH/results"
BASE_PICKLE_PATH = os.path.join(PROCESSED_FILE_DIR, "TrustWorthy_FH_MAGrec.pickle")
ITEM_MAP_PICKLE_PATH = os.path.join(PROCESSED_FILE_DIR, "FH_item_update.pickle")
KEY_FILE_PATH = os.path.join(DATA_FILE_DIR, "key_healthstory.csv")
SEEDS_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/FH/seeds"


# --- 2. Data Preparation Functions ---
def file_load(path):
    with open(path, "r") as my_file:
        data = my_file.read()
    data_into_list = data.split("\n")[:-1]
    return [int(item) for item in data_into_list]

def calculate_cosine_similarity(user1, user2, user_item_sets):
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
    Calculates user reputation and prepares trustworthy neighbor data for the FakeHealth dataset.
    """
    print(f"--- Starting Data Preparation for reputation threshold: {threshold} ---")
    
    # Load and process training data
    train = pd.read_csv(os.path.join(DATA_FILE_DIR, "train_ratings.csv")).iloc[:, [1, 2, 3]]
    train.drop_duplicates(subset=['user', 'item'], inplace=True)
    
    # --- New Reputation Calculation Logic ---
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
    
    # Set reputation as index for easy lookup
    interaction_counts.set_index('user', inplace=True)

    untrusted_users = list(interaction_counts[interaction_counts['reputation'] <= threshold].index)
    
    untrusted_pickle_path = os.path.join(PROCESSED_FILE_DIR, f'untrusted_users_{threshold}_threshold.pickle')
    with open(untrusted_pickle_path, 'wb') as f:
        pickle.dump(untrusted_users, f)
    print(f"Saved {len(untrusted_users)} untrusted users to {untrusted_pickle_path}")
    # --- End of New Reputation Logic ---

    # Calculate cosine similarity matrix
    print("Calculating cosine similarity...")
    user_item_sets = train.groupby('user')['item'].apply(set).to_dict()
    unique_users = train['user'].unique()
    cosine_similarities = [
        {'User1': u1, 'User2': u2, 'Similarity': calculate_cosine_similarity(u1, u2, user_item_sets)}
        for u1, u2 in product(unique_users, repeat=2) if u1 != u2
    ]
    cosine_df = pd.DataFrame(cosine_similarities)
    cosine_matrix = cosine_df.pivot(index='User1', columns='User2', values='Similarity')
    idx = range(len(cosine_matrix))
    cosine_matrix.values[idx, idx] = 1.0

    # Identify trustworthy neighbors using the new reputation scores
    print(f"Identifying trustworthy neighbors (reputation > {threshold})...")
    top_trustworthy_similar_users = defaultdict(set)

    # Align interaction_counts with cosine_matrix indices
    reputations_aligned = interaction_counts.reindex(cosine_matrix.index, fill_value=0)['reputation']

    for user in cosine_matrix.index:
        user_similarity = cosine_matrix.loc[user]
        
        # Get reputations of all other users
        other_users_reputation = reputations_aligned[user_similarity.index]
        
        # Filter for trustworthy users (reputation > threshold)
        trustworthy_mask = other_users_reputation > threshold
        
        # Exclude the user itself and users with zero similarity
        self_mask = user_similarity.index != user
        similarity_mask = user_similarity > 0
        
        # Combine masks
        final_mask = trustworthy_mask & self_mask & similarity_mask
        
        similar_trustworthy_users = user_similarity[final_mask]
        
        top_trustworthy_similar_users[user] = set(similar_trustworthy_users.sort_values(ascending=False).index[:10].tolist())

    # Load original data and replace social_adj_lists
    print("Updating social adjacency lists in main pickle file...")
    with open(BASE_PICKLE_PATH, 'rb') as file:
        loaded_data = pickle.load(file)
    
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, _, ratings_list = loaded_data
    
    data_to_save = (history_u_lists, history_ur_lists, history_v_lists, history_vr_lists,
                    train_u, train_v, trai  n_r, test_u, test_v, test_r, top_trustworthy_similar_users, ratings_list)

    final_pickle_path = os.path.join(PROCESSED_FILE_DIR, f"FH_Trustworthy_Neighbors_{threshold}_threshold.pickle")
    with open(final_pickle_path, 'wb') as file:
        pickle.dump(data_to_save, file)
    print(f"Data preparation complete. Final data saved to: {final_pickle_path}")
    return final_pickle_path


# --- 3. Model Definition and Training ---
class GraphRec(nn.Module):
    # ... (GraphRec class remains unchanged)
    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim
        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)
        scores_u = torch.matmul(embeds_u, embeds_v.t())
        return scores_u.squeeze()

    def bpr_loss(self, nodes_u, nodes_v, device, num_items):
        scores_u = self.forward(nodes_u, nodes_v)
        nodes_v_negative = torch.randint(0, num_items, size=nodes_u.size(), dtype=torch.long).to(device)
        scores_u_negative = self.forward(nodes_u, nodes_v_negative)
        return -torch.log(torch.sigmoid(scores_u - scores_u_negative)).mean()

def model_train_step(model, device, train_loader, optimizer, epoch, num_items):
    # ... (model_train_step function remains unchanged)
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, _ = data
        optimizer.zero_grad()
        loss = model.bpr_loss(batch_nodes_u.to(device), batch_nodes_v.to(device), device, num_items)
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print(f'[{epoch}, {i:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

def get_recommendations(model, device, test_loader, k=100):
    # ... (get_recommendations function remains unchanged)
    model.eval()
    user_recommendations_dict = {}
    with torch.no_grad():
        for test_u, test_v, _ in test_loader:
            test_u, test_v = test_u.to(device), test_v.to(device)
            scores = model.forward(test_u, test_v).squeeze().tolist()
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            user_ids = test_u.tolist()
            for user_id in user_ids:
                user_recommendations = [(rank, test_v[item_index].item()) for rank, item_index in enumerate(top_indices, start=1)]
                user_recommendations_dict[user_id] = user_recommendations
    return user_recommendations_dict

def train_and_recommend(args, final_pickle_path):
    """
    Handles model training and recommendation generation for the FH dataset.
    """
    print(f"\n--- Starting Model Training for threshold: {args.threshold} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(final_pickle_path, 'rb') as data_file:
        history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, _ = pickle.load(data_file)

    total_num_items = len(set(train_v + test_v))
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v), torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v), torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    
    num_users = len(history_u_lists)
    num_items = len(history_v_lists)

    u2e = nn.Embedding(num_users, args.embed_dim).to(device)
    v2e = nn.Embedding(num_items, args.embed_dim).to(device)
    r2e = nn.Embedding(total_num_items, args.embed_dim).to(device)

    agg_u_history = UV_Aggregator(v2e, r2e, u2e, args.embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, args.embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, args.embed_dim, cuda=device)
    enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), args.embed_dim, social_adj_lists, agg_u_social, base_model=enc_u_history, cuda=device)
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, args.embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, args.embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

    graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    for epoch in range(1, args.epochs + 1):
        model_train_step(graphrec, device, train_loader, optimizer, epoch, total_num_items)

    print("Training complete. Generating recommendations...")
    recommendations = get_recommendations(graphrec, device, test_loader, k=100)
    
    print("Processing recommendations...")
    with open(ITEM_MAP_PICKLE_PATH, 'rb') as file:
        item_map = pickle.load(file)
    
    # Create a reverse map for efficient lookup
    value_to_key_map = {v: k for k, v in item_map.items()}
    
    rows = [{"user": user, "rank": rank, "item_update": item, "item": value_to_key_map.get(item)} for user, rankings in recommendations.items() for rank, item in rankings]
    all_recs = pd.DataFrame(rows)
    
    cold_start_items = [2, 571]
    filtered_recs = all_recs[~all_recs['item'].isin(cold_start_items)].copy()
    filtered_recs['re_rank'] = filtered_recs.groupby('user')['rank'].rank(method='first')
    filtered_recs['rank'] = filtered_recs['re_rank'].astype(int)
    final_recs = filtered_recs[['user', 'rank', 'item_update', 'item']]
    
    # Save top-k recommendations to CSV files
    top_k_values = [5, 10, 15]
    for k in top_k_values:
        recs_to_save = final_recs[(final_recs['rank'] <= k) & (final_recs['user'] != 5406)]
        filename = f'{DATA_FILE_DIR}/FH_Trustworthy_Neighbor_{args.threshold}_threshold_top_{k}.csv'
        recs_to_save[['user', 'rank', 'item']].to_csv(filename, index=False)
        print(f"✓ Saved {len(recs_to_save)} recommendations to {filename}")


# --- 4. Evaluation Functions ---
def mmc(dataframe, top_n):
    # ... (mmc function remains unchanged)
    dataframe = dataframe[dataframe['rank'] <= top_n]
    sorted_df = dataframe.sort_values(by=['user', 'rank'])
    misinformation_df = sorted_df[sorted_df['label'] == 'fake']
    user_mc = misinformation_df.groupby('user').size().reset_index(name='M_item')
    if not user_mc.empty:
        user_mc['MC'] = user_mc['M_item'] / top_n
        return user_mc['MC'].mean()
    return 0.0

def evaluate_model(threshold):
    """
    Calculates and saves MRR and MC metrics for the FH dataset.
    """
    print(f"\n--- Starting Evaluation for threshold: {threshold} ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    test = pd.read_csv(f"{DATA_FILE_DIR}/test_ratings_final_before_train.csv").iloc[:, [1, 2, 3]]
    if 5406 in test.index:
        test.drop(index=5406, inplace=True)

    top_k_values = [5, 10, 15]
    
    # MRR Calculation
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.recip_rank)
    mrr_results = {}
    for k in top_k_values:
        recs_filename = f"{DATA_FILE_DIR}/FH_Trustworthy_Neighbor_{threshold}_threshold_top_{k}.csv"
        recs_df = pd.read_csv(recs_filename)
        results = rla.compute(recs_df, test)
        mrr_results[k] = results.mean()['recip_rank']
    
    mrr_summary = pd.DataFrame(list(mrr_results.items()), columns=['Top-K', 'MRR'])
    mrr_summary_path = os.path.join(RESULTS_DIR, f'fh_trust_neighbour_mrr_summary_{threshold}_threshold.csv')
    mrr_summary.to_csv(mrr_summary_path, index=False)
    print(f"MRR Results:\n{mrr_summary}")
    print(f"✓ MRR summary saved to {mrr_summary_path}")

    # MC Calculation
    key = pd.read_csv(KEY_FILE_PATH)
    key_news_id = key[["item", "label"]].drop_duplicates(subset=["item"], keep="last")
    mc_results = {}
    for k in top_k_values:
        recs_filename = f"{DATA_FILE_DIR}/FH_Trustworthy_Neighbor_{threshold}_threshold_top_{k}.csv"
        recs_df = pd.read_csv(recs_filename)
        final_recs_mc = pd.merge(recs_df, key_news_id, on="item", how='inner')
        mc_results[k] = mmc(final_recs_mc, k)
        
    mc_summary = pd.DataFrame(list(mc_results.items()), columns=['Top-K', 'MC'])
    mc_summary_path = os.path.join(RESULTS_DIR, f'fh_trust_neighbour_mc_summary_{threshold}_threshold.csv')
    mc_summary.to_csv(mc_summary_path, index=False)
    print(f"\nMC Results:\n{mc_summary}")
    print(f"✓ MC summary saved to {mc_summary_path}")


def create_seed_files(threshold):
    """
    Creates seed files based on the generated recommendations for a given threshold.
    Each seed file corresponds to a fake news item and contains the user IDs
    who were recommended that item.
    """
    print(f"\n--- Creating Seed Files for threshold: {threshold} ---")
    top_k_values = [5, 10, 15]

    # Load test data and key file to identify all fake news items
    test = pd.read_csv(f"{DATA_FILE_DIR}/test_ratings_final_before_train.csv").iloc[:, [1, 2, 3]]
    if 5406 in test.index:
        test.drop(index=5406, inplace=True)
    key = pd.read_csv(KEY_FILE_PATH)
    key_news_id = key[["item", "label"]].drop_duplicates(subset=["item"], keep="last")
    key_news_id["fake_real_labels"] = key_news_id["label"].apply(lambda x: 0 if x == "fake" else 1)
    
    test['label'] = test['item'].map(key_news_id.set_index('item')['fake_real_labels'])
    all_fake_items_in_test = set(test[test['label'] == 0]['item'])

    for k in top_k_values:
        # Define the directory for this k-value
        seed_dir = os.path.join(SEEDS_DIR, 'trustworthy_neighbor', f'threshold_{threshold}', f'top_{k}')
        os.makedirs(seed_dir, exist_ok=True)
        print(f"Created directory for seeds: {seed_dir}")

        # Load the recommendation file for this k-value
        recs_filename = f"{DATA_FILE_DIR}/FH_Trustworthy_Neighbor_{threshold}_threshold_top_{k}.csv"
        if not os.path.exists(recs_filename):
            print(f"Warning: Recommendation file not found, skipping seed creation for k={k}. Path: {recs_filename}")
            continue
        
        recs_df = pd.read_csv(recs_filename)

        # Find which fake news items were actually recommended
        recommended_fake_items = set(recs_df['item']).intersection(all_fake_items_in_test)
        print(f"Found {len(recommended_fake_items)} recommended fake news items for k={k}.")

        # Create a seed file for each recommended fake news item
        for item_id in recommended_fake_items:
            # Find all users who were recommended this item
            seed_users = recs_df[recs_df['item'] == item_id]['user'].unique().tolist()
            
            if seed_users:
                seed_file_path = os.path.join(seed_dir, f"{item_id}GraphRec.txt")
                with open(seed_file_path, "w") as f:
                    for user_id in seed_users:
                        f.write(f"{user_id}\n")
        
        print(f"Finished creating seed files for k={k}.")

# --- 5. Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(description='Trustworthy Neighbor model for FakeHealth dataset.')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--threshold', type=float, default=0.6, help='Reputation threshold for identifying trustworthy users.')
    parser.add_argument('--skip_data_prep', action='store_true', help='Skip the data preparation step if files already exist.')
    args = parser.parse_args()

    final_pickle_path = os.path.join(PROCESSED_FILE_DIR, f"FH_Trustworthy_Neighbors_{args.threshold}_threshold.pickle")

    if not args.skip_data_prep:
        final_pickle_path = prepare_trust_data(args.threshold)
        if final_pickle_path is None:
            return # Stop execution if data prep failed
    else:
        print(f"--- Skipping Data Preparation, using existing file: {final_pickle_path} ---")
        if not os.path.exists(final_pickle_path):
            print(f"Error: Pickle file not found for threshold {args.threshold}. Please run without --skip_data_prep first.")
            return

    train_and_recommend(args, final_pickle_path)
    evaluate_model(args.threshold)

    create_seed_files(args.threshold)
    
    print("\n--- Script finished successfully! ---")

if __name__ == "__main__":
    main()
