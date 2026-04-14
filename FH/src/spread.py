import numpy as np
import pandas as pd
import os
import pickle
import random
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import argparse
from tqdm import tqdm

# --- 1. Configuration and File Paths ---
PROCESSED_FILE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/FH/processed-files"
DATA_FILE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/FH/data"
RESULTS_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/FH/results"
SEEDS_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/FH/seeds"
NETWORK_FILE_PATH = os.path.join(DATA_FILE_DIR, "networkN.txt")
SIMILARITY_FILE_PATH = os.path.join(DATA_FILE_DIR, "fakehealth_news_similarity.csv")
TRAIN_FILE_PATH = os.path.join(DATA_FILE_DIR, "train_ratings.csv")
TEST_FILE_PATH = os.path.join(DATA_FILE_DIR, "test_ratings_final_before_train.csv")

# --- 2. Helper Functions ---

def network_load(path):
    """Loads the network from a tab-separated file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Network file not found at: {path}")
    nodes = []
    edges = []
    with open(path) as file:
        for line in file:
            source, target = line.strip().split('\t')
            nodes.extend([int(source), int(target)])
            edges.append((int(source), int(target)))
    return list(set(nodes)), edges

def make_graph(nodes, edges):
    """Creates a networkx graph."""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def seed_load(path):
    """Loads a single seed file."""
    with open(path, "r") as my_file:
        data = my_file.read()
    data_into_list = data.strip().split("\n")[:-1]
    return [int(item) for item in data_into_list if item]

# --- 3. Diffusion Model Implementations ---

def linear_threshold_model(G, seeds, num_nodes):
    """Runs a single simulation of the Linear Threshold Model."""
    model = ep.ThresholdModel(G, seed=123)
    config = mc.Configuration()
    config.add_model_initial_configuration("Infected", seeds)
    
    for i in G.nodes():
        threshold = random.uniform(0, 1)
        config.add_node_configuration("threshold", i, threshold)
        
    model.set_initial_status(config)
    iterations = model.iteration_bunch(50) # 200 iterations for convergence
    
    # Get spread at iteration 30
    infected_at_30 = {key for key, value in iterations[30]["status"].items() if value == 1}
    spread_at_30 = len(infected_at_30) / 5406 * 100

    # Get final spread
    final_infected = {key for iter_result in iterations for key, value in iter_result["status"].items() if value == 1}
    final_spread = len(final_infected) / 5406 * 100
    
    return spread_at_30, final_spread

def independent_cascade_model(G, seeds, num_nodes):
    """Runs a single simulation of the Independent Cascades Model."""
    model = ep.IndependentCascadesModel(G)
    config = mc.Configuration()
    config.add_model_initial_configuration("Infected", seeds)
    
    for e in G.edges():
        config.add_edge_configuration("threshold", e, 1) # Using a common value like 0.1
        
    model.set_initial_status(config)
    iterations = model.iteration_bunch(50)
    
    # Get spread at iteration 30
    infected_at_30 = {key for key, value in iterations[30]["status"].items() if value == 1}
    spread_at_30 = len(infected_at_30) / 5406 * 100

    # Get final spread
    final_infected = {key for iter_result in iterations for key, value in iter_result["status"].items() if value == 1}
    final_spread = len(final_infected) / 5406 * 100

    return spread_at_30, final_spread

def node_profile_threshold_model(G, seeds, num_nodes, node_profiles):
    """Runs a single simulation of the Node Profile Threshold Model."""
    model = ep.ProfileThresholdModel(G, seed = 123)
    config = mc.Configuration()
    config.add_model_initial_configuration("Infected", seeds)
    
    config.add_model_parameter('blocked', 0)
    config.add_model_parameter('adopter_rate', 0)

    for i in G.nodes():
        threshold = random.uniform(0, 1)
        config.add_node_configuration("threshold", i, threshold)
        # Use pre-calculated profile values
        profile_val = node_profiles.get(i, 0)
        config.add_node_configuration("profile", i, profile_val)
        
    model.set_initial_status(config)
    iterations = model.iteration_bunch(50)
    
    # Get spread at iteration 30
    infected_at_30 = {key for key, value in iterations[30]["status"].items() if value == 1}
    spread_at_30 = len(infected_at_30) / 5406 * 100

    # Get final spread
    final_infected = {key for iter_result in iterations for key, value in iter_result["status"].items() if value == 1}
    final_spread = len(final_infected) / 5406 * 100

    return spread_at_30, final_spread

def prepare_node_profiles():
    """Pre-calculates node profile values for all users."""
    print("Preparing node profiles...")
    try:
        similarity = pd.read_csv(SIMILARITY_FILE_PATH, index_col=0)
        train = pd.read_csv(TRAIN_FILE_PATH).iloc[:, [1, 2, 3]]
        test = pd.read_csv(TEST_FILE_PATH).iloc[:, [1, 2, 3]]
    except FileNotFoundError as e:
        print(f"Error loading file for node profiles: {e}")
        return {}

    # Ensure index is correct for iloc
    similarity.index = np.arange(1, len(similarity) + 1)
    similarity.columns = np.arange(1, len(similarity.columns) + 1)

    test_items = test.set_index('user')['item'].to_dict()
    train_items = train.groupby('user')['item'].apply(list).to_dict()
    
    node_profiles = {}
    all_users = set(test_items.keys()) | set(train_items.keys())

    for user in all_users:
        test_item = test_items.get(user)
        user_train_items = train_items.get(user, [])
        
        if not test_item or not user_train_items:
            node_profiles[user] = 0.0
            continue
            
        possible_comb = [(test_item, train_item) for train_item in user_train_items]
        
        sim_values = []
        for x, y in possible_comb:
            if x in similarity.index and y in similarity.columns:
                sim_values.append(similarity.loc[x, y])
        
        if sim_values:
            node_profiles[user] = sum(sim_values) / len(sim_values)
        else:
            node_profiles[user] = 0.0
            
    print("Node profiles prepared.")
    return node_profiles

# --- 4. Main Execution Block ---

def main():
    parser = argparse.ArgumentParser(description='Calculate information spread using various diffusion models.')
    parser.add_argument('--experiment_type', type=str, required=True, help="Name of the experiment (e.g., 'trustworthy_neighbor').")
    parser.add_argument('--threshold', type=str, required=True, help='Reputation threshold used for the experiment.')
    parser.add_argument('--top_k', nargs='+', type=int, default=[5, 10, 15], help='List of top-k values to evaluate.')
    parser.add_argument('--simulations', type=int, default=100, help='Number of simulations to run per seed file.')
    args = parser.parse_args()

    # Load network graph
    nodes, edges = network_load(NETWORK_FILE_PATH)
    G = make_graph(nodes, edges)
    num_nodes = G.number_of_nodes()
    print(f"Network loaded: {num_nodes} nodes, {len(edges)} edges.")

    # Pre-calculate node profiles for the Node Profile Threshold Model
    node_profiles = prepare_node_profiles()

    all_results = []

    for k in args.top_k:
        print(f"\n--- Processing Top-{k} Recommendations ---")
        
        # Construct path to seed files
        seed_dir = os.path.join(SEEDS_DIR, args.experiment_type, f'threshold_{args.threshold}', f'top_{k}_Constant')
        if not os.path.isdir(seed_dir):
            print(f"Warning: Seed directory not found, skipping k={k}. Path: {seed_dir}")
            continue

        seed_files = [os.path.join(seed_dir, f) for f in os.listdir(seed_dir) if f.endswith('.txt')]
        if not seed_files:
            print(f"Warning: No seed files found in {seed_dir}, skipping k={k}.")
            continue
        
        print(f"Found {len(seed_files)} seed files for k={k}.")

        # Store average spread for each model for this k
        avg_spread_lt_30, avg_spread_lt_final = [], []
        avg_spread_icm_30, avg_spread_icm_final = [], []
        avg_spread_npt_30, avg_spread_npt_final = [], []

        for seed_path in tqdm(seed_files, desc=f"Simulating k={k}"):
            seeds = seed_load(seed_path)
            if not seeds:
                continue

            # --- Run simulations and collect results ---
            results_lt = [linear_threshold_model(G, seeds, num_nodes) for _ in range(args.simulations)]
            results_icm = [independent_cascade_model(G, seeds, num_nodes) for _ in range(args.simulations)]
            results_npt = [node_profile_threshold_model(G, seeds, num_nodes, node_profiles) for _ in range(args.simulations)]

            # --- Unzip results and average across simulations for this seed file ---
            # Linear Threshold
            sim_spreads_lt_30, sim_spreads_lt_final = zip(*results_lt)
            avg_spread_lt_30.append(np.mean(sim_spreads_lt_30))
            avg_spread_lt_final.append(np.mean(sim_spreads_lt_final))

            # Independent Cascade
            sim_spreads_icm_30, sim_spreads_icm_final = zip(*results_icm)
            avg_spread_icm_30.append(np.mean(sim_spreads_icm_30))
            avg_spread_icm_final.append(np.mean(sim_spreads_icm_final))

            # Node Profile Threshold
            sim_spreads_npt_30, sim_spreads_npt_final = zip(*results_npt)
            avg_spread_npt_30.append(np.mean(sim_spreads_npt_30))
            avg_spread_npt_final.append(np.mean(sim_spreads_npt_final))


        # --- Average across all seed files for this k ---
        e_spread_lt_30 = np.mean(avg_spread_lt_30) if avg_spread_lt_30 else 0
        e_spread_lt_final = np.mean(avg_spread_lt_final) if avg_spread_lt_final else 0
        
        e_spread_icm_30 = np.mean(avg_spread_icm_30) if avg_spread_icm_30 else 0
        e_spread_icm_final = np.mean(avg_spread_icm_final) if avg_spread_icm_final else 0

        e_spread_npt_30 = np.mean(avg_spread_npt_30) if avg_spread_npt_30 else 0
        e_spread_npt_final = np.mean(avg_spread_npt_final) if avg_spread_npt_final else 0

        
        print(f"Results for Top-{k}:")
        print(f"  E(Spread) - Linear Threshold (30 iter): {e_spread_lt_30:.4f}%, (Final): {e_spread_lt_final:.4f}%")
        print(f"  E(Spread) - Independent Cascade (30 iter): {e_spread_icm_30:.4f}%, (Final): {e_spread_icm_final:.4f}%")
        print(f"  E(Spread) - Node Profile Threshold (30 iter): {e_spread_npt_30:.4f}%, (Final): {e_spread_npt_final:.4f}%")

        all_results.append({
            'experiment': args.experiment_type,
            'threshold': args.threshold,
            'top_k': k,
            'spread_lt_30_iter': e_spread_lt_30,
            'spread_lt_final': e_spread_lt_final,
            'spread_icm_30_iter': e_spread_icm_30,
            'spread_icm_final': e_spread_icm_final,
            'spread_npt_30_iter': e_spread_npt_30,
            'spread_npt_final': e_spread_npt_final
        })

    # Save final results to a CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_path = os.path.join(RESULTS_DIR, f"spread_results_{args.experiment_type}_{args.threshold}_Constant.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\nSpread calculation complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
