import pandas as pd
import os
import argparse
import re

# --- 1. Configuration and File Paths ---
BASE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact"
SEEDS_DIR = os.path.join(BASE_DIR, "seeds")
DATA_DIR = os.path.join(BASE_DIR, "data")
KEY_FILE_PATH = os.path.join(DATA_DIR, "politifact_Shu_fake_news_keyforSOCIALMF.csv")

def load_fake_news_ids():
    """Loads the set of fake news item IDs from the key file."""
    try:
        key_df = pd.read_csv(KEY_FILE_PATH)
        fake_news_items = set(key_df[key_df['news_label'] == 'fake']['news_id_new'])
        print(f"Successfully loaded {len(fake_news_items)} unique fake news item IDs from the key file.")
        return fake_news_items
    except Exception as e:
        print(f"FATAL: Could not load or process key file at {KEY_FILE_PATH}. Error: {e}")
        return None

def combined_analysis(experiment_type, threshold, fake_news_ids):
    """
    Analyzes the impact of fake news recommendations by inspecting the generated seed files.
    """
    print(f"\n--- Starting Combined Analysis for Experiment: '{experiment_type}' with Threshold: {threshold} ---")

    top_k_values = [5, 10, 15]

    for k in top_k_values:
        print(f"\n--- Analysis for Top-{k} ---")
        
        # 1. Dynamically construct the path to the seed directory
        seed_dir_path = os.path.join(SEEDS_DIR, experiment_type, f'threshold_{threshold}', f'top_{k}')

        if not os.path.isdir(seed_dir_path):
            print(f"  - Seed directory not found, skipping: {seed_dir_path}")
            continue

        # 2. Find all potential seed files
        try:
            all_files = [f for f in os.listdir(seed_dir_path) if f.endswith('.txt')]
        except FileNotFoundError:
            print(f"  - Seed directory not found, skipping: {seed_dir_path}")
            continue

        if not all_files:
            print("  - No seed files found in this directory.")
            continue
            
        # 3. Analyze each file, verifying it corresponds to a known fake news item
        seed_set_sizes = {}
        all_seeded_users = set()
        
        for filename in all_files:
            match = re.match(r'(\d+)', filename)
            if not match:
                continue
            item_id = int(match.group(1))

            # This is the key step: connect the seed file to the fake news list
            if item_id in fake_news_ids:
                try:
                    with open(os.path.join(seed_dir_path, filename), 'r') as f:
                        users = {int(line.strip()) for line in f if line.strip()}
                        seed_set_sizes[item_id] = len(users)
                        all_seeded_users.update(users)
                except (IOError, ValueError) as e:
                    print(f"  - Warning: Could not read or parse file {filename}. Error: {e}")

        # 4. Report the findings for this k-value
        if not seed_set_sizes:
            print("  - No seed files corresponding to known fake news items were found.")
            continue

        print(f"  - Found {len(seed_set_sizes)} seed files corresponding to known fake news items.")
        total_unique_users_seeded = len(all_seeded_users)
        print(f"  - Total unique users seeded for diffusion: {total_unique_users_seeded}")

        print("\n  Impact Analysis (Users Seeded per Fake News Item):")
        sorted_seeds = sorted(seed_set_sizes.items(), key=lambda item: item[1], reverse=True)
        
        for item_id, user_count in sorted_seeds:
            print(f"    - Item {item_id} (FAKE): Seeded to {user_count} unique users.")
        
        # Check for a dominant item that could explain the high spread
        dominant_item_count = sorted_seeds[0][1]
        if total_unique_users_seeded > 0 and (dominant_item_count / total_unique_users_seeded) > 0.8:
            dominant_item_id = sorted_seeds[0][0]
            print(f"\n  >> ALERT: Item {dominant_item_id} is the 'lottery winner' for k={k}, seeding {dominant_item_count} of the {total_unique_users_seeded} total users! <<")

def main():
    parser = argparse.ArgumentParser(description='Perform a combined analysis of fake news exposure and seed file impact.')
    parser.add_argument('--experiment_type', type=str, required=True, help="Name of the experiment (e.g., 'trustworthy_social_intersect_neighbor').")
    parser.add_argument('--threshold', type=float, required=True, help='Reputation threshold used for the experiment (e.g., 0.5).')
    args = parser.parse_args()

    # Load the master list of fake news items first
    fake_news_ids = load_fake_news_ids()
    if fake_news_ids is None:
        return # Stop if we can't load the fake news list

    combined_analysis(args.experiment_type, args.threshold, fake_news_ids)

if __name__ == "__main__":
    main()
# ```

# ### How to Run and Interpret the Results

# Run the script from your `src` directory as before:
# ```bash
# python analyze-recommendation.py --experiment_type trustworthy_social_intersect_neighbor --threshold 0.5
# ```

# **The output will now give you the definitive answer.**

# *   For `top-5`, you should see a list under "Impact Analysis" where one or two fake news items have a very high user count next to them. The "ALERT" message is highly likely to appear here, confirming that a single item is responsible for the vast majority of the initial infections.
# *   For `top-10` and `top-15`, you will see a different distribution. There might be more fake news items listed, but the user counts for each will be much lower and more evenly distributed.

