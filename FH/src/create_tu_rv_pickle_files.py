"""
Script to create T_u-only and R_v-only pickle files for ablation studies (FakeHealth dataset).

This script:
1. Loads the original base pickle file
2. Computes T_u (user trustworthiness) from train/test labels
3. Computes R_v (item engagement penalty) from user interaction counts
4. Normalizes both to 0.5-4.0 scale
5. Creates two new pickle files with updated rating indices
"""

import pickle
import pandas as pd
import numpy as np
import os
from collections import defaultdict

# File paths
PROCESSED_FILE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/FH/processed-files"
BASE_PICKLE_PATH = os.path.join(PROCESSED_FILE_DIR, "TrustWorthy_FH_MAGrec.pickle")
TRAINSET_PATH = "/home/shoaib/recommender-system/Ashita/data/FakeHealth/train_healthstory.csv"
TESTSET_PATH = "/home/shoaib/recommender-system/Ashita/data/FakeHealth/test_healthstory.csv"

OUTPUT_TU_PATH = os.path.join(PROCESSED_FILE_DIR, "TrustWorthy_FH_MAGrec_Tu_only.pickle")
OUTPUT_RV_PATH = os.path.join(PROCESSED_FILE_DIR, "TrustWorthy_FH_MAGrec_Rv_only.pickle")

# Rating bins (same as original)
RATING_BINS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
RATING_TO_INDEX = {0.5: 0, 1.0: 1, 1.5: 2, 2.0: 3, 2.5: 4, 3.0: 5, 3.5: 6, 4.0: 7}


def normalize_to_rating_scale(values, min_val=None, max_val=None):
    """
    Normalize values to 0.5-4.0 scale.
    
    Args:
        values: dict or array of values to normalize
        min_val: optional minimum value (calculated if None)
        max_val: optional maximum value (calculated if None)
    
    Returns:
        dict or array of normalized values in range [0.5, 4.0]
    """
    if isinstance(values, dict):
        val_array = np.array(list(values.values()))
    else:
        val_array = np.array(values)
    
    if min_val is None:
        min_val = val_array.min()
    if max_val is None:
        max_val = val_array.max()
    
    # Avoid division by zero
    if max_val == min_val:
        normalized = np.full_like(val_array, 2.25)  # Middle of range
    else:
        # Min-max normalization to [0, 1] then scale to [0.5, 4.0]
        normalized = 0.5 + ((val_array - min_val) / (max_val - min_val)) * 3.5
    
    if isinstance(values, dict):
        return {k: v for k, v in zip(values.keys(), normalized)}
    else:
        return normalized


def bin_to_discrete_rating(value):
    """
    Round value to nearest 0.5 and map to index 0-7.
    
    Args:
        value: float in range [0.5, 4.0]
    
    Returns:
        int index in range [0, 7]
    """
    # Round to nearest 0.5
    rounded = round(value * 2) / 2
    # Clamp to valid range
    rounded = max(0.5, min(4.0, rounded))
    return RATING_TO_INDEX[rounded]


def compute_user_trustworthiness(trainset, testset):
    """
    Compute T_u (trustworthiness) for each user.
    
    T_u = (real news interacted) / (total news interacted)
    
    Args:
        trainset: DataFrame with columns [user, item, label]
        testset: DataFrame with columns [user, item, label]
    
    Returns:
        dict: {user_id: T_u_value} in range [0, 1]
    """
    print("Computing user trustworthiness (T_u)...")
    
    # Combine train and test interactions
    all_interactions = pd.concat([
        trainset[['user', 'item', 'label']], 
        testset[['user', 'item', 'label']]
    ])
    
    # Calculate T_u for each user
    user_stats = all_interactions.groupby('user').agg({
        'label': [
            ('total', 'count'),
            ('real', lambda x: (x == 'real').sum())
        ]
    })
    user_stats.columns = ['total', 'real']
    user_stats['T_u'] = user_stats['real'] / user_stats['total']
    
    # Convert to dictionary
    T_u = user_stats['T_u'].to_dict()
    
    print(f"  Computed T_u for {len(T_u)} users")
    print(f"  T_u range: [{min(T_u.values()):.4f}, {max(T_u.values()):.4f}]")
    print(f"  Mean T_u: {np.mean(list(T_u.values())):.4f}")
    
    return T_u


def compute_item_engagement_penalty(trainset, testset, total_items):
    """
    Compute R_v (engagement penalty) for each item.
    
    R_v = ln(N / share_count)
    where N = total unique items, share_count = number of users who interacted with item
    
    Args:
        trainset: DataFrame with columns [user, item]
        testset: DataFrame with columns [user, item]
        total_items: int, total number of unique items (N)
    
    Returns:
        dict: {item_id: R_v_value}
    """
    print("Computing item engagement penalty (R_v)...")
    
    # Combine train and test interactions
    all_interactions = pd.concat([trainset[['user', 'item']], testset[['user', 'item']]])
    
    # Count unique users per item (share count)
    share_counts = all_interactions.groupby('item')['user'].nunique()
    
    # Compute R_v = ln(N / share_count)
    R_v = {}
    for item, share_count in share_counts.items():
        R_v[item] = np.log(total_items / share_count)
    
    print(f"  Computed R_v for {len(R_v)} items")
    print(f"  Share count range: [{share_counts.min()}, {share_counts.max()}]")
    print(f"  R_v range: [{min(R_v.values()):.4f}, {max(R_v.values()):.4f}]")
    print(f"  Mean R_v: {np.mean(list(R_v.values())):.4f}")
    
    return R_v


def create_tu_only_pickle(base_pickle_data, T_u_normalized, ratings_list):
    """
    Create T_u-only pickle file.
    
    In T_u-only version:
    - history_ur_lists: All ratings for a user become same T_u index (user property)
    - history_vr_lists: Ratings reflect different users' T_u values (user-specific)
    
    Args:
        base_pickle_data: tuple of original pickle data
        T_u_normalized: dict {user_id: normalized_T_u} in [0.5, 4.0]
        ratings_list: original ratings_list mapping
    
    Returns:
        tuple: new pickle data with updated rating indices
    """
    print("\n=== Creating T_u-only pickle file ===")
    
    (history_u_lists, history_ur_lists, history_v_lists, history_vr_lists,
     train_u, train_v, train_r, test_u, test_v, test_r, 
     social_adj_lists, _) = base_pickle_data
    
    # Convert T_u values to indices
    T_u_indices = {}
    for user_id, tu_value in T_u_normalized.items():
        T_u_indices[user_id] = bin_to_discrete_rating(tu_value)
    
    print(f"T_u index distribution:")
    index_counts = {}
    for idx in T_u_indices.values():
        index_counts[idx] = index_counts.get(idx, 0) + 1
    for idx in sorted(index_counts.keys()):
        print(f"  Index {idx}: {index_counts[idx]} users")
    
    # Update history_ur_lists: Each user's ratings become constant (their T_u)
    new_history_ur_lists = {}
    for user_id, item_list in history_u_lists.items():
        user_tu_index = T_u_indices.get(user_id, 0)
        # All items for this user get same T_u index
        new_history_ur_lists[user_id] = [user_tu_index] * len(item_list)
    
    # Update history_vr_lists: Each item's ratings show different users' T_u values
    new_history_vr_lists = {}
    for item_id, user_list in history_v_lists.items():
        # Get T_u index for each user who interacted with this item
        new_history_vr_lists[item_id] = [T_u_indices.get(u, 0) for u in user_list]
    
    # Keep train_r and test_r as original float values (unchanged)
    print(f"✓ Updated history_ur_lists for {len(new_history_ur_lists)} users")
    print(f"✓ Updated history_vr_lists for {len(new_history_vr_lists)} items")
    print(f"✓ Keeping train_r and test_r as original float values")
    
    return (history_u_lists, new_history_ur_lists, history_v_lists, new_history_vr_lists,
            train_u, train_v, train_r, test_u, test_v, test_r,
            social_adj_lists, ratings_list)


def create_rv_only_pickle(base_pickle_data, R_v_normalized, ratings_list):
    """
    Create R_v-only pickle file.
    
    In R_v-only version:
    - history_ur_lists: Ratings reflect different items' R_v values (item-specific)
    - history_vr_lists: All ratings for an item become same R_v index (item property)
    
    Args:
        base_pickle_data: tuple of original pickle data
        R_v_normalized: dict {item_id: normalized_R_v} in [0.5, 4.0]
        ratings_list: original ratings_list mapping
    
    Returns:
        tuple: new pickle data with updated rating indices
    """
    print("\n=== Creating R_v-only pickle file ===")
    
    (history_u_lists, history_ur_lists, history_v_lists, history_vr_lists,
     train_u, train_v, train_r, test_u, test_v, test_r, 
     social_adj_lists, _) = base_pickle_data
    
    # Convert R_v values to indices
    R_v_indices = {}
    for item_id, rv_value in R_v_normalized.items():
        R_v_indices[item_id] = bin_to_discrete_rating(rv_value)
    
    print(f"R_v index distribution:")
    index_counts = {}
    for idx in R_v_indices.values():
        index_counts[idx] = index_counts.get(idx, 0) + 1
    for idx in sorted(index_counts.keys()):
        print(f"  Index {idx}: {index_counts[idx]} items")
    
    # Update history_ur_lists: Each user's ratings show different items' R_v values
    new_history_ur_lists = {}
    for user_id, item_list in history_u_lists.items():
        # Get R_v index for each item this user interacted with
        new_history_ur_lists[user_id] = [R_v_indices.get(item, 0) for item in item_list]
    
    # Update history_vr_lists: Each item's ratings become constant (their R_v)
    new_history_vr_lists = {}
    for item_id, user_list in history_v_lists.items():
        item_rv_index = R_v_indices.get(item_id, 0)
        # All users for this item get same R_v index
        new_history_vr_lists[item_id] = [item_rv_index] * len(user_list)
    
    # Keep train_r and test_r as original float values (unchanged)
    print(f"✓ Updated history_ur_lists for {len(new_history_ur_lists)} users")
    print(f"✓ Updated history_vr_lists for {len(new_history_vr_lists)} items")
    print(f"✓ Keeping train_r and test_r as original float values")
    
    return (history_u_lists, new_history_ur_lists, history_v_lists, new_history_vr_lists,
            train_u, train_v, train_r, test_u, test_v, test_r,
            social_adj_lists, ratings_list)


def main():
    print("=" * 80)
    print("Creating T_u-only and R_v-only pickle files for ablation studies")
    print("FakeHealth Dataset")
    print("=" * 80)
    
    # Load original pickle file
    print(f"\n1. Loading base pickle file...")
    print(f"   Path: {BASE_PICKLE_PATH}")
    with open(BASE_PICKLE_PATH, 'rb') as f:
        base_pickle_data = pickle.load(f)
    print("   ✓ Loaded successfully")
    
    # Load raw data
    print(f"\n2. Loading raw data files...")
    trainset = pd.read_csv(TRAINSET_PATH)
    testset = pd.read_csv(TESTSET_PATH)
    
    print(f"   Trainset: {len(trainset)} interactions")
    print(f"   Testset: {len(testset)} interactions")
    print(f"   Unique users: {pd.concat([trainset['user'], testset['user']]).nunique()}")
    print(f"   Unique items: {pd.concat([trainset['item'], testset['item']]).nunique()}")
    
    # Get total items count
    total_items = pd.concat([trainset['item'], testset['item']]).nunique()
    print(f"   Total unique items (N): {total_items}")
    
    # Show label distribution
    all_labels = pd.concat([trainset['label'], testset['label']])
    print(f"   Labels: {all_labels.value_counts().to_dict()}")
    
    # Compute T_u (user trustworthiness)
    print(f"\n3. Computing T_u (user trustworthiness)...")
    T_u = compute_user_trustworthiness(trainset, testset)
    
    # Normalize T_u to [0.5, 4.0]
    print(f"\n4. Normalizing T_u to [0.5, 4.0] scale...")
    # T_u is already in [0, 1], so we can directly scale
    T_u_normalized = {user: 0.5 + (tu * 3.5) for user, tu in T_u.items()}
    print(f"   Normalized T_u range: [{min(T_u_normalized.values()):.4f}, {max(T_u_normalized.values()):.4f}]")
    
    # Compute R_v (item engagement penalty)
    print(f"\n5. Computing R_v (item engagement penalty)...")
    R_v = compute_item_engagement_penalty(trainset, testset, total_items)
    
    # Normalize R_v to [0.5, 4.0]
    print(f"\n6. Normalizing R_v to [0.5, 4.0] scale...")
    R_v_normalized = normalize_to_rating_scale(R_v)
    print(f"   Normalized R_v range: [{min(R_v_normalized.values()):.4f}, {max(R_v_normalized.values()):.4f}]")
    
    # Get ratings_list from original pickle
    ratings_list = base_pickle_data[11]
    
    # Create T_u-only pickle
    print(f"\n7. Creating T_u-only pickle file...")
    tu_pickle_data = create_tu_only_pickle(base_pickle_data, T_u_normalized, ratings_list)
    with open(OUTPUT_TU_PATH, 'wb') as f:
        pickle.dump(tu_pickle_data, f)
    print(f"   ✓ Saved to: {OUTPUT_TU_PATH}")
    
    # Create R_v-only pickle
    print(f"\n8. Creating R_v-only pickle file...")
    rv_pickle_data = create_rv_only_pickle(base_pickle_data, R_v_normalized, ratings_list)
    with open(OUTPUT_RV_PATH, 'wb') as f:
        pickle.dump(rv_pickle_data, f)
    print(f"   ✓ Saved to: {OUTPUT_RV_PATH}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✓ Original pickle: {BASE_PICKLE_PATH}")
    print(f"✓ T_u-only pickle: {OUTPUT_TU_PATH}")
    print(f"✓ R_v-only pickle: {OUTPUT_RV_PATH}")
    
    print(f"\nT_u Statistics:")
    print(f"  Users: {len(T_u)}")
    print(f"  Raw range: [{min(T_u.values()):.4f}, {max(T_u.values()):.4f}]")
    print(f"  Normalized range: [{min(T_u_normalized.values()):.4f}, {max(T_u_normalized.values()):.4f}]")
    
    print(f"\nR_v Statistics:")
    print(f"  Items: {len(R_v)}")
    print(f"  Raw range: [{min(R_v.values()):.4f}, {max(R_v.values()):.4f}]")
    print(f"  Normalized range: [{min(R_v_normalized.values()):.4f}, {max(R_v_normalized.values()):.4f}]")
    
    print("\nNext steps:")
    print("  1. Run experiments with original pickle (t(u,v) = T_u × R_v)")
    print("  2. Run experiments with T_u-only pickle")
    print("  3. Run experiments with R_v-only pickle")
    print("  4. Compare MRR and MC metrics across all three versions")
    
    print("\n" + "=" * 80)
    print("✓ Script completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()