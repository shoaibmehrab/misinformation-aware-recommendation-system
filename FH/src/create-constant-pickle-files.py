"""
Script to create constant rating pickle file (baseline for ablation studies) - FakeHealth Dataset.

This script:
1. Loads the original base pickle file
2. Sets all ratings in history_ur_lists and history_vr_lists to constant index 1
3. Keeps train_r and test_r as original float values
4. Creates new pickle file with constant ratings (baseline)

This serves as a baseline to compare against T_u-only and R_v-only versions.
"""

import pickle
import os

# File paths
PROCESSED_FILE_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/FH/processed-files"
BASE_PICKLE_PATH = os.path.join(PROCESSED_FILE_DIR, "TrustWorthy_FH_MAGrec.pickle")
OUTPUT_CONSTANT_PATH = os.path.join(PROCESSED_FILE_DIR, "TrustWorthy_FH_MAGrec_Constant.pickle")

# Constant rating index (1 corresponds to rating 1.0)
CONSTANT_RATING_INDEX = 1


def create_constant_pickle(base_pickle_data, constant_index, ratings_list):
    """
    Create constant rating pickle file.
    
    In constant version:
    - history_ur_lists: All ratings become constant index (e.g., 1)
    - history_vr_lists: All ratings become constant index (e.g., 1)
    - train_r and test_r: Keep as original float values
    
    Args:
        base_pickle_data: tuple of original pickle data
        constant_index: int, the constant index to assign (0-7)
        ratings_list: original ratings_list mapping
    
    Returns:
        tuple: new pickle data with constant rating indices
    """
    print("\n=== Creating Constant Rating Pickle File ===")
    
    (history_u_lists, history_ur_lists, history_v_lists, history_vr_lists,
     train_u, train_v, train_r, test_u, test_v, test_r, 
     social_adj_lists, _) = base_pickle_data
    
    # Get the rating value corresponding to the index
    rating_value = None
    for rating, idx in ratings_list.items():
        if idx == constant_index:
            rating_value = rating
            break
    
    print(f"Constant rating index: {constant_index} (rating value: {rating_value})")
    
    # Update history_ur_lists: All ratings become constant
    new_history_ur_lists = {}
    total_ur_ratings = 0
    for user_id, item_list in history_u_lists.items():
        # All ratings for this user become constant
        new_history_ur_lists[user_id] = [constant_index] * len(item_list)
        total_ur_ratings += len(item_list)
    
    # Update history_vr_lists: All ratings become constant
    new_history_vr_lists = {}
    total_vr_ratings = 0
    for item_id, user_list in history_v_lists.items():
        # All ratings for this item become constant
        new_history_vr_lists[item_id] = [constant_index] * len(user_list)
        total_vr_ratings += len(user_list)
    
    # Keep train_r and test_r as original float values (unchanged)
    print(f"✓ Updated history_ur_lists for {len(new_history_ur_lists)} users ({total_ur_ratings} ratings)")
    print(f"✓ Updated history_vr_lists for {len(new_history_vr_lists)} items ({total_vr_ratings} ratings)")
    print(f"✓ All ratings set to constant index {constant_index} (rating value {rating_value})")
    print(f"✓ Keeping train_r and test_r as original float values")
    
    return (history_u_lists, new_history_ur_lists, history_v_lists, new_history_vr_lists,
            train_u, train_v, train_r, test_u, test_v, test_r,
            social_adj_lists, ratings_list)


def main():
    print("=" * 80)
    print("Creating Constant Rating Pickle File (Baseline for Ablation Studies)")
    print("FakeHealth Dataset")
    print("=" * 80)
    
    # Load original pickle file
    print(f"\n1. Loading base pickle file...")
    print(f"   Path: {BASE_PICKLE_PATH}")
    with open(BASE_PICKLE_PATH, 'rb') as f:
        base_pickle_data = pickle.load(f)
    print("   ✓ Loaded successfully")
    
    # Display pickle structure
    print(f"\n2. Analyzing pickle structure...")
    history_u_lists = base_pickle_data[0]
    history_v_lists = base_pickle_data[2]
    train_u = base_pickle_data[4]
    test_u = base_pickle_data[7]
    ratings_list = base_pickle_data[11]
    
    print(f"   Users: {len(history_u_lists)}")
    print(f"   Items: {len(history_v_lists)}")
    print(f"   Training instances: {len(train_u)}")
    print(f"   Test instances: {len(test_u)}")
    print(f"   Ratings mapping: {ratings_list}")
    
    # Create constant pickle
    print(f"\n3. Creating constant rating pickle file...")
    constant_pickle_data = create_constant_pickle(base_pickle_data, CONSTANT_RATING_INDEX, ratings_list)
    
    # Save to file
    print(f"\n4. Saving pickle file...")
    with open(OUTPUT_CONSTANT_PATH, 'wb') as f:
        pickle.dump(constant_pickle_data, f)
    print(f"   ✓ Saved to: {OUTPUT_CONSTANT_PATH}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✓ Original pickle: {BASE_PICKLE_PATH}")
    print(f"✓ Constant pickle: {OUTPUT_CONSTANT_PATH}")
    print(f"\nConfiguration:")
    print(f"  Constant rating index: {CONSTANT_RATING_INDEX}")
    print(f"  Constant rating value: {[k for k, v in ratings_list.items() if v == CONSTANT_RATING_INDEX][0]}")
    print(f"  All history_ur_lists ratings: {CONSTANT_RATING_INDEX}")
    print(f"  All history_vr_lists ratings: {CONSTANT_RATING_INDEX}")
    print(f"  train_r and test_r: Kept as original float values")
    
    print("\nNext steps:")
    print("  1. Run experiments with original pickle (t(u,v) = T_u × R_v)")
    print("  2. Run experiments with T_u-only pickle")
    print("  3. Run experiments with R_v-only pickle")
    print("  4. Run experiments with constant pickle (baseline)")
    print("  5. Compare MRR and MC metrics across all versions")
    
    print("\n" + "=" * 80)
    print("✓ Script completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
