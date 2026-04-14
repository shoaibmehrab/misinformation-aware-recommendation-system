#!/usr/bin/env python3
"""Extract sample data from pickle file"""

import pickle

PICKLE_PATH = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/processed-files/TrustWorthy_Politifact_MAGrec.pickle"
OUTPUT_PATH = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/pickle_data_sample.txt"
SAMPLE_SIZE = 20

print(f"Loading pickle file...")

with open(PICKLE_PATH, 'rb') as f:
    data = pickle.load(f)

history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, \
train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = data

print(f"✓ Loaded successfully")

# Write to file
with open(OUTPUT_PATH, 'w') as out:
    out.write("="*80 + "\n")
    out.write("PICKLE FILE DATA SAMPLE\n")
    out.write("="*80 + "\n\n")
    
    out.write("OVERVIEW\n")
    out.write("-"*80 + "\n")
    out.write(f"Training instances: {len(train_u):,}\n")
    out.write(f"Test instances: {len(test_u):,}\n")
    out.write(f"Users: {len(history_u_lists):,}\n")
    out.write(f"Items: {len(history_v_lists):,}\n")
    out.write(f"Social connections: {len(social_adj_lists):,} users\n")
    out.write(f"Score levels: {len(ratings_list)}\n\n")
    
    # ratings_list
    out.write("="*80 + "\n")
    out.write("ratings_list: Score → Embedding Index Mapping\n")
    out.write("="*80 + "\n")
    for score, idx in sorted(ratings_list.items()):
        out.write(f"  {score:4.1f} → {idx}\n")
    out.write("\n")
    
    # Sample training data
    out.write("="*80 + "\n")
    out.write("TRAINING DATA SAMPLE (First 20)\n")
    out.write("="*80 + "\n")
    out.write(f"{'#':<4} {'User':<6} {'Item':<6} {'Score':<7} {'→ Wrong':<9} {'Correct':<9} {'Status'}\n")
    out.write("-"*80 + "\n")
    
    for i in range(min(SAMPLE_SIZE, len(train_u))):
        user, item, score = train_u[i], train_v[i], train_r[i]
        wrong_idx = int(score)
        correct_idx = ratings_list[score]
        status = "✓" if wrong_idx == correct_idx else "✗ WRONG"
        out.write(f"{i:<4} {user:<6} {item:<6} {score:<7.1f} {wrong_idx:<9} {correct_idx:<9} {status}\n")
    
    out.write("\n")
    
    # User history samples
    out.write("="*80 + "\n")
    out.write("USER HISTORY SAMPLES (First 5 users)\n")
    out.write("="*80 + "\n")
    
    sample_users = list(history_u_lists.keys())[:5]
    for user_id in sample_users:
        items = history_u_lists[user_id]
        ratings = history_ur_lists[user_id]
        out.write(f"\nUser {user_id}:\n")
        out.write(f"  Items interacted: {list(items)[:10]}\n")
        out.write(f"  Ratings (scores): {list(ratings)[:10]}\n")
        out.write(f"  Total interactions: {len(items)}\n")
    
    out.write("\n")
    
    # Social network sample
    out.write("="*80 + "\n")
    out.write("SOCIAL NETWORK SAMPLE (First 5 users)\n")
    out.write("="*80 + "\n")
    
    sample_social = list(social_adj_lists.keys())[:5]
    for user_id in sample_social:
        connections = social_adj_lists[user_id]
        out.write(f"User {user_id}: {len(connections)} connections → {list(connections)[:10]}\n")
    
    out.write("\n" + "="*80 + "\n")
    out.write("END OF SAMPLE\n")
    out.write("="*80 + "\n")

print(f"✓ Sample data written to: {OUTPUT_PATH}")
print(f"\nQuick stats:")
print(f"  Training: {len(train_u):,} instances")
print(f"  Score range: {min(train_r):.1f} to {max(train_r):.1f}")
print(f"  Unique scores: {len(set(train_r))}")
