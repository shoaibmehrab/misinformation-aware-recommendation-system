#!/usr/bin/env python3
"""Extract comprehensive sample data from pickle file for all objects"""

import pickle

PICKLE_PATH = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/processed-files/TrustWorthy_Politifact_MAGrec_Constant.pickle"
OUTPUT_PATH = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/pickle_data_comprehensive_Constant.txt"
LIST_SAMPLE = 50
DICT_SAMPLE = 10

print("Loading pickle file...")

with open(PICKLE_PATH, 'rb') as f:
    data = pickle.load(f)

history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, \
train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = data

print("Writing comprehensive data...")

with open(OUTPUT_PATH, 'w') as out:
    out.write("="*88 + "\n")
    out.write("COMPREHENSIVE PICKLE FILE DATA - ALL 12 OBJECTS\n")
    out.write("="*88 + "\n\n")
    
    # OVERVIEW
    out.write("DATASET OVERVIEW\n")
    out.write("-"*88 + "\n")
    out.write(f"Training instances: {len(train_u):,}\n")
    out.write(f"Test instances: {len(test_u):,}\n")
    out.write(f"Users: {len(history_u_lists):,}\n")
    out.write(f"Items: {len(history_v_lists):,}\n")
    out.write(f"Social connections: {len(social_adj_lists):,} users\n")
    out.write(f"Score levels: {len(ratings_list)}\n\n")
    
    # 1. history_u_lists
    out.write("="*88 + "\n")
    out.write("1. history_u_lists: User → Items Interacted\n")
    out.write("="*88 + "\n")
    out.write(f"Total: {len(history_u_lists):,} users\n\n")
    
    for i, uid in enumerate(list(history_u_lists.keys())[:DICT_SAMPLE]):
        items = list(history_u_lists[uid])
        out.write(f"User {uid:4d}: {len(items):3d} items → {items[:20]}\n")
    
    # 2. history_ur_lists
    out.write("\n" + "="*88 + "\n")
    out.write("2. history_ur_lists: User → Rating Indices (ALREADY CONVERTED!)\n")
    out.write("="*88 + "\n")
    out.write(f"Total: {len(history_ur_lists):,} users\n\n")
    
    for i, uid in enumerate(list(history_ur_lists.keys())[:DICT_SAMPLE]):
        ratings = list(history_ur_lists[uid])
        out.write(f"User {uid:4d}: {len(ratings):3d} ratings → {ratings[:20]}\n")
    
    # 3. history_v_lists
    out.write("\n" + "="*88 + "\n")
    out.write("3. history_v_lists: Item → Users Interacted\n")
    out.write("="*88 + "\n")
    out.write(f"Total: {len(history_v_lists):,} items\n\n")
    
    for i, iid in enumerate(list(history_v_lists.keys())[:DICT_SAMPLE]):
        users = list(history_v_lists[iid])
        out.write(f"Item {iid:4d}: {len(users):3d} users → {users[:20]}\n")
    
    # 4. history_vr_lists
    out.write("\n" + "="*88 + "\n")
    out.write("4. history_vr_lists: Item → Rating Indices (ALREADY CONVERTED!)\n")
    out.write("="*88 + "\n")
    out.write(f"Total: {len(history_vr_lists):,} items\n\n")
    
    for i, iid in enumerate(list(history_vr_lists.keys())[:DICT_SAMPLE]):
        ratings = list(history_vr_lists[iid])
        out.write(f"Item {iid:4d}: {len(ratings):3d} ratings → {ratings[:20]}\n")
    
    # 5-7. Training lists
    out.write("\n" + "="*88 + "\n")
    out.write("5-7. TRAINING DATA (train_u, train_v, train_r)\n")
    out.write("="*88 + "\n")
    out.write(f"Total instances: {len(train_u):,}\n")
    out.write(f"Unique users: {len(set(train_u)):,}\n")
    out.write(f"Unique items: {len(set(train_v)):,}\n\n")
    
    out.write(f"First {LIST_SAMPLE} instances:\n")
    out.write(f"{'#':<5} {'User':<6} {'Item':<6} {'Score(float)':<13} {'WrongIdx':<10} {'CorrectIdx'}\n")
    out.write("-"*88 + "\n")
    
    for i in range(min(LIST_SAMPLE, len(train_u))):
        u, v, r = train_u[i], train_v[i], train_r[i]
        wrong = int(r)
        correct = ratings_list[r]
        out.write(f"{i:<5} {u:<6} {v:<6} {r:<13.1f} {wrong:<10} {correct}\n")
    
    out.write(f"\nScore distribution in train_r:\n")
    counts = {}
    for r in train_r:
        counts[r] = counts.get(r, 0) + 1
    for score in sorted(counts.keys()):
        pct = counts[score]*100/len(train_r)
        out.write(f"  {score:3.1f}: {counts[score]:6,} ({pct:5.2f}%)\n")
    
    # 8-10. Test lists
    out.write("\n" + "="*88 + "\n")
    out.write("8-10. TEST DATA (test_u, test_v, test_r)\n")
    out.write("="*88 + "\n")
    out.write(f"Total instances: {len(test_u):,}\n")
    out.write(f"Unique users: {len(set(test_u)):,}\n")
    out.write(f"Unique items: {len(set(test_v)):,}\n\n")
    
    out.write(f"First {LIST_SAMPLE} instances:\n")
    out.write(f"{'#':<5} {'User':<6} {'Item':<6} {'Score(float)':<13} {'WrongIdx':<10} {'CorrectIdx'}\n")
    out.write("-"*88 + "\n")
    
    for i in range(min(LIST_SAMPLE, len(test_u))):
        u, v, r = test_u[i], test_v[i], test_r[i]
        wrong = int(r)
        correct = ratings_list[r]
        out.write(f"{i:<5} {u:<6} {v:<6} {r:<13.1f} {wrong:<10} {correct}\n")
    
    out.write(f"\nScore distribution in test_r:\n")
    counts = {}
    for r in test_r:
        counts[r] = counts.get(r, 0) + 1
    for score in sorted(counts.keys()):
        pct = counts[score]*100/len(test_r)
        out.write(f"  {score:3.1f}: {counts[score]:4,} ({pct:5.2f}%)\n")
    
    # 11. social_adj_lists
    out.write("\n" + "="*88 + "\n")
    out.write("11. social_adj_lists: User → Social Connections\n")
    out.write("="*88 + "\n")
    total_conn = sum(len(v) for v in social_adj_lists.values())
    out.write(f"Total users: {len(social_adj_lists):,}\n")
    out.write(f"Total connections: {total_conn:,}\n")
    out.write(f"Avg connections: {total_conn/len(social_adj_lists):.2f}\n\n")
    
    for i, uid in enumerate(list(social_adj_lists.keys())[:DICT_SAMPLE]):
        conn = list(social_adj_lists[uid])
        out.write(f"User {uid:4d}: {len(conn):3d} connections → {conn[:15]}\n")
    
    # 12. ratings_list
    out.write("\n" + "="*88 + "\n")
    out.write("12. ratings_list: Score → Index Mapping (THE KEY!)\n")
    out.write("="*88 + "\n\n")
    
    descriptions = {
        0.5: "Lowest trust × engagement",
        1.0: "Very low trust × engagement",
        1.5: "Low-medium trust × engagement",
        2.0: "Medium-low trust × engagement",
        2.5: "Medium trust × engagement",
        3.0: "Medium-high trust × engagement",
        3.5: "High trust × engagement",
        4.0: "Highest trust × engagement"
    }
    
    out.write(f"{'Score':<8} {'→':<3} {'Index':<8} {'Description'}\n")
    out.write("-"*88 + "\n")
    for score, idx in sorted(ratings_list.items()):
        out.write(f"{score:<8.1f} {'→':<3} {idx:<8} {descriptions.get(score, '')}\n")
    
    out.write("\n" + "="*88 + "\n")
    out.write("END OF COMPREHENSIVE DATA\n")
    out.write("="*88 + "\n")

print(f"✓ Written to: {OUTPUT_PATH}")
print(f"✓ All 12 objects documented!")
