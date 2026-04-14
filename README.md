# Misinformation-Aware Recommendation System

This repository contains code and experiment artifacts for a misinformation-aware social recommendation framework on two datasets:

- FH (FakeHealth)
- Politifact

The codebase includes baseline recommenders (GraphRec, Popularity, SocialMF, FANAR) and misinformation-aware variants that use user trust/reputation and trusted-neighbor filtering in social graphs.

## Repository Layout

The project is organized by dataset name.

```text
misaware-recomm-github/
|-- FH/
|   |-- src/            # Training, evaluation, spread simulation scripts
|   |-- results/        # Generated MRR, MC, segmented, spread summaries

|-- Politifact/
|   |-- src/            # Training, evaluation, visualization, analysis scripts
|   |-- results/        # Generated MRR, MC, segmented, spread summaries

`-- README.md
```

## Main Methods in This Repository

Baselines:

- GraphRec (`graphrec.py`)
- Popularity (`popularity.py`)
- SocialMF (`socialMF.py`)

Misinformation-aware methods:

- Trustworthy Neighbor (`trustworthy-neighbour.py`)
- Trustworthy Neighbor (custom neighbor size) (`trustworthy-neighbour-custom.py`)
- Trustworthy Social (`trustworthy-social.py`)
- Trustworthy Social + Intersect Neighbor (`trustworthy-social-intersect-neighbour.py`)
- Trustworthy Social + Union Neighbor (`trustworthy-social-union-neighbour.py`)
- Custom neighbor-size variants for intersect/union (files ending in `-custom.py`)

Supporting analysis:

- Spread simulation (`spread.py`)
- Segmented evaluation (`evaluate-segmented-neighbour.py`)
- Politifact extra analysis (`analyze-recommendation.py`, `visualize_results.py`)

## Python Dependencies

There is currently no pinned `requirements.txt` in this repository. Install the required packages manually:

```bash
pip install numpy pandas torch lenskit tqdm networkx matplotlib seaborn
```

Notes:

- All major model scripts use PyTorch.
- `lenskit.topn` is used for ranking metrics.
- Some scripts only need a subset of these dependencies.

## Important: Path Configuration Is Required

Most scripts currently use hardcoded absolute paths such as:

- `/home/shoaib/recommender-system/royal/journal-revision/New/FH/...`
- `/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/...`

Before running experiments, open the script you want to execute and update path constants at the top of the file, typically:

- `PROCESSED_FILE_DIR`
- `DATA_FILE_DIR`
- `RESULTS_DIR`
- `SEEDS_DIR`
- `KEY_FILE_PATH`
- `BASE_PICKLE_PATH` or `ORIGINAL_PICKLE_PATH`

You can either:

1. Mirror the exact expected folder structure under your own machine paths, or
2. Edit those constants in each script to your local folders.

## Required Data Files (Per Dataset)

### FH

Typical required files referenced by scripts:

- Data folder:
  - `train_ratings.csv`
  - `test_ratings_final_before_train.csv`
  - `key_healthstory.csv`
- Processed folder:
  - `TrustWorthy_FH_MAGrec.pickle`
  - `FH_item_update.pickle`

### Politifact

Typical required files referenced by scripts:

- Data folder:
  - `train_ratings.csv`
  - `politifact_test.csv`
  - `politifact_Shu_fake_news_keyforSOCIALMF.csv`
- Processed folder:
  - `TrustWorthy_Politifact_MAGrec.pickle`

For ablation scripts, additional pickles may be needed (for example constant, T_u-only, or R_v-only variants).

## How to Run

Run commands from the dataset `src` directory so local module imports work correctly.

### FH: Example Runs

```bash
cd FH/src

# Baselines
python graphrec.py
python popularity.py --top_k 5 10 15
python socialMF.py --epochs 50 --top_k 5 10 15
python fanar.py --threshold 0.5

# Misinformation-aware core model
python trustworthy-neighbour.py --threshold 0.5

# Custom neighbor-size variant
python trustworthy-neighbour-custom.py --threshold 0.5 --neighbor_size 20

# Social variants
python trustworthy-social.py --threshold 0.5
python trustworthy-social-intersect-neighbour.py --threshold 0.5
python trustworthy-social-union-neighbour.py --threshold 0.5

# Spread simulation using generated seeds
python spread.py --experiment_type trustworthy_neighbor --threshold 0.5 --top_k 5 10 15 --simulations 100

# Segmented metrics
python evaluate-segmented-neighbour.py --experiment_type Trustworthy_Neighbor --threshold 0.5
```

### Politifact: Example Runs

```bash
cd Politifact/src

# Baselines
python graphrec.py
python popularity.py --top_k 5 10 15
python socialMF.py --epochs 50 --top_k 5 10 15
python fanar.py

# Misinformation-aware core model
python trustworthy-neighbour.py --threshold 0.5

# Custom neighbor-size variant
python trustworthy-neighbour-custom.py --threshold 0.5 --neighbor_size 20

# Social variants
python trustworthy-social.py --threshold 0.5
python trusworthy-social-intersect-neighbour.py --threshold 0.5
python trustworthy-social-union-neighbour.py --threshold 0.5

# Spread simulation
python spread.py --experiment_type trustworthy_neighbor --threshold 0.5 --top_k 5 10 15 --simulations 100

# Additional analysis
python analyze-recommendation.py --experiment_type trustworthy_social_intersect_neighbor --threshold 0.5
python visualize_results.py --threshold 0.5 --top_k 5 10 15
```

## Output Artifacts

A typical run generates:

- Recommendation CSV files (written to dataset data folders configured in scripts)
- Summary metrics in `results/`:
  - MRR summaries
  - MC summaries
  - Segmented evaluation summaries
  - Spread simulation summaries
- Seed user files in `seeds/` grouped by method, threshold, and top-k

This repository already includes many precomputed `results/` and `seeds/` files for both datasets.

## Reproducibility Tips

- Keep threshold and neighbor-size settings consistent across training, evaluation, and spread scripts.
- Run prerequisite scripts first for methods that depend on generated pickle files.
- If you use `--skip_data_prep`, verify the required intermediate pickle already exists.
- In this codebase, some script names intentionally use `neighbour` while others use `neighbor`. Use the exact file names as present in each folder.

## Citation

If you use this codebase in academic work, please cite your corresponding paper or report for the misinformation-aware recommendation framework.
