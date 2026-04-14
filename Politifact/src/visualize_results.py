import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# --- 1. Configuration ---
RESULTS_DIR = "/home/shoaib/recommender-system/royal/journal-revision/New/Politifact/results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# --- 2. New, Clearer Plotting Functions ---

def plot_bar_comparison(data, metric_name, threshold):
    """
    Creates a clean, grouped bar plot to compare MRR or MC across all experiments.
    """
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Standardize column names for plotting
    data = data.rename(columns=str.lower)
    metric_name_lower = metric_name.lower()
    
    ax = sns.barplot(x='top-k', y=metric_name_lower, hue='experiment', data=data, palette='viridis')

    # Add value labels on top of each bar
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=9)

    plt.title(f'{metric_name} Comparison Across Experiments (Threshold: {threshold})', fontsize=18, pad=20)
    plt.xlabel('Top-K Recommendations', fontsize=14)
    plt.ylabel(f'Average {metric_name}', fontsize=14)
    plt.legend(title='Experiment Type', fontsize=11)
    plt.ylim(top=plt.ylim()[1] * 1.15) # Add some space for the labels
    plt.tight_layout()
    
    plot_filename = os.path.join(PLOTS_DIR, f'comparison_{metric_name.lower()}_{threshold}.png')
    plt.savefig(plot_filename)
    print(f"✓ Saved clear {metric_name} comparison plot to {plot_filename}")
    plt.close()


def plot_spread_comparison(data, threshold):
    """
    Creates a clean line plot to compare Spread across all experiments.
    Generates one plot per spread model (LTM, ICM, etc.).
    """
    # Filter for columns that are for FINAL spread values only
    spread_columns = [col for col in data.columns if col.endswith('_final')]
    
    # Standardize column names for plotting
    data = data.rename(columns=str.lower)
    spread_columns_lower = [col.lower() for col in spread_columns]

    for spread_col in spread_columns_lower:
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        plot_title = spread_col.replace('_', ' ').replace('spread', 'Spread').title()

        sns.lineplot(
            x='top_k', 
            y=spread_col, 
            hue='experiment', 
            style='experiment',
            data=data, 
            markers=True, 
            markersize=8,
            dashes=False,
            linewidth=2.5
        )
        
        plt.title(f'{plot_title} Comparison Across Experiments (Threshold: {threshold})', fontsize=18)
        plt.xlabel('Top-K Recommendations', fontsize=14)
        plt.ylabel('Average Final Spread', fontsize=14)
        plt.xticks(data['top_k'].unique())
        plt.legend(title='Experiment Type', fontsize=11)
        plt.tight_layout()
        
        plot_filename = os.path.join(PLOTS_DIR, f'comparison_{spread_col.lower()}_{threshold}.png')
        plt.savefig(plot_filename)
        print(f"✓ Saved clear {plot_title} comparison plot to {plot_filename}")
        plt.close()


# --- 3. Main Execution Block ---

def main(args):
    """
    Main function to load data and generate a set of clear, separate plots.
    """
    print("--- Starting Visualization Script ---")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # --- Load all data ---
    all_mrr_data, all_mc_data, all_spread_data = [], [], []

    if args.metric_exp_types:
        for exp_type in args.metric_exp_types:
            mrr_file = os.path.join(RESULTS_DIR, f'{exp_type}_mrr_summary_{args.threshold}_threshold.csv')
            if os.path.exists(mrr_file):
                df = pd.read_csv(mrr_file); df['Experiment'] = exp_type; all_mrr_data.append(df)
            else: print(f"  ✗ Warning: MRR file not found: {mrr_file}")
            
            mc_file = os.path.join(RESULTS_DIR, f'{exp_type}_mc_summary_{args.threshold}_threshold.csv')
            if os.path.exists(mc_file):
                df = pd.read_csv(mc_file); df['Experiment'] = exp_type; all_mc_data.append(df)
            else: print(f"  ✗ Warning: MC file not found: {mc_file}")

    if args.spread_exp_types:
        for exp_type in args.spread_exp_types:
            spread_file = os.path.join(RESULTS_DIR, f'spread_results_{exp_type}_{args.threshold}.csv')
            if os.path.exists(spread_file):
                df = pd.read_csv(spread_file)
                # Normalize experiment name to match metric files if needed
                df['experiment'] = df['experiment'].replace('trustworthy_neighbor', 'trust_neighbour')
                all_spread_data.append(df)
            else: print(f"  ✗ Warning: Spread file not found: {spread_file}")

    # --- Generate Plots ---
    if all_mrr_data:
        plot_bar_comparison(pd.concat(all_mrr_data, ignore_index=True), 'MRR', args.threshold)
    else: print("\nNo MRR data found to plot.")

    if all_mc_data:
        plot_bar_comparison(pd.concat(all_mc_data, ignore_index=True), 'MC', args.threshold)
    else: print("No MC data found to plot.")
            
    if all_spread_data:
        # Filter for user-specified top-k values
        spread_df = pd.concat(all_spread_data, ignore_index=True)
        if args.top_k:
            spread_df = spread_df[spread_df['top_k'].isin(args.top_k)]
        if not spread_df.empty:
            plot_spread_comparison(spread_df, args.threshold)
        else:
            print("No Spread data left to plot after filtering for Top-K.")
    else: print("No Spread data found to plot.")
        
    print("\n--- Visualization script finished. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate clear comparison plots for recommendation metrics.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Reputation threshold used for the experiments.')
    parser.add_argument('--top_k', type=int, nargs='+', help='A list of Top-K values to filter the spread plot by (e.g., 5 10 15).')
    parser.add_argument('--metric_exp_types', nargs='+', help='Experiment names for MRR/MC files (e.g., trust_neighbour).')
    parser.add_argument('--spread_exp_types', nargs='+', help='Experiment names for Spread files (e.g., trustworthy_neighbor).')
    args = parser.parse_args()
    main(args)


