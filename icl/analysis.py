import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import ast

from scipy.stats import spearmanr
import statsmodels.api as sm


# ==========================================
# CONFIGURATION
# ==========================================

# Map your model names to their respective result files
FILE_MAP = {
    "GPT4-4.1": "LiverpoolFC_gpt4_1_scaling_curve_results__buckets0-1-2-4-8-12-16.json",
    "GPT4-4o": "LiverpoolFC_gpt4_o_scaling_curve_results__buckets0-1-2-4-8-12-16.json",
    "Llama3-8B-IT": "LiverpoolFC_llama3_meta-llama-Meta-Llama-3-8B-Instruct_scaling_curve_results__buckets0-1-2-4-8-12-16.json",
    "Qwen3-4B-Thinking": "LiverpoolFC_qwen_Qwen-Qwen3-4B-Thinking-2507_scaling_curve_results__buckets0-1-2-4-8-12-16.json"
}

# The set of target words that actually changed (Ground Truth = 1)
# Replace this list with your actual annotated target words
CHANGED_WORDS = {"example_word1", "example_word2", "target_word_3"}

# Established color palette for consistency
COLOR_PALETTE = {
    "GPT4-4.1": "#F06292", # Pink
    "GPT4-4o": "#AFB42B",  # Olive
    "Llama3-8B-IT": "#4DB6AC", # Teal
    "Qwen3-4B-Thinking": "#9575CD" # Purple
}

# ==========================================
# DATA PROCESSING
# ==========================================


def load_and_process_data():
    all_data = []
    
    for model_name, file_path in FILE_MAP.items():
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping {model_name}.")
            continue
            
        with open(file_path, 'r') as f:
            content = json.load(f)
            
        # Iterate through ICL counts (0, 1, 2, 4, 8, 12, 16)
        for icl_count, seeds_data in content.items():
            for seed, stats in seeds_data.items():
                eval_data = stats.get('evaluation', {})
                scores = stats.get('scores', {})
                
                # 1. Extract standard metrics
                precision = eval_data.get('precision', 0)
                recall = eval_data.get('recall', 0)
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
                
                # 2. Calculate Mean Absolute Error (MAE)
                # target is 1 if in CHANGED_WORDS, else 0
                errors = []
                for word, score in scores.items():
                    target = 1 if word in CHANGED_WORDS else 0
                    errors.append(abs(score - target))
                
                mae = np.mean(errors) if errors else 0
                
                all_data.append({
                    "Model": model_name,
                    "ICL_Examples": int(icl_count),
                    "Seed": seed,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "MAE": mae
                })
                
    return pd.DataFrame(all_data)

# ==========================================
# PLOTTING
# ==========================================

def plot_metrics(df, metric_col, ylabel, title_suffix, filename):
    sns.set_theme(style="whitegrid")
    models_to_plot = ["GPT4-4.1", "GPT4-4o", "Llama3-8B-IT", "Qwen3-4B-Thinking"]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), sharex=True)
    axes = axes.flatten()
    
    for i, model in enumerate(models_to_plot):
        ax = axes[i]
        subset = df[df["Model"] == model]
        if subset.empty:
            continue
        
        sns.lineplot(
            data=subset,
            x="ICL_Examples",
            y=metric_col,
            ax=ax,
            color=COLOR_PALETTE[model],
            errorbar="se", # Standard Error shaded region
            marker='o',
            linewidth=4,
            markersize=12,
            label=model
        )
            
        # Aesthetic Styling
        ax.set_title(f"{model} - {title_suffix}", fontsize=24, fontweight='bold', pad=20)
        ax.set_xlabel("Number of ICL Examples", fontsize=20, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=20, fontweight='bold')
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=16, frameon=True, shadow=True)
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Saved: {filename}")


def plot_model_comparison(dict1, dict2, model1_name="Model 1", model2_name="Model 2", save_path=None):
    """
    Computes Spearman correlation and generates a high-visibility scatter plot
    with jittering and a smoothed trend line.
    """
    # 1. Align common keys and extract scores
    common_keys = sorted(list(set(dict1.keys()) & set(dict2.keys())))
    if not common_keys:
        print("No overlapping keys found.")
        return

    x = np.array([dict1[k] for k in common_keys])
    y = np.array([dict2[k] for k in common_keys])

    # 2. Compute Spearman Correlation
    rho, _ = spearmanr(x, y)

    # 3. Create Jittered Data for Plotting (to show density at 0 and 1)
    # We add a small amount of Gaussian noise
    jitter_x = x + np.random.normal(0, 0.015, size=len(x))
    jitter_y = y + np.random.normal(0, 0.015, size=len(y))

    # 4. Generate the Plot
    plt.figure(figsize=(10, 8), facecolor='white')
    ax = plt.gca()

    # Scatter: Yellow points with black outlines (Styled like the sample)
    plt.scatter(jitter_x, jitter_y,
                color='#f3d047',
                edgecolor='black',
                alpha=0.35,
                s=60,
                linewidth=0.5,
                label='Word Scores')

    # 5. Add Smoothed Trend Curve (LOWESS)
    # Note: We use original x,y for the math, but sort them for the line plot
    lowess = sm.nonparametric.lowess(y, x, frac=0.3)
    plt.plot(lowess[:, 0], lowess[:, 1], color='#4267b2', linewidth=5, zorder=10)

    # 6. Formatting & Readability
    plt.title(f'Spearman correlation = {rho:.2f}', fontsize=26, fontweight='bold', pad=20)
    plt.xlabel(f'{model1_name} Scores', fontsize=22, fontweight='bold')
    plt.ylabel(f'{model2_name} Scores', fontsize=22, fontweight='bold')

    # Setting logical bounds for probability scores
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18, fontweight='bold')
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18, fontweight='bold')

    # Aesthetics: Grid and Bold Spines
    plt.grid(True, linestyle='--', color='gray', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# Example Usage:
# plot_model_comparison(gpt4_scores, llama_scores, "GPT-4.1", "Llama3-8B")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    df_results = load_and_process_data()
    
    if not df_results.empty:
        # Generate the four figures
        plot_metrics(df_results, "Precision", "Precision", "Precision Scaling", "precision_scaling.png")
        plot_metrics(df_results, "Recall", "Recall", "Recall Scaling", "recall_scaling.png")
        plot_metrics(df_results, "F1", "F1 Score", "F1 Score Scaling", "f1_scaling.png")
        plot_metrics(df_results, "MAE", "Mean Absolute Error (|Score - Target|)", "MAE Scaling", "mae_scaling.png")
    else:
        print("No data found to plot.")
