"""
Visualization functions for the MiniLab.
Creates plots for evaluating models and learning.

  - Visualization of model complexity (Features) and performance (Task 2.4)
  - Heatmap & curve plots for polynomial models (Task 3.2, 3.3)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Visualization of model complexity (Features) and performance (Task 2.4)
# ---------------------------------------------------------------------
def plot_feature_performance(results, output_dir="results", file_name="Task_2"):
    """
    Create and save a performance plot showing how model performance
    (R² and RMSE) changes with the number of features.

    Parameters
    ----------
    results : list of dict
        Example format:
        [
            {"n_features": 1, "features": ["livingSpace"], "r2": 0.70, "rmse": 250.0},
            {"n_features": 2, "features": ["livingSpace", "numberOfRooms"], "r2": 0.78, "rmse": 210.0},
            ...
        ]
    output_dir : str, default="results"
        Directory where the plot is saved.
    file_name : str, default="Task_2"
        Identifier used for the filename.

    Returns
    -------
    str
        Path to the saved plot file.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_features = [r["n_features"] for r in results]
    r2_values = [r["r2"] for r in results]
    rmse_values = [r["rmse"] for r in results]

    # --- Create figure ---
    fig, ax1 = plt.subplots(figsize=(6, 4))

    color_r2 = "tab:blue"
    color_rmse = "tab:red"

    ax1.set_xlabel("Number of features")
    ax1.set_ylabel("R²", color=color_r2)
    ax1.tick_params(axis="y", labelcolor=color_r2)

    ax2 = ax1.twinx()
    ax2.set_ylabel("RMSE (€)", color=color_rmse)
    ax2.tick_params(axis="y", labelcolor=color_rmse)

    plt.title("Model performance (validation) vs. number of features")
    fig.tight_layout()

    output_path = os.path.join(output_dir, f"{file_name}.pdf")
    plt.savefig(output_path)
    plt.close(fig)

    return output_path

# ---------------------------------------------------------------------
# Heatmap for polynomial performance (Matplotlib version, Task 3.2)
# ---------------------------------------------------------------------
def plot_heatmap_performance(results_list, output_dir="results", metric="r2_val", file_name="Task_3_heatmap"):
    """
    Visualize validation performance as a heatmap over polynomial degree x number of features.
    Works directly from flat list of dicts.
    """
    # Creates a 3×6 heatmap with random values between 0 and 1.
    heatmap_data = np.random.rand(3, 6)

    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(heatmap_data, cmap="viridis", origin="lower", aspect="auto")

    # Axis labels
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Feature set index")

    ax.set_xticks(range(6))
    ax.set_xticklabels([1, 2, 3, 4, 5, 6])

    ax.set_yticks(range(3))
    ax.set_yticklabels([f"Set {i+1}" for i in range(3)])

    # Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Dummy R² value")

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{file_name}.pdf")
    plt.savefig(path)
    plt.close(fig)
    #print(f"Saved heatmap to: {path}")
    return path


# ---------------------------------------------------------------------
# 2D plot: train vs. validation curves over degree (Task 3.3)
# ---------------------------------------------------------------------
def plot_polynomial_results(results_list, output_dir="results", file_name="Task_3_curves"):
    """
    Plot R² (train vs validation) over polynomial degree for a flat list of results.

    Parameters
    ----------
    results_list : list of dict
        Flat list from evaluate_polynomial_models().

    Dummy version: create exactly three PDF plots with names:
      Task_3_3_A.pdf
      Task_3_3_A_B.pdf
      Task_3_3_A_B_C.pdf
    """
    os.makedirs(output_dir, exist_ok=True)

    feature_sets = [
        ["A"],
        ["A", "B"],
        ["A", "B", "C"],
    ]

    created_paths = []

    for fs in feature_sets:
        fig, ax = plt.subplots(figsize=(4, 3))

        # Dummy placeholder curve
        degrees = [1, 2, 3, 4, 5, 6]
        dummy_train = np.random.rand(len(degrees))
        dummy_val   = np.random.rand(len(degrees))

        ax.plot(degrees, dummy_train, marker="o", label="Train R²")

        ax.set_title("Dummy polynomial curve")
        ax.set_xlabel("Degree")
        ax.set_ylabel("R²")
        ax.grid(True, linestyle=":")
        ax.legend(fontsize=6)

        # Build filename from base name + feature suffix
        suffix = "_".join(fs)
        path = os.path.join(output_dir, f"{file_name}_{suffix}.pdf")
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

        created_paths.append(path)

    return created_paths