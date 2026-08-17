"""
Visualization utilities for cross-modal attention and attribution results.

Handles heatmap overlays for individual samples and aggregate publication
figures (per-failure-category grid, coverage comparison bar chart).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def overlay_heatmap_on_image(orig_image, norm_heatmap, alpha=0.5):
    """
    Blend a normalized [0,1] heatmap onto the original RGB image using the
    jet colormap.

    Args:
        orig_image: PIL.Image (RGB)
        norm_heatmap: np.ndarray [H, W], normalized to [0, 1]
        alpha: blend weight for the heatmap layer

    Returns:
        np.ndarray [H, W, 3] float32 in [0, 1] - the blended overlay
    """
    orig_array = np.array(orig_image).astype(np.float32) / 255.0
    heatmap_rgb = cm.jet(norm_heatmap)[:, :, :3]
    overlay = (1 - alpha) * orig_array + alpha * heatmap_rgb
    return overlay


def plot_sample_explainability(orig_image, norm_heatmap, overlay, sample, prediction,
                                title_prefix, save_path, n_steps=None):
    """
    Standard 3-panel figure: original image | raw heatmap | overlay.

    Used for both per-sample attention and attribution visualizations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig_image)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    subtitle = f"{title_prefix}" + (f"\n(n_steps={n_steps})" if n_steps else "")
    axes[1].imshow(norm_heatmap, cmap="jet")
    axes[1].set_title(subtitle, fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\n{sample['instrument']} | {sample['failure_category']}",
        fontsize=12
    )
    axes[2].axis("off")

    plt.suptitle(
        f"Q: {sample['question'][:60]}\nPred: {prediction[:60]}",
        fontsize=10, y=1.02
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def plot_sample_explainability_compact(orig_image, overlay, sample, save_path):
    """
    Compact 2-panel figure (original | overlay only) - used in batch loops
    over many samples where the 3-panel version is unnecessarily heavy.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"{sample['instrument']} | {sample['failure_category']}")
    axes[1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_category_grid(results, categories, value_key="patch_attention",
                        grid_h_key="grid_h", grid_w_key="grid_w",
                        coverage_key="image_coverage_ratio",
                        title="Cross-Modal Attention by Failure Category",
                        save_path=None):
    """
    KEY PUBLICATION FIGURE: one representative heatmap per failure category,
    arranged in a grid (default 2x4 for 7 categories).

    Args:
        results: list of result dicts (attention_results or attribution_results)
        categories: ordered list of failure category names
        value_key: dict key holding the flat patch score array
        grid_h_key, grid_w_key: dict keys holding patch grid dimensions
        coverage_key: dict key holding the coverage ratio to display in subtitle
        save_path: if given, save the figure to this path
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, cat in enumerate(categories):
        cat_samples = [r for r in results if r["failure_category"] == cat]
        if not cat_samples:
            axes[i].axis("off")
            continue

        rep = cat_samples[0]
        patch_map = np.array(rep[value_key]).reshape(rep[grid_h_key], rep[grid_w_key])

        axes[i].imshow(patch_map, cmap="jet")
        axes[i].set_title(
            f"{cat}\n{rep['instrument']} | coverage={rep[coverage_key]:.3f}",
            fontsize=11
        )
        axes[i].axis("off")

    for j in range(len(categories), len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    if save_path:
        print(f"Saved: {save_path}")


def plot_coverage_comparison(df_categories, attention_col="avg_attention_coverage",
                              attribution_col="avg_attribution_coverage",
                              title="Cross-Modal Grounding by Failure Category",
                              save_path=None):
    """
    Bar chart comparing attention vs. attribution coverage across failure
    categories.

    Args:
        df_categories: pd.DataFrame with columns [category, attention_col, attribution_col]
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_categories))
    width = 0.35

    ax.bar(x - width / 2, df_categories[attention_col], width,
           label="Attention Coverage", color="steelblue")
    ax.bar(x + width / 2, df_categories[attribution_col], width,
           label="Attribution Coverage", color="coral")

    ax.set_xlabel("Failure Category")
    ax.set_ylabel("Average Image Coverage Ratio")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df_categories["category"], rotation=15, ha="right")
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    if save_path:
        print(f"Saved: {save_path}")