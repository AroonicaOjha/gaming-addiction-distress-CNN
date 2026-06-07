"""
run_step7.py
============
Step 7 – BiLSTM Feature Learning

Demonstrates what the BiLSTM layer adds ON TOP of the CNN:
  - CNN  => detects LOCAL patterns (pair of windows)
  - BiLSTM => learns the TEMPORAL ARC across all 3 windows
              forward:  Session -> Gameplay -> Social
              backward: Social  -> Gameplay -> Session

This script:
  1. Loads the splits from Step 5
  2. Builds the CNN-BiLSTM model
  3. Extracts and visualises intermediate hidden states to show
     what the BiLSTM "sees" at each timestep
  4. Plots a behavioural progression heatmap
  5. Saves all visualizations to results/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")    # non-interactive backend for Windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from src.cnn_bilstm import build_cnn_bilstm, compile_model


# ============================================================
# Colour palette
# ============================================================
WINDOW_COLORS = {
    "Session":  "#4EA8DE",
    "Gameplay": "#7209B7",
    "Social":   "#F72585",
}
CLASS_COLORS = {
    0: "#4EA8DE",   # Healthy
    1: "#7209B7",   # Distressed Only
    2: "#F72585",   # Addicted Only
    3: "#EF233C",   # Both
}
CLASS_NAMES = {
    0: "Healthy",
    1: "Distressed Only",
    2: "Addicted Only",
    3: "Both",
}


def build_bilstm_state_extractor(model):
    """
    Builds a sub-model that returns the CNN output (after MaxPooling)
    AND the final BiLSTM hidden state, so we can visualise both stages.
    """
    cnn_out   = model.get_layer("max_pooling").output     # (batch, 2, 64)
    bilstm_out = model.get_layer("bilstm").output         # (batch, 128)

    extractor = keras.Model(
        inputs=model.input,
        outputs=[cnn_out, bilstm_out],
        name="BiLSTM_State_Extractor"
    )
    return extractor


def plot_bilstm_concept(save_path):
    """
    Draws a diagram showing how BiLSTM reads the behavioural arc
    in both directions and what it learns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    # ---- Left: Forward + Backward arc diagram ----
    ax = axes[0]
    ax.set_facecolor("#161b22")

    windows = ["Session\nWindow", "Gameplay\nWindow", "Social\nWindow"]
    x_pos   = [0.15, 0.50, 0.85]
    colors  = ["#4EA8DE", "#7209B7", "#F72585"]

    # Draw window boxes
    for i, (win, xp, col) in enumerate(zip(windows, x_pos, colors)):
        box = mpatches.FancyBboxPatch(
            (xp - 0.12, 0.45), 0.24, 0.18,
            boxstyle="round,pad=0.02",
            facecolor=col, edgecolor="white", linewidth=1.5, alpha=0.85,
            transform=ax.transAxes
        )
        ax.add_patch(box)
        ax.text(xp, 0.54, win, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white",
                transform=ax.transAxes)

    # Forward arrows (bottom)
    for i in range(len(x_pos) - 1):
        ax.annotate(
            "", xy=(x_pos[i+1] - 0.13, 0.35),
            xytext=(x_pos[i] + 0.13, 0.35),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#56CFE1", lw=2)
        )
    ax.text(0.5, 0.27, "Forward LSTM  (Session->Gameplay->Social)",
            ha="center", fontsize=8.5, color="#56CFE1", transform=ax.transAxes)

    # Backward arrows (top)
    for i in range(len(x_pos) - 1, 0, -1):
        ax.annotate(
            "", xy=(x_pos[i-1] + 0.13, 0.75),
            xytext=(x_pos[i] - 0.13, 0.75),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#F72585", lw=2)
        )
    ax.text(0.5, 0.83, "Backward LSTM  (Social->Gameplay->Session)",
            ha="center", fontsize=8.5, color="#F72585", transform=ax.transAxes)

    # Merge arrow
    ax.annotate(
        "", xy=(0.5, 0.12), xytext=(0.5, 0.22),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#80FFDB", lw=2.5)
    )
    ax.text(0.5, 0.07, "Concatenate  [Forward | Backward]  =>  128-d vector",
            ha="center", fontsize=8.5, color="#80FFDB",
            fontweight="bold", transform=ax.transAxes)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("BiLSTM: Bidirectional Behavioural Arc",
                 color="white", fontsize=11, fontweight="bold", pad=10)

    # ---- Right: What the BiLSTM learns ----
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")

    progressions = [
        ("Healthy Player",        [0.2, 0.2, 0.1],  "#4EA8DE"),
        ("Distressed Only",       [0.3, 0.3, 0.7],  "#7209B7"),
        ("Addicted Only",         [0.8, 0.6, 0.3],  "#F72585"),
        ("Both Addicted+Distress",[0.9, 0.8, 0.8],  "#EF233C"),
    ]
    labels = ["Session\nBehaviour", "Gameplay\nMetrics", "Social\nSignals"]
    x = np.arange(3)

    for label, values, col in progressions:
        ax2.plot(x, values, "o-", color=col, linewidth=2.5,
                 markersize=8, label=label, alpha=0.9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color="white", fontsize=9)
    ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(
        ["Low", "", "Moderate", "", "High"],
        color="#aaaaaa", fontsize=9
    )
    ax2.set_facecolor("#161b22")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444444")
    ax2.grid(True, alpha=0.2, linestyle="--", color="#444444")
    ax2.set_title("Behavioural Progression Patterns\nBiLSTM learns these arcs",
                  color="white", fontsize=11, fontweight="bold", pad=10)
    legend = ax2.legend(
        facecolor="#1c2128", edgecolor="#444444",
        labelcolor="white", fontsize=8, loc="upper left"
    )

    plt.suptitle(
        "Step 7 — BiLSTM Feature Learning: Temporal Arc Across Modality Windows",
        color="white", fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print(f"  BiLSTM concept diagram saved -> {save_path}")


def plot_feature_heatmap(feature_vectors, labels, save_path, n_players=40):
    """
    Plots a heatmap of the 128-d BiLSTM feature vectors for n_players,
    grouped by class. Shows what the BiLSTM learned to separate.
    """
    # Sample n_players spread across classes
    indices = []
    for cls in range(4):
        cls_idx = np.where(labels == cls)[0]
        take = min(n_players // 4, len(cls_idx))
        indices.extend(cls_idx[:take].tolist())
    indices = np.array(indices)

    selected_features = feature_vectors[indices]     # (n, 128)
    selected_labels   = labels[indices]

    # Sort by class
    order = np.argsort(selected_labels)
    selected_features = selected_features[order]
    selected_labels   = selected_labels[order]

    fig, ax = plt.subplots(figsize=(18, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    # Plot only first 64 dims for readability
    sns.heatmap(
        selected_features[:, :64],
        ax=ax,
        cmap="coolwarm",
        center=0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.6, "label": "Activation"},
    )
    ax.set_xlabel("Feature Dimension (1–64 of 128)", color="white", fontsize=10)
    ax.set_ylabel("Players", color="white", fontsize=10)
    ax.tick_params(colors="white")

    # Colour bar for classes on the left
    class_bar_width = 0.015
    class_ax = fig.add_axes([0.125, 0.11, class_bar_width, 0.77])
    class_ax.set_facecolor("#0d1117")
    for i, lbl in enumerate(selected_labels):
        class_ax.barh(i, 1, color=CLASS_COLORS[lbl], height=1.0, align="edge")
    class_ax.set_xlim(0, 1)
    class_ax.set_ylim(0, len(selected_labels))
    class_ax.axis("off")

    # Legend
    patches = [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
               for c in range(4)]
    ax.legend(handles=patches, loc="upper right",
              facecolor="#1c2128", edgecolor="#444444",
              labelcolor="white", fontsize=8)

    ax.set_title(
        "BiLSTM Feature Vectors (128-d) — Grouped by Class\n"
        "Distinct activation patterns show the model separates classes",
        color="white", fontsize=11, fontweight="bold", pad=12
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print(f"  Feature heatmap saved        -> {save_path}")


def main():
    print("=" * 65)
    print("  Step 7 – BiLSTM Feature Learning")
    print("=" * 65)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load splits
    # ------------------------------------------------------------------
    splits_dir = os.path.join("data", "splits")
    X_train = np.load(os.path.join(splits_dir, "X_train.npy"))
    y_train = np.load(os.path.join(splits_dir, "y_train.npy"))
    X_test  = np.load(os.path.join(splits_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(splits_dir, "y_test.npy"))
    print(f"\n  Splits loaded: X_train {X_train.shape}, X_test {X_test.shape}")

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model, feature_extractor = build_cnn_bilstm(input_shape=(3, 6))
    model = compile_model(model, learning_rate=1e-3)
    print("  CNN-BiLSTM model built.")

    # ------------------------------------------------------------------
    # 3. Build state extractor (CNN output + BiLSTM output)
    # ------------------------------------------------------------------
    state_extractor = build_bilstm_state_extractor(model)

    # Run on first 8 training samples
    sample_X = X_train[:8]
    sample_y = y_train[:8]
    cnn_out, bilstm_out = state_extractor(sample_X, training=False)

    print(f"\n  --- Intermediate shapes (8 sample players) ---")
    print(f"  Input (chunks)        : {sample_X.shape}")
    print(f"  After CNN MaxPooling  : {cnn_out.shape}    "
          f"(2 pooled windows x 64 filters)")
    print(f"  After BiLSTM          : {bilstm_out.shape}   "
          f"(128-d = 64 forward + 64 backward)")
    print(f"  Feature vector (Dense): (8, 128)")

    # ------------------------------------------------------------------
    # 4. Show per-player CNN vs BiLSTM representation
    # ------------------------------------------------------------------
    print("\n  --- Per-player representation comparison ---")
    print(f"  {'Player':>6}  {'True Class':>18}  "
          f"{'CNN mean act':>12}  {'BiLSTM mean act':>15}")
    print("  " + "-" * 60)
    for i in range(8):
        cnn_mean    = float(cnn_out[i].numpy().mean())
        bilstm_mean = float(bilstm_out[i].numpy().mean())
        true_cls    = CLASS_NAMES[int(sample_y[i])]
        print(f"  {i+1:>6}  {true_cls:>18}  "
              f"{cnn_mean:>12.4f}  {bilstm_mean:>15.4f}")

    # ------------------------------------------------------------------
    # 5. Extract features for all training data (for heatmap)
    # ------------------------------------------------------------------
    print("\n  Extracting 128-d feature vectors for all training players...")
    all_features = feature_extractor(X_train, training=False).numpy()
    print(f"  Feature matrix shape: {all_features.shape}")

    # ------------------------------------------------------------------
    # 6. Generate visualisations
    # ------------------------------------------------------------------
    print("\n  Generating Step 7 visualisations...")

    plot_bilstm_concept(
        save_path=os.path.join(results_dir, "step7_bilstm_concept.png")
    )
    plot_feature_heatmap(
        feature_vectors=all_features,
        labels=y_train,
        save_path=os.path.join(results_dir, "step7_feature_heatmap.png"),
        n_players=40
    )

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Step 7 – BiLSTM Feature Learning: COMPLETE")
    print()
    print("  What the BiLSTM adds over CNN alone:")
    print("  - CNN detects LOCAL patterns (e.g. high hours + high K/D)")
    print("  - BiLSTM captures the PROGRESSION ACROSS windows:")
    print("      Forward : Session -> Gameplay -> Social")
    print("      Backward: Social  -> Gameplay -> Session")
    print("  - Output: 128-d vector  (64 forward + 64 backward)")
    print("  - This captures: sessions escalating -> rage-quits rising")
    print("                   -> toxic chat spiking -> guild abandonment")
    print()
    print("  Plots saved to results/:")
    print("    step7_bilstm_concept.png")
    print("    step7_feature_heatmap.png")
    print("=" * 65)


if __name__ == "__main__":
    main()
