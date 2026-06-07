"""
fusion.py
=========
Step 9 — Decision-Level Fusion.

The paper's best strategy: each modality's CNN-BiLSTM model
produces its own deep feature vector (128-d) independently.
These three vectors are concatenated → fed to the SVM.

This is called FEATURE-LEVEL FUSION of the deep representations
(which is the paper's "Decision-Level" when each model is trained
separately and only features are merged at the end).

Three modality streams:
  • Session   : Chunk 0  → CNN-BiLSTM backbone → 128-d vector
  • Gameplay  : Chunk 1  → CNN-BiLSTM backbone → 128-d vector
  • Social    : Chunk 2  → CNN-BiLSTM backbone → 128-d vector
  Concatenated           →                        384-d vector → SVM

Author : YOU (Person 2) — Step 9
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from src.cnn_bilstm import build_cnn_bilstm_backbone


# ── Build three separate modality backbones ────────────────────────────────
def build_modality_models(input_shape=(1, 6)):
    """
    Build one CNN-BiLSTM backbone per modality.
    Each sees only its own chunk (1 timestep × 6 features).

    Returns dict: {"session": model, "gameplay": model, "social": model}
    """
    models = {}
    for name in ["session", "gameplay", "social"]:
        models[name] = build_cnn_bilstm_backbone(
            input_shape=input_shape,
            conv_filters=64,
            kernel_size=1,       # kernel=1 since each chunk has 1 timestep
            lstm_units=64,
            dense_units=128,
            dropout_rate=0.3,
            name=f"backbone_{name}",
        )
    return models


# ── Extract features per modality ─────────────────────────────────────────
def extract_modality_features(
    backbone_models: dict,
    X_chunks: np.ndarray,          # (N, 3, 6)
    training: bool = False,
) -> np.ndarray:
    """
    Pass each chunk through its backbone → concatenate.

    Parameters
    ----------
    backbone_models : dict  {"session", "gameplay", "social"}
    X_chunks        : (N, 3, 6)  — full chunk tensor
    training        : bool       — True during training (dropout active)

    Returns
    -------
    fused_features : (N, 384)  — concatenated 3×128 vectors
    """
    feat_session  = backbone_models["session"] (X_chunks[:, 0:1, :], training=training)
    feat_gameplay = backbone_models["gameplay"](X_chunks[:, 1:2, :], training=training)
    feat_social   = backbone_models["social"]  (X_chunks[:, 2:3, :], training=training)

    # Concatenate: 128 + 128 + 128 = 384
    fused = np.concatenate(
        [feat_session.numpy(), feat_gameplay.numpy(), feat_social.numpy()],
        axis=1,
    )
    return fused.astype(np.float32)


# ── Unified training model (trains all 3 backbones jointly) ───────────────
def build_fusion_training_model(
    num_classes: int = 4,
    dense_units: int = 128,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Joint training model that feeds each chunk into its own backbone,
    concatenates the 384-d fused vector, and classifies with softmax.

    This model is used for training (Steps 11).
    After training, the three backbones are extracted and the SVM
    classifier replaces the Dense softmax head (Step 10).

    Input  : (batch, 3, 6)
    Output : (batch, 4)  — softmax probabilities
    """
    inp = keras.Input(shape=(3, 6), name="all_chunks")

    # Slice each modality window
    chunk_session  = layers.Lambda(lambda x: x[:, 0:1, :], name="slice_session") (inp)
    chunk_gameplay = layers.Lambda(lambda x: x[:, 1:2, :], name="slice_gameplay")(inp)
    chunk_social   = layers.Lambda(lambda x: x[:, 2:3, :], name="slice_social")  (inp)

    # Three separate backbones
    bb_session  = build_cnn_bilstm_backbone(
        input_shape=(1,6), kernel_size=1, name="bb_session")
    bb_gameplay = build_cnn_bilstm_backbone(
        input_shape=(1,6), kernel_size=1, name="bb_gameplay")
    bb_social   = build_cnn_bilstm_backbone(
        input_shape=(1,6), kernel_size=1, name="bb_social")

    feat_s = bb_session (chunk_session)    # (batch, 128)
    feat_g = bb_gameplay(chunk_gameplay)   # (batch, 128)
    feat_c = bb_social  (chunk_social)     # (batch, 128)

    # Concatenation — decision-level fusion point
    fused = layers.Concatenate(name="fusion_concat")([feat_s, feat_g, feat_c])
    # fused shape: (batch, 384)

    # Dropout before classifier
    x = layers.Dropout(dropout_rate, name="fusion_dropout")(fused)

    # Softmax head (replaced by SVM in Step 10)
    out = layers.Dense(num_classes, activation="softmax", name="softmax_head")(x)

    model = keras.Model(inputs=inp, outputs=out, name="FusionModel")
    return model, (bb_session, bb_gameplay, bb_social)


def print_fusion_summary():
    model, _ = build_fusion_training_model()
    print("=" * 56)
    print("  STEP 9 — DECISION-LEVEL FUSION MODEL")
    print("=" * 56)
    model.summary(line_length=56)
    print()
    print("  Fusion point: 128 (session) + 128 (gameplay)")
    print("              + 128 (social)  = 384-d fused vector")
    print("  This 384-d vector is the SVM input in Step 10.")
    print()
