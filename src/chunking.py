# ============================================================
# chunking.py
# Step 4: Behaviour Chunking
# Adapts the paper's utterance-level chunking strategy to
# create three overlapping behavioural windows from the
# tabular gaming dataset.
# ============================================================

import numpy as np
import pandas as pd

# ============================================================
# Feature group definitions
# These correspond to the three behavioural modalities described
# in the project specification.
# ============================================================

SESSION_COLS = [
    'avg_daily_hours',
    'sessions_per_week',
    'late_night_fraction',
    'session_escalation_slope',
    'rage_quits_per_week'
]

GAMEPLAY_COLS = [
    'kd_ratio',
    'team_damage_incidents',
    'performance_volatility',
    'monthly_spending_usd'
]

SOCIAL_COLS = [
    'solo_play_fraction',
    'chat_msgs_per_session',
    'toxic_chat_score',
    'guild_changes_per_month'
]

# ============================================================
# Overlapping chunk definitions
#
# WHY OVERLAPPING?
# -  The paper's #1 insight is that local patterns must also
#    retain context from adjacent modalities.
# -  Overlap means the boundary features appear in two chunks,
#    giving the CNN an opportunity to detect cross-modality
#    patterns (e.g. high gaming hours + high K/D ratio).
#
# CHUNK STRUCTURE (each chunk has exactly 6 features):
#
#   Chunk 0 – Session:
#       5 session features + 1 gameplay overlap (kd_ratio)
#
#   Chunk 1 – Gameplay:
#       1 session overlap (rage_quits_per_week)
#       + 4 gameplay features
#       + 1 social overlap (solo_play_fraction)
#
#   Chunk 2 – Social:
#       2 gameplay overlap (performance_volatility, monthly_spending_usd)
#       + 4 social features
# ============================================================

CHUNK_0_COLS = [
    # session (5)
    'avg_daily_hours',
    'sessions_per_week',
    'late_night_fraction',
    'session_escalation_slope',
    'rage_quits_per_week',
    # overlap into gameplay (1)
    'kd_ratio',
]

CHUNK_1_COLS = [
    # overlap from session (1)
    'rage_quits_per_week',
    # gameplay (4)
    'kd_ratio',
    'team_damage_incidents',
    'performance_volatility',
    'monthly_spending_usd',
    # overlap into social (1)
    'solo_play_fraction',
]


CHUNK_2_COLS = [
    # overlap from gameplay (2)
    'performance_volatility',
    'monthly_spending_usd',
    # social (4)
    'solo_play_fraction',
    'chat_msgs_per_session',
    'toxic_chat_score',
    'guild_changes_per_month',
]

ALL_CHUNKS = [CHUNK_0_COLS, CHUNK_1_COLS, CHUNK_2_COLS]
CHUNK_NAMES = ['Session', 'Gameplay', 'Social']
CHUNK_SIZE = 6   # Each chunk has exactly 6 features


def create_behaviour_chunks(df_scaled):
    """
    Step 4 – Behaviour Chunking.

    Converts the scaled tabular DataFrame into a 3-D numpy array:

        Shape: (num_players, num_chunks, chunk_size)
               = (N, 3, 6)

    This is the input format expected by Conv1D in Keras:
        (batch, timesteps, features)

    The three 'timesteps' represent the sequence:
        Session  →  Gameplay  →  Social

    Parameters
    ----------
    df_scaled : pd.DataFrame
        The StandardScaler-transformed dataset from Step 3.

    Returns
    -------
    X_chunks : np.ndarray  shape (N, 3, 6)
        3-D chunk array ready for CNN input.
    chunk_arrays : list of np.ndarray
        Individual 2-D arrays for each modality:
            chunk_arrays[0] shape (N, 6) – Session
            chunk_arrays[1] shape (N, 6) – Gameplay
            chunk_arrays[2] shape (N, 6) – Social
    """
    print("\n--- Step 4: Behaviour Chunking ---")

    chunk_arrays = []
    for i, (cols, name) in enumerate(zip(ALL_CHUNKS, CHUNK_NAMES)):
        missing = [c for c in cols if c not in df_scaled.columns]
        if missing:
            raise ValueError(
                f"Chunk {i} ({name}) is missing columns in the DataFrame: {missing}"
            )
        arr = df_scaled[cols].values          # shape (N, 6)
        chunk_arrays.append(arr)
        print(f"  Chunk {i} ({name:8s}): {len(cols)} features  {arr.shape}")

    # Stack along axis=1 to form (N, 3, 6)
    X_chunks = np.stack(chunk_arrays, axis=1)  # (N, 3, 6)
    print(f"\n  Final chunk tensor shape : {X_chunks.shape}  "
          f"(players × windows × features)")

    # Quick sanity check
    assert X_chunks.shape[1] == 3,    "Expected 3 chunks (timesteps)."
    assert X_chunks.shape[2] == CHUNK_SIZE, f"Expected {CHUNK_SIZE} features per chunk."

    return X_chunks, chunk_arrays


def describe_chunks():
    """
    Print a human-readable summary of the three chunk definitions
    and their overlapping features.
    """
    print("\nBehaviour Chunk Summary")
    print("=" * 55)
    for i, (cols, name) in enumerate(zip(ALL_CHUNKS, CHUNK_NAMES)):
        print(f"\n  Chunk {i} – {name} Window  ({len(cols)} features):")
        for j, col in enumerate(cols):
            tag = ""
            # Mark features that appear in more than one chunk
            count = sum(col in c for c in ALL_CHUNKS)
            if count > 1:
                tag = "  [OVERLAP]"
            print(f"    [{j}] {col}{tag}")
    print("=" * 55)
