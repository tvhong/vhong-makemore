"""
Bug Hunt #04 — Smoothing Goes Wrong

GOAL: We have a 3x3 bigram count matrix. Before converting to probabilities,
we add Laplace smoothing (add a small constant to every cell) so no
probability is zero. Then we normalize each row to sum to 1.

After smoothing + normalization, each row should be a valid probability
distribution, and the smoothing should be UNIFORM (same amount added to
every cell).

RUN:  uv run python exercises/bug04_smoothing_broadcast.py
FIND: Why are the probabilities skewed? (Compare smoothed vs expected.)

HINT: Check what shape the smoothing constant has and how it broadcasts.
"""

import torch

# Bigram counts: 3 contexts, 3 next characters
N = torch.tensor(
    [
        [10, 0, 0],
        [0, 5, 5],
        [3, 3, 4],
    ],
    dtype=torch.float32,
)

# Laplace smoothing: add 1 to every cell
# We want to add 1 uniformly everywhere
smoothing = 1
N_smooth = N + smoothing

# Normalize rows to get probabilities
P = N_smooth / N_smooth.sum(dim=1, keepdim=True)

print("Original counts:")
print(N)
print("\nSmoothing added:")
print(smoothing)
print("\nSmoothed counts:")
print(N_smooth)
print("\nProbabilities:")
print(P)
print(f"\nRow sums: {P.sum(dim=1)}")

# For row 0 ([10,0,0]), after adding 1 to each, we expect [11,1,1]/13
# = [0.846, 0.077, 0.077]
print("\nRow 0 expected (uniform smoothing): [0.846, 0.077, 0.077]")
print(f"Row 0 actual: {P[0].tolist()}")
