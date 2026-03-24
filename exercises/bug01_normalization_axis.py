"""
Bug Hunt #01 — Row Normalization

GOAL: We have a 3x4 frequency count matrix. Each row represents a different
context (e.g., which character came before), and each column represents the
next character. We want to convert each row into a probability distribution
by dividing by the row sum.

After normalization, every ROW should sum to 1.0.

RUN:  uv run python exercises/bug01_normalization_axis.py
FIND: Why don't the rows sum to 1?
"""

import torch

# Frequency counts: 3 contexts, 4 possible next characters
N = torch.tensor(
    [
        [5, 3, 2, 0],
        [1, 7, 1, 1],
        [0, 2, 6, 2],
    ],
    dtype=torch.float32,
)

# Normalize each row to get probabilities
P = N / N.sum(dim=1, keepdim=True)

# Check: every row should sum to 1.0
row_sums = P.sum(dim=1)
print("Probability matrix:")
print(P)
print()
print(f"Row sums (should all be 1.0): {row_sums}")
print(f"All rows sum to 1? {torch.allclose(row_sums, torch.ones(3))}")
