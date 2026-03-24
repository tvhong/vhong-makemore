"""
Bug Hunt #03 — Silent Dtype Truncation

GOAL: We have bigram counts and want to normalize each row into a probability
distribution. We pre-allocate the probability matrix, compute row probabilities,
and store them.

After normalization, every row of P should sum to 1.0.

RUN:  uv run python exercises/bug03_integer_division.py
FIND: Why are all the probabilities zero?
"""

import torch

# Bigram counts: 3 contexts, 4 possible next characters
N = torch.tensor(
    [
        [5, 3, 2, 0],
        [1, 7, 1, 1],
        [0, 2, 6, 2],
    ],
    dtype=torch.float32,
)

# Pre-allocate probability matrix
P = torch.zeros(3, 4, dtype=torch.float32)

# Normalize each row
for i in range(3):
    row_sum = N[i].sum()
    print(N[i] / row_sum)
    P[i] = N[i] / row_sum

print("Probability matrix:")
print(P)
print(f"\nRow sums (should be 1.0): {P.sum(dim=1)}")
print(f"P dtype: {P.dtype}")
