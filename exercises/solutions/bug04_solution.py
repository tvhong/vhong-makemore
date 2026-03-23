"""
Solution: Bug #04 — Smoothing Goes Wrong

BUG: smoothing = torch.tensor([1, 2, 3]) has shape (3,).
     When added to N (shape 3x3), it broadcasts: each COLUMN gets a different
     smoothing value. Column 0 gets +1, column 1 gets +2, column 2 gets +3.
     This biases the probabilities toward later columns.

FIX: Use a scalar for uniform smoothing: smoothing = 1 (or torch.tensor(1.0)).
"""

import torch

N = torch.tensor([
    [10, 0, 0],
    [0, 5, 5],
    [3, 3, 4],
], dtype=torch.float32)

# FIX: Scalar smoothing applies uniformly to all cells
smoothing = 1
N_smooth = N + smoothing

P = N_smooth / N_smooth.sum(dim=1, keepdim=True)

print("Original counts:")
print(N)
print(f"\nSmoothing: {smoothing} (uniform scalar)")
print(f"\nSmoothed counts:")
print(N_smooth)
print(f"\nProbabilities:")
print(P)
print(f"\nRow 0 expected: [0.846, 0.077, 0.077]")
print(f"Row 0 actual:   {[f'{x:.3f}' for x in P[0].tolist()]}")
