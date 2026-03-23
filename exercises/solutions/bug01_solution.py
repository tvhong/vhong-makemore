"""
Solution: Bug #01 — Row Normalization

BUG: N.sum(dim=0) sums across rows (collapsing axis 0), giving column totals.
     We needed N.sum(dim=1, keepdim=True) to sum across columns per row.

FIX: Change dim=0 → dim=1, and add keepdim=True so the shape broadcasts.

WHY keepdim=True?
  N has shape (3, 4).
  N.sum(dim=1) gives shape (3,) — a 1D vector.
  N / (3,) tries to broadcast: (3,4) / (3,) → broadcasts (3,) along columns,
  which divides each COLUMN by the same number. Wrong!

  N.sum(dim=1, keepdim=True) gives shape (3, 1).
  N / (3,1) broadcasts correctly: each row is divided by its own sum.
"""

import torch

N = torch.tensor([
    [5, 3, 2, 0],
    [1, 7, 1, 1],
    [0, 2, 6, 2],
], dtype=torch.float32)

# FIX: dim=1 (sum across columns), keepdim=True (keep shape for broadcasting)
P = N / N.sum(dim=1, keepdim=True)

row_sums = P.sum(dim=1)
print("Probability matrix:")
print(P)
print(f"\nRow sums: {row_sums}")
print(f"All rows sum to 1? {torch.allclose(row_sums, torch.ones(3))}")
