"""
Solution: Bug #02 — Broadcasting Shape Mismatch

BUG: Two problems working together to create a silent failure:

  1. mean(dim=1) computes the mean of each ROW, not each COLUMN.
     We wanted mean(dim=0) for column means.

  2. Because the matrix is square (3x3), the result shape (3,) happens to
     broadcast without error against (3,3). If the matrix were non-square
     (e.g., 4x3), this would crash — but the square shape hides the bug.

     What happens: feature_means = [row0_mean, row1_mean, row2_mean]
     Broadcasting (3,3) - (3,) treats (3,) as a row vector, so it subtracts
     [row0_mean, row1_mean, row2_mean] from EVERY row. This is nonsensical.

FIX: Change dim=1 → dim=0 to compute column means.

LESSON: Square matrices are dangerous because they can hide dimension bugs
that would crash with non-square shapes. When debugging, try non-square
shapes to expose broadcasting errors.
"""

import torch

torch.manual_seed(42)

data = torch.randn(3, 3)

# FIX: dim=0 computes mean across rows → one mean per column
feature_means = data.mean(dim=0)

centered = data - feature_means

col_means = centered.mean(dim=0)
print("Original data:")
print(data)
print(f"\nMeans computed: {feature_means}")
print(f"Means shape:   {feature_means.shape}")
print(f"\nCentered data:")
print(centered)
print(f"\nColumn means (should be ~0): {col_means}")
print(f"All close to zero? {torch.allclose(col_means, torch.zeros(3), atol=1e-6)}")
