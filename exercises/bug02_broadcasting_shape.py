"""
Bug Hunt #02 — Broadcasting Shape Mismatch

GOAL: We have a 3x3 matrix of data (3 examples, 3 features). We want to
center each feature (column) by subtracting the per-feature mean. The mean
should be computed across the batch dimension (dim=0).

After centering, the mean of each COLUMN should be ~0.

RUN:  uv run python exercises/bug02_broadcasting_shape.py
FIND: Why aren't the column means zero after centering?

HINT: Print the shapes of intermediate tensors. What does the mean vector
      actually contain?
"""

import torch

torch.manual_seed(42)

# 3 data points, 3 features each
data = torch.randn(3, 3)

# Compute mean of each feature (across the batch dimension)
# We want the mean of each COLUMN
feature_means = data.mean(dim=0)  # (3,)

# Center the data
centered = data - feature_means

# Check: column means should be ~0
col_means = centered.mean(dim=0)
print("Original data:")
print(data)
print(f"\nMeans computed: {feature_means}")
print(f"Means shape:   {feature_means.shape}")
print("\nCentered data:")
print(centered)
print(f"\nColumn means (should be ~0): {col_means}")
print(f"All close to zero? {torch.allclose(col_means, torch.zeros(3), atol=1e-6)}")
