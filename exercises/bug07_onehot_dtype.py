"""
Bug Hunt #07 — One-Hot Encoding Dtype

CONCEPT: One-hot encoding represents a categorical value (like character index 3
out of 5 possible characters) as a vector of zeros with a single 1:
    index 3 → [0, 0, 0, 1, 0]

In a neural network, we multiply one-hot vectors by a weight matrix to get
the output. This is a matrix multiplication (@ operator).

GOAL: One-hot encode a batch of character indices and multiply by a weight
matrix to get output logits.

RUN:  uv run python exercises/bug07_onehot_dtype.py
FIND: Why does the matrix multiplication fail?
"""

import torch
import torch.nn.functional as F

torch.manual_seed(42)

# Character indices for a batch of 5 inputs (values 0-3, 4 possible chars)
indices = torch.tensor([0, 3, 1, 2, 1])
num_classes = 4

# One-hot encode
one_hot = F.one_hot(indices, num_classes=num_classes)
print(f"One-hot shape: {one_hot.shape}")
print(f"One-hot dtype: {one_hot.dtype}")
print(f"One-hot:\n{one_hot}")

# Weight matrix: maps from 4 input features to 3 output classes
W = torch.randn(4, 3)
print(f"\nWeight matrix dtype: {W.dtype}")

# Forward pass: multiply one-hot inputs by weights
logits = one_hot @ W
print(f"\nLogits:\n{logits}")
