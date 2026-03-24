"""
Bug Hunt #06 — Softmax Dimension

CONCEPT: In a neural network, the final layer outputs raw scores called
"logits" — one per class. Softmax converts logits into probabilities:

    softmax(x_i) = exp(x_i) / sum(exp(x_j))

For a batch of examples, each ROW is one example and each COLUMN is a class.
We want softmax to normalize across classes (columns), so each ROW sums to 1.

GOAL: Apply softmax to a batch of logits so each example (row) becomes a
valid probability distribution.

RUN:  uv run python exercises/bug06_softmax_dim.py
FIND: Why don't the rows sum to 1?
"""

import torch

torch.manual_seed(42)

# Batch of 4 examples, 3 classes each
# Each row = one example's raw logit scores
logits = torch.randn(4, 3)

# Apply softmax to get probabilities per example
probs = logits.softmax(dim=1)

print("Logits (4 examples, 3 classes):")
print(logits)
print("\nProbabilities after softmax:")
print(probs)
print(f"\nRow sums (should be 1.0): {probs.sum(dim=1)}")
print(f"Col sums (should NOT be 1.0): {probs.sum(dim=0)}")
