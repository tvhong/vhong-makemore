"""
Solution: Bug #06 — Softmax Dimension

BUG: softmax(dim=0) normalizes down each column (across the batch).
     This makes COLUMNS sum to 1, not rows. Each example's probabilities
     are entangled with other examples in the batch.

FIX: Use softmax(dim=1) to normalize across classes (columns) per example.
"""

import torch

torch.manual_seed(42)

logits = torch.randn(4, 3)

# FIX: dim=1 normalizes across classes (columns) per example
probs = logits.softmax(dim=1)

print("Logits (4 examples, 3 classes):")
print(logits)
print(f"\nProbabilities after softmax:")
print(probs)
print(f"\nRow sums (should be 1.0): {probs.sum(dim=1)}")
