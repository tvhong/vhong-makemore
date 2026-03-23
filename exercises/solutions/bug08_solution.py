"""
Solution: Bug #08 — Cross-Entropy Expects Logits

BUG: F.cross_entropy() applies softmax internally. When you pass probabilities
     (which already went through softmax), softmax is applied TWICE:

     softmax([0.99, 0.003, 0.003]) ≈ [0.37, 0.31, 0.31]

     The second softmax "flattens" the confident predictions back toward
     uniform, making the loss much higher than it should be.

FIX: Pass raw logits directly to F.cross_entropy(), not softmax probabilities.

RULE OF THUMB: If you see softmax() right before cross_entropy(), one of them
is redundant. Either:
  - Use F.cross_entropy(logits, labels)  ← does softmax internally
  - Use F.nll_loss(log_softmax(logits), labels)  ← manual version
"""

import torch
import torch.nn.functional as F

labels = torch.tensor([0, 1, 2, 0])

logits = torch.tensor([
    [ 5.0, -1.0, -1.0],
    [-1.0,  5.0, -1.0],
    [-1.0, -1.0,  5.0],
    [ 5.0, -1.0, -1.0],
])

# FIX: Pass logits directly — F.cross_entropy applies softmax internally
loss = F.cross_entropy(logits, labels)

print(f"Loss: {loss.item():.4f}")
print(f"Expected for confident correct model: ~0.0")
