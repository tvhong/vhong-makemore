"""
Bug Hunt #08 — Cross-Entropy Expects Logits

CONCEPT: Cross-entropy loss measures how well predicted probabilities match
the true labels. PyTorch's F.cross_entropy is designed to take RAW LOGITS
(before softmax) as input — it applies softmax internally for numerical
stability.

GOAL: We have a "perfect" model that assigns high logits to the correct
class. The loss should be very low (near 0) since the model is confident
and correct.

RUN:  uv run python exercises/bug08_cross_entropy_input.py
FIND: Why is the loss so high for a model that's clearly right?
"""

import torch
import torch.nn.functional as F

# 4 examples, 3 classes. Labels are the correct answers.
labels = torch.tensor([0, 1, 2, 0])

# The model is very confident and correct:
# Row 0: class 0 has the highest logit (5.0)
# Row 1: class 1 has the highest logit (5.0)
# etc.
logits = torch.tensor(
    [
        [5.0, -1.0, -1.0],  # strongly predicts class 0 ✓
        [-1.0, 5.0, -1.0],  # strongly predicts class 1 ✓
        [-1.0, -1.0, 5.0],  # strongly predicts class 2 ✓
        [5.0, -1.0, -1.0],  # strongly predicts class 0 ✓
    ]
)

# Convert logits to probabilities first, then compute loss
probs = logits.softmax(dim=1)
print("Probabilities (model is clearly confident and correct):")
print(probs)

loss = F.cross_entropy(logits, labels)

print(f"\nLoss: {loss.item():.4f}")
print("Expected for a confident correct model: ~0.0")
print("Something is wrong!" if loss.item() > 0.5 else "Loss looks reasonable")
