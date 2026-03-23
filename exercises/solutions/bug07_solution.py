"""
Solution: Bug #07 — One-Hot Encoding Dtype

BUG: F.one_hot() returns a tensor of dtype int64 (Long). Matrix multiplication
     (@ operator) requires both tensors to have the same dtype. W is float32,
     one_hot is int64 → RuntimeError.

FIX: Convert one-hot to float before multiplying: one_hot.float() @ W
"""

import torch
import torch.nn.functional as F

torch.manual_seed(42)

indices = torch.tensor([0, 3, 1, 2, 1])
num_classes = 4

one_hot = F.one_hot(indices, num_classes=num_classes)
W = torch.randn(4, 3)

# FIX: Convert one-hot from int64 to float32
logits = one_hot.float() @ W

print(f"One-hot dtype: {one_hot.dtype}")
print(f"After .float(): {one_hot.float().dtype}")
print(f"\nLogits:\n{logits}")
