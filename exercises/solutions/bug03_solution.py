"""
Solution: Bug #03 — Silent Dtype Truncation

BUG: P is created with dtype=torch.int64. When you assign float values
     (the normalized probabilities, which are between 0 and 1) into an
     integer tensor, PyTorch silently truncates them to integers.
     Every value between 0 and 1 becomes 0.

     This is especially dangerous because there's no error or warning.

FIX: Create P with a float dtype: torch.zeros(3, 4, dtype=torch.float32)
     Or simply: torch.zeros(3, 4) since float32 is the default.
"""

import torch

N = torch.tensor([
    [5, 3, 2, 0],
    [1, 7, 1, 1],
    [0, 2, 6, 2],
], dtype=torch.float32)

# FIX: Use float32 (or just omit dtype, float32 is the default)
P = torch.zeros(3, 4, dtype=torch.float32)

for i in range(3):
    row_sum = N[i].sum()
    P[i] = N[i] / row_sum

print("Probability matrix:")
print(P)
print(f"\nRow sums (should be 1.0): {P.sum(dim=1)}")
print(f"P dtype: {P.dtype}")
