"""
Solution: Bug #09 — Gradient Accumulation

BUG: PyTorch accumulates gradients by default. Without zeroing gradients
     before each backward(), W.grad grows every iteration:
       iter 0: grad = g0
       iter 1: grad = g0 + g1
       iter 2: grad = g0 + g1 + g2
     The effective learning rate grows without bound, causing wild oscillation.

FIX: Zero gradients before each backward pass:
     W.grad.zero_() and b.grad.zero_()
     (Must check .grad is not None for the first iteration.)
"""

import torch

torch.manual_seed(42)

x = torch.randn(20, 1)
y_true = 2 * x + 1 + 0.1 * torch.randn(20, 1)

W = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.1
losses = []

for i in range(50):
    y_pred = x @ W + b
    loss = ((y_pred - y_true) ** 2).mean()
    losses.append(loss.item())

    # FIX: Zero gradients before backward
    if W.grad is not None:
        W.grad.zero_()
        b.grad.zero_()

    loss.backward()

    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad

    if i % 10 == 0:
        print(f"Step {i:3d}: loss = {loss.item():.4f}, W = {W.item():.3f}, b = {b.item():.3f}")

print(f"\nFinal: W = {W.item():.3f} (expected ~2.0), b = {b.item():.3f} (expected ~1.0)")
print(f"Loss trend: {losses[0]:.2f} -> {losses[-1]:.2f}")
print(f"Loss stable? {losses[-1] < losses[0]}")
