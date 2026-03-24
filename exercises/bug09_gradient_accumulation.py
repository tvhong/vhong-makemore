"""
Bug Hunt #09 — Gradient Accumulation

CONCEPT: Training a neural network works like this:
  1. Forward pass: compute predictions from inputs
  2. Compute loss (how wrong the predictions are)
  3. Backward pass: compute gradients (loss.backward())
  4. Update weights: w -= learning_rate * w.grad
  5. Repeat

The key detail: PyTorch ACCUMULATES gradients by default. Each call to
.backward() ADDS to the existing .grad tensor instead of replacing it.

GOAL: Train a simple linear model (y = Wx + b) to fit some data. The loss
should decrease smoothly over iterations.

RUN:  uv run python exercises/bug09_gradient_accumulation.py
FIND: Why does the loss oscillate wildly instead of decreasing?
"""

import torch

torch.manual_seed(42)

# Generate simple data: y = 2*x + 1 (with noise)
x = torch.randn(20, 1)
y_true = 2 * x + 1 + 0.1 * torch.randn(20, 1)

# Model parameters (random init)
W = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.1
losses = []

print(f"{x.shape=}")
print(f"{W.shape=}")
for i in range(50):
    # Forward pass
    y_pred = x @ W + b
    loss = ((y_pred - y_true) ** 2).mean()
    losses.append(loss.item())

    # Backward pass
    loss.backward()

    # Update weights (no_grad so PyTorch doesn't track these ops)
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad

    W.grad.zero_()
    b.grad.zero_()
    if i % 10 == 0:
        print(
            f"Step {i:3d}: loss = {loss.item():.4f}, W = {W.item():.3f}, b = {b.item():.3f}"
        )

print(
    f"\nFinal: W = {W.item():.3f} (expected ~2.0), b = {b.item():.3f} (expected ~1.0)"
)
print(f"Loss trend: {losses[0]:.2f} -> {losses[-1]:.2f} (should decrease)")
print(f"Loss stable? {losses[-1] < losses[0]}")
