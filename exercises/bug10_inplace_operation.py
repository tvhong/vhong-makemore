"""
Bug Hunt #10 — Weight Update Kills Gradients

CONCEPT: In PyTorch, tensors with requires_grad=True are "leaf" tensors
that track gradients. During the weight update step, we need to modify
the tensor's data WITHOUT creating a new tensor, because a new tensor
would lose the requires_grad property.

    W = W - lr * W.grad   # Creates NEW tensor (bad!)
    W -= lr * W.grad      # Modifies in-place (good... if done in no_grad block)

GOAL: Train a simple linear model (y = Wx) for 20 steps. The loss should
decrease and W should converge to the true value.

RUN:  uv run python exercises/bug10_inplace_operation.py
FIND: Why does the model only learn for 1 step and then stop?
"""

import torch

torch.manual_seed(42)

# True relationship: y = 3x
x = torch.randn(20, 1)
y_true = 3 * x

# Weight to learn (should converge to 3.0)
W = torch.randn(1, 1, requires_grad=True)

lr = 0.1

for i in range(20):
    # Forward pass
    y_pred = x @ W
    loss = ((y_pred - y_true) ** 2).mean()

    # Backward pass
    if not W.requires_grad:
        print(f"  ⚠ W.requires_grad is False! backward() will fail.")
        print(f"  Training stopped early at step {i}.")
        break
    loss.backward()

    # Update weights
    with torch.no_grad():
        W = W - lr * W.grad

    # Zero gradients for next iteration
    if W.grad is not None:
        W.grad.zero_()

    if i % 5 == 0:
        print(f"Step {i:3d}: loss = {loss.item():.4f}, W = {W.item():.3f}, requires_grad = {W.requires_grad}")

print(f"\nFinal W = {W.item():.3f} (expected ~3.0)")
