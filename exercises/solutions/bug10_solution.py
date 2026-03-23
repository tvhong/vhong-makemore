"""
Solution: Bug #10 — Weight Update Kills Gradients

BUG: `W = W - lr * W.grad` creates a BRAND NEW tensor and assigns it to W.
     The new tensor has requires_grad=False (it was created in a no_grad block).
     From step 2 onward, backward() can't compute gradients for W because
     it's no longer a leaf tensor that tracks gradients.

     You can see this in the output: requires_grad switches from True to False
     after the first step.

FIX: Use the in-place subtraction operator:
     W -= lr * W.grad

     Or equivalently:
     W.data -= lr * W.grad

     Both modify the existing tensor's data without creating a new tensor,
     so requires_grad stays True.

LESSON: `a = a - b` and `a -= b` are NOT the same in PyTorch!
  - `a = a - b` → creates new tensor, assigns to variable name
  - `a -= b` → modifies existing tensor in-place
"""

import torch

torch.manual_seed(42)

x = torch.randn(20, 1)
y_true = 3 * x

W = torch.randn(1, 1, requires_grad=True)
lr = 0.1

for i in range(20):
    y_pred = x @ W
    loss = ((y_pred - y_true) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        # FIX: Use -= (in-place) instead of = ... - ... (creates new tensor)
        W -= lr * W.grad

    W.grad.zero_()

    if i % 5 == 0:
        print(f"Step {i:3d}: loss = {loss.item():.4f}, W = {W.item():.3f}, requires_grad = {W.requires_grad}")

print(f"\nFinal W = {W.item():.3f} (expected ~3.0)")
