# Tensor Bug Hunting Exercises

10 short Python scripts, each with one subtle bug that produces wrong results
(but usually doesn't crash). Your job: find the bug, explain what's wrong, and fix it.

## How to use

```bash
uv run python exercises/bug01_normalization_axis.py
```

Read the docstring at the top of each file — it explains what the code is
supposed to do. Run it, look at the output, and figure out why it's wrong.

Solutions are in `exercises/solutions/`.

## Exercises

### Block 1 — Tensor & Probability Bugs (things you already know)

| # | File | Topic | Difficulty |
|---|------|-------|------------|
| 1 | `bug01_normalization_axis.py` | Wrong axis in normalization | Easy |
| 2 | `bug02_broadcasting_shape.py` | Broadcasting shape mismatch | Easy |
| 3 | `bug03_integer_division.py` | Silent dtype truncation | Easy |
| 4 | `bug04_smoothing_broadcast.py` | Smoothing with wrong shape | Medium |
| 5 | `bug05_log_of_zero.py` | Log of zero probability | Medium |

### Block 2 — Neural Network Bugs (preview of upcoming concepts)

Each file includes a brief explanation of the concept involved.

| # | File | Topic | Difficulty |
|---|------|-------|------------|
| 6 | `bug06_softmax_dim.py` | Softmax on wrong dimension | Easy |
| 7 | `bug07_onehot_dtype.py` | One-hot dtype mismatch | Easy |
| 8 | `bug08_cross_entropy_input.py` | Logits vs probabilities | Medium |
| 9 | `bug09_gradient_accumulation.py` | Forgetting to zero gradients | Medium |
| 10 | `bug10_inplace_operation.py` | Weight update kills gradients | Hard |
