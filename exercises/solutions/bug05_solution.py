"""
Solution: Bug #05 — Log of Zero

BUG: P[0, 3] = 0.0. The sequence includes the transition a→d (context=0, next=3),
     so we compute log(0) = -inf. The mean of anything with -inf is -inf,
     giving NLL = inf.

FIX: Add Laplace smoothing to the probability matrix so no entry is exactly 0.
     P_smooth = (P + epsilon) and re-normalize, or better: smooth the counts
     before normalizing.
"""

import torch

P = torch.tensor([
    [0.7, 0.2, 0.1, 0.0],
    [0.3, 0.3, 0.4, 0.0],
    [0.1, 0.1, 0.5, 0.3],
    [0.2, 0.3, 0.2, 0.3],
])

# FIX: Add small smoothing and re-normalize so no probability is 0
P_smooth = P + 1e-6
P_smooth = P_smooth / P_smooth.sum(dim=1, keepdim=True)

contexts =   [0, 1, 0, 3]
next_chars = [1, 0, 3, 0]

log_probs = []
for ctx, nxt in zip(contexts, next_chars):
    prob = P_smooth[ctx, nxt]
    log_prob = torch.log(prob)
    log_probs.append(log_prob)
    print(f"  P[{ctx},{nxt}] = {prob:.6f}, log = {log_prob:.4f}")

nll = -torch.tensor(log_probs).mean()
print(f"\nAverage NLL: {nll:.4f}")
print(f"Is loss finite? {torch.isfinite(nll).item()}")
