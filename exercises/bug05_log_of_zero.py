"""
Bug Hunt #05 — Log of Zero

GOAL: Compute the average negative log-likelihood (NLL) of a sequence under
a bigram model. NLL measures how "surprised" the model is by the data —
lower is better.

  NLL = -mean(log(P[context, next_char]))

RUN:  uv run python exercises/bug05_log_of_zero.py
FIND: Why is the loss inf?

HINT: What happens when log() gets a zero input?
"""

import torch

# A small 4x4 probability matrix (4 characters: a, b, c, d)
# Each row is a probability distribution over next characters
P = torch.tensor(
    [
        [0.7, 0.2, 0.1, 0.0],  # after 'a': never goes to 'd'
        [0.3, 0.3, 0.4, 0.0],  # after 'b': never goes to 'd'
        [0.1, 0.1, 0.5, 0.3],  # after 'c'
        [0.2, 0.3, 0.2, 0.3],  # after 'd'
    ]
)

P += 1e-6
P /= P.sum(dim=0, keepdim=True)

# A sequence to evaluate: a -> b -> a -> d -> a
contexts = [0, 1, 0, 3]  # a, b, a, d
next_chars = [1, 0, 3, 0]  # b, a, d, a

# Compute NLL
log_probs = []
for ctx, nxt in zip(contexts, next_chars):
    prob = P[ctx, nxt]
    log_prob = torch.log(prob)
    log_probs.append(log_prob)
    print(f"  P[{ctx},{nxt}] = {prob:.4f}, log = {log_prob:.4f}")

nll = -torch.tensor(log_probs).mean()
print(f"\nAverage NLL: {nll:.4f}")
print(f"Is loss finite? {torch.isfinite(nll).item()}")
