import torch
import torch.nn.functional as F

# --- Setup ---

words = open("names.txt").read().splitlines()

chars = sorted(set("".join(words)))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}

# --- Build training data ---

xs, ys = [], []
for word in words:
    w = "." + word + "."
    for c1, c2 in zip(w, w[1:]):
        xs.append(stoi[c1])
        ys.append(stoi[c2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)

# --- Forward pass ---

# One-hot encode inputs
xenc = F.one_hot(xs, num_classes=27).float()
print(xs.shape, xenc.shape)

# Initialize weights
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# --- Training loop ---

for epoch in range(50):
    # Forward pass
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)

    # Loss: negative log likelihood
    loss = -probs[range(len(ys)), ys].log().mean()

    # Backward pass
    loss.backward()

    # Update weights
    W.data -= 50 * W.grad

    W.grad.zero_()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# Check: print probability for first 5 examples
for i in range(5):
    print(f"Input: {itos[xs[i].item()]!r} -> Target: {itos[ys[i].item()]!r}")
    print(f"  P(target) = {probs[i, ys[i]].item():.4f}")
