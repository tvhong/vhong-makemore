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

for epoch in range(200):
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

# --- Step 7: Compare neural net to counting model ---

# Counting model: build bigram table and normalize
N = torch.zeros((27, 27), dtype=torch.int32)
for word in words:
    w = "." + word + "."
    for c1, c2 in zip(w, w[1:]):
        N[stoi[c1]][stoi[c2]] += 1
P_count = N.float()
P_count /= P_count.sum(dim=1, keepdim=True)

# Neural net model: softmax of weight matrix
P_nn = torch.softmax(W.data, dim=1)

# Compare
diff = (P_nn - P_count).abs()
print(f"\nMax absolute difference: {diff.max().item():.4f}")
print(f"Mean absolute difference: {diff.mean().item():.4f}")
