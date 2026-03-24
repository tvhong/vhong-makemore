import torch
import torch.nn.functional as F

# --- Setup ---

words = open("names.txt").read().splitlines()

chars = sorted(set("".join(words)))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}

# --- Build dataset with context window ---

block_size = 3

xs, ys = [], []
for word in words:
    w = "." * block_size + word + "."
    for i in range(len(w) - block_size):
        context = [stoi[c] for c in w[i : i + block_size]]
        target = stoi[w[i + block_size]]
        xs.append(context)
        ys.append(target)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

print(f"Dataset: {xs.shape[0]} examples, context shape: {xs.shape}")

# Sanity check: print first few examples
for i in range(5):
    context_str = "".join(itos[ix.item()] for ix in xs[i])
    print(f"  {context_str!r} -> {itos[ys[i].item()]!r}")

# --- Initialize parameters ---

emb_dim = 2
n_hidden = 100

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, emb_dim), generator=g, requires_grad=True)
W1 = torch.randn((block_size * emb_dim, n_hidden), generator=g, requires_grad=True)
b1 = torch.randn(n_hidden, generator=g, requires_grad=True)
W2 = (torch.randn((n_hidden, 27), generator=g) * 0.01).requires_grad_(True)
b2 = torch.zeros(27, requires_grad=True)

parameters = [C, W1, b1, W2, b2]

# --- Training loop ---

for epoch in range(100):
    # Forward pass
    emb = C[xs]
    emb_cat = emb.view(-1, block_size * emb_dim)
    h = torch.tanh(emb_cat @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, ys)

    # Backward pass
    loss.backward()

    # Update weights
    for p in parameters:
        p.data -= 0.1 * p.grad

    # Zero gradients
    for p in parameters:
        p.grad = None

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
