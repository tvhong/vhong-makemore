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
        context = [stoi[c] for c in w[i:i + block_size]]
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
