import torch

# --- Data loading ---
words = open("names.txt").read().splitlines()

# --- Step 1: Explore the dataset ---
print(f"Total names: {len(words)}")
print(f"Shortest: {min(words, key=len)!r} ({len(min(words, key=len))} chars)")
print(f"Longest: {max(words, key=len)!r} ({len(max(words, key=len))} chars)")

chars = sorted(set("".join(words)))
print(f"Unique characters ({len(chars)}): {''.join(chars)}")

# --- Step 2: Build bigram frequency table ---
stoi = {ch: i + 1 for i, ch in enumerate(chars)}  # char -> index ('a'=1, ..., 'z'=26)
stoi["."] = 0  # special start/end token
itos = {i: ch for ch, i in stoi.items()}  # index -> char

N = torch.zeros((27, 27), dtype=torch.int32)  # bigram count matrix

for word in words:
    w = "." + word + "."
    for c1, c2 in zip(w, w[1:]):
        N[stoi[c1]][stoi[c2]] += 1

print(f"Total bigrams counted: {N.sum().item()}")

# --- Step 3: Sample from the bigram model ---
P = N.float()
P /= P.sum(dim=1, keepdims=True)
assert torch.allclose(P.sum(dim=1), torch.ones(27))

g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    ix = torch.multinomial(P[0], num_samples=1, generator=g).item()
    while ix != 0:
        out.append(itos[ix])
        ix = torch.multinomial(P[ix], num_samples=1, generator=g).item()

    print("".join(out))

# --- Step 4: Compute NLL loss ---
# TODO
