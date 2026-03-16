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

# --- Step 3: Sample from the bigram model ---
# TODO

# --- Step 4: Compute NLL loss ---
# TODO
