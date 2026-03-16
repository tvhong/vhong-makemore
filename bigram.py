import torch

# --- Data loading ---
words = open("names.txt").read().splitlines()

def explore_dataset(words):
    print(f"Total names: {len(words)}")
    print(f"Shortest: {min(words, key=len)!r} ({len(min(words, key=len))} chars)")
    print(f"Longest: {max(words, key=len)!r} ({len(max(words, key=len))} chars)")
    chars = sorted(set("".join(words)))
    print(f"Unique characters ({len(chars)}): {''.join(chars)}")
    return chars


def generate_bigrams(words):
    for word in words:
        w = "." + word + "."
        yield from zip(w, w[1:])


def build_bigram_table(words, stoi):
    N = torch.zeros((27, 27), dtype=torch.int32)
    for c1, c2 in generate_bigrams(words):
        N[stoi[c1]][stoi[c2]] += 1
    return N


chars = explore_dataset(words)

stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}

N = build_bigram_table(words, stoi)
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
log_likelihood = 0.0
n = 0  # total number of bigrams

# TODO(human): Loop through all bigrams, sum up log probabilities, compute NLL
for word in words:
    w = "." + word + "."
    for c1, c2 in zip(w, w[1:]):
        log_likelihood += torch.log(P[stoi[c1]][stoi[c2]])
        n += 1

log_likelihood /= n
log_likelihood *= -1

print(f"{log_likelihood=}")
