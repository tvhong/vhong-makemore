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

def normalize(N):
    P = N.float()
    P /= P.sum(dim=1, keepdim=True)
    assert torch.allclose(P.sum(dim=1), torch.ones(27))
    return P

def sample_name(P, itos, g):
    out = []
    ix = 0
    while True:
        ix = torch.multinomial(P[ix], num_samples=1, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    return "".join(out)


# --- Step 3: Sample from the bigram model ---
P = normalize(N)

g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    print(sample_name(P, itos, g))

def compute_nll(P, words, stoi):
    log_likelihood = 0.0
    n = 0
    for c1, c2 in generate_bigrams(words):
        log_likelihood += torch.log(P[stoi[c1]][stoi[c2]])
        n += 1
    return -log_likelihood / n


# --- Step 4: Compute NLL loss ---
nll = compute_nll(P, words, stoi)
print(f"NLL: {nll.item():.4f}")

# Sanity check: NLL should be less than uniform model (log(27) ≈ 3.30)
import math
uniform_nll = math.log(27)
assert nll < uniform_nll, f"NLL {nll:.4f} >= uniform {uniform_nll:.4f}, model is worse than random!"
print(f"Uniform NLL: {uniform_nll:.4f} (our model is better: {nll.item():.4f} < {uniform_nll:.4f})")

# Sanity check: uniform P should give NLL ≈ log(27)
P_uniform = torch.ones(27, 27) / 27
nll_uniform = compute_nll(P_uniform, words, stoi)
assert torch.isclose(nll_uniform, torch.tensor(uniform_nll), atol=1e-4)
print(f"Uniform P sanity check passed: {nll_uniform.item():.4f} ≈ {uniform_nll:.4f}")
