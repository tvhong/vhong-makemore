import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import random

    return mo, random, torch


@app.cell
def _(mo, random, torch):
    # --- Setup ---
    words = open("names.txt").read().splitlines()

    chars = sorted(set("".join(words)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(stoi)

    block_size = 3

    def build_dataset(words):
        xs, ys = [], []
        for w in words:
            context = [0] * block_size
            for ch in w + ".":
                ix = stoi[ch]
                xs.append(context)
                ys.append(ix)
                context = context[1:] + [ix]
        return torch.tensor(xs), torch.tensor(ys)

    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    Xtr, Ytr = build_dataset(words[:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xte, Yte = build_dataset(words[n2:])

    mo.md(
        f"""
        **Dataset sizes:**
        - Train: {Xtr.shape[0]} examples
        - Val: {Xdev.shape[0]} examples
        - Test: {Xte.shape[0]} examples
        """
    )
    return Xtr, Ytr, block_size, vocab_size


@app.cell
def _(torch):
    # Utility function to compare manual gradients to PyTorch gradients
    def cmp(s, dt, t):
        _ex = torch.all(dt == t.grad).item()
        _app = torch.allclose(dt, t.grad)
        _maxdiff = (dt - t.grad).abs().max().item()
        print(f"{s:15s} | exact: {str(_ex):5s} | approximate: {str(_app):5s} | maxdiff: {_maxdiff}")

    return (cmp,)


@app.cell
def _(block_size, mo, torch, vocab_size):
    # --- Initialize parameters ---
    n_embd = 10
    n_hidden = 64

    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((vocab_size, n_embd), generator=g)
    # Layer 1
    W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5 / 3) / ((n_embd * block_size) ** 0.5)
    b1 = torch.randn(n_hidden, generator=g) * 0.1
    # Layer 2
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
    b2 = torch.randn(vocab_size, generator=g) * 0.1
    # BatchNorm parameters
    bngain = torch.randn((1, n_hidden)) * 0.1 + 1.0
    bnbias = torch.randn((1, n_hidden)) * 0.1

    # Note: initializing in non-standard ways on purpose
    # so that all-zeros doesn't mask incorrect backward pass

    parameters = [C, W1, b1, W2, b2, bngain, bnbias]
    for _p in parameters:
        _p.requires_grad = True

    _n_params = sum(p.nelement() for p in parameters)
    mo.md(f"**Parameters:** {_n_params}")
    return C, W1, W2, b1, b2, bnbias, bngain


@app.cell
def _(C, W1, W2, Xtr, Ytr, b1, b2, bnbias, bngain, torch):
    # --- Construct minibatch and forward pass (chunkated into atomic ops) ---
    batch_size = 32
    n = batch_size

    _g = torch.Generator().manual_seed(2147483647)
    _ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=_g)
    Xb, Yb = Xtr[_ix], Ytr[_ix]

    # Forward pass — broken into atomic steps for backprop practice
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    # Linear layer 1
    hprebn = embcat @ W1 + b1
    # BatchNorm layer
    bnmeani = 1 / n * hprebn.sum(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff ** 2
    bnvar = 1 / (n - 1) * (bndiff2).sum(0, keepdim=True)  # Bessel's correction
    bnvar_inv = (bnvar + 1e-5) ** -0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    # Non-linearity
    h = torch.tanh(hpreact)
    # Linear layer 2
    logits = h @ W2 + b2
    # Cross entropy loss (same as F.cross_entropy(logits, Yb))
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdims=True)
    counts_sum_inv = counts_sum ** -1
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss = -logprobs[range(n), Yb].mean()

    # PyTorch backward pass (for verification)
    for _p in [C, W1, b1, W2, b2, bngain, bnbias]:
        _p.grad = None
    for _t in [
        logprobs, probs, counts, counts_sum, counts_sum_inv,
        norm_logits, logit_maxes, logits, h, hpreact, bnraw,
        bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
        embcat, emb,
    ]:
        _t.retain_grad()
    loss.backward()

    print(f"loss: {loss.item():.4f}")
    return (
        Yb,
        bnraw,
        counts,
        counts_sum,
        counts_sum_inv,
        h,
        hpreact,
        logit_maxes,
        logits,
        logprobs,
        n,
        norm_logits,
        probs,
    )


@app.cell
def _(
    W2,
    Yb,
    b2,
    bnbias,
    bngain,
    bnraw,
    cmp,
    counts,
    counts_sum,
    counts_sum_inv,
    h,
    hpreact,
    logit_maxes,
    logits,
    logprobs,
    n,
    norm_logits,
    probs,
    torch,
):
    # ============================================================
    # Exercise 1: backprop through the whole thing manually
    # backpropagating through every variable one at a time
    # ============================================================

    # loss = -logprobs[range(n), Yb].mean()
    dlogprobs = torch.zeros_like(logprobs)
    dlogprobs[range(n), Yb] = -1.0 / n
    cmp("logprobs", dlogprobs, logprobs)

    # logprobs = probs.log()
    dprobs = (1.0 / probs) * dlogprobs
    cmp("probs", dprobs, probs)

    # probs = counts * counts_sum_inv
    dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
    dcounts = counts_sum_inv * dprobs
    cmp("counts_sum_inv", dcounts_sum_inv, counts_sum_inv)

    # counts_sum_inv = counts_sum ** -1
    dcounts_sum = (-counts_sum ** -2) * dcounts_sum_inv
    cmp("counts_sum", dcounts_sum, counts_sum)

    # counts_sum = counts.sum(1, keepdims=True)
    dcounts += torch.ones_like(counts) * dcounts_sum
    cmp("counts", dcounts, counts)

    # counts = norm_logits.exp()
    dnorm_logits = counts * dcounts
    cmp("norm_logits", dnorm_logits, norm_logits)

    # norm_logits = logits - logit_maxes
    dlogits = dnorm_logits.clone()
    dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
    cmp("logit_maxes", dlogit_maxes, logit_maxes)

    # logit_maxes = logits.max(1, keepdim=True).values
    dlogits += (logits == logit_maxes) * dlogit_maxes
    cmp("logits", dlogits, logits)

    # logits = h @ W2 + b2
    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)
    cmp("h", dh, h)
    cmp("W2", dW2, W2)
    cmp("b2", db2, b2)

    # h = torch.tanh(hpreact)
    dhpreact = (1 - h ** 2) * dh
    cmp("hpreact", dhpreact, hpreact)

    # hpreact = bngain * bnraw + bnbias
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnraw = bngain * dhpreact
    dbnbias = dhpreact.sum(0, keepdim=True)
    cmp("bngain", dbngain, bngain)
    cmp("bnraw", dbnraw, bnraw)
    cmp("bnbias", dbnbias, bnbias)

    # YOUR CODE: continue backwards from here...
    return


@app.cell
def _():
    # ============================================================
    # Exercise 2: backprop through cross_entropy in one go
    # Derive the gradient of the loss w.r.t. logits analytically
    # ============================================================
    # YOUR CODE HERE
    return


@app.cell
def _():
    # ============================================================
    # Exercise 3: backprop through batchnorm in one go
    # Derive dhprebn given dhpreact analytically
    # ============================================================
    # YOUR CODE HERE
    return


@app.cell
def _():
    # ============================================================
    # Exercise 4: putting it all together!
    # Train the MLP with your own manual backward pass
    # ============================================================
    # YOUR CODE HERE
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
