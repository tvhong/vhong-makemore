import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn.functional as F
    import random

    return F, mo, random, torch


@app.cell
def _(mo, random, torch):
    # --- Setup ---
    words = open("names.txt").read().splitlines()

    chars = sorted(set("".join(words)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(stoi)

    # --- Build dataset with context window ---
    block_size = 3

    def build_dataset(words):
        xs, ys = [], []
        for word in words:
            w = "." * block_size + word + "."
            for i in range(len(w) - block_size):
                context = [stoi[c] for c in w[i : i + block_size]]
                target = stoi[w[i + block_size]]
                xs.append(context)
                ys.append(target)
        return torch.tensor(xs), torch.tensor(ys)

    # Split words: 80% train, 10% val, 10% test
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    xs_train, ys_train = build_dataset(words[:n1])
    xs_val, ys_val = build_dataset(words[n1:n2])
    xs_test, ys_test = build_dataset(words[n2:])

    mo.md(
        f"""
        **Dataset sizes:**
        - Train: {xs_train.shape[0]} examples
        - Val: {xs_val.shape[0]} examples
        - Test: {xs_test.shape[0]} examples
        """
    )
    return block_size, vocab_size, xs_train, ys_train


@app.cell
def _(block_size, mo, torch, vocab_size):
    # --- Initialize network ---
    emb_dim = 10
    n_hidden = 200

    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((vocab_size, emb_dim), generator=g, requires_grad=True)
    fan_in = block_size * emb_dim
    W1 = torch.randn((fan_in, n_hidden), generator=g, requires_grad=True) * fan_in**-0.5
    b1 = torch.zeros(n_hidden, requires_grad=True)
    W2 = torch.randn((n_hidden, vocab_size), generator=g, requires_grad=True) * 0.01
    b2 = torch.zeros(vocab_size, requires_grad=True)
    bn_gain = torch.ones((1, n_hidden), requires_grad=True)
    bn_bias = torch.zeros((1, n_hidden), requires_grad=True)

    parameters = [C, W1, b1, W2, b2, bn_gain, bn_bias]
    n_params = sum(p.numel() for p in parameters)

    mo.md(f"**Number of parameters:** {n_params}")
    return C, W1, W2, b1, b2, bn_bias, bn_gain, emb_dim


@app.cell
def _(
    C,
    F,
    W1,
    W2,
    b1,
    b2,
    block_size,
    bn_bias,
    bn_gain,
    emb_dim,
    mo,
    torch,
    vocab_size,
    xs_train,
    ys_train,
):
    # --- Step 1: Diagnose initial loss ---
    expected_loss = -torch.log(torch.tensor(1.0 / vocab_size)).item()

    emb = C[xs_train]
    emb_cat = emb.view(-1, block_size * emb_dim)
    h_preact = emb_cat @ W1 + b1

    # Batch normalization
    h_preact = bn_gain * (h_preact - h_preact.mean(0, keepdim=True)) / (h_preact.std(0, keepdim=True) + 1e-5) + bn_bias

    h = torch.tanh(h_preact)
    logits = h @ W2 + b2
    actual_loss = F.cross_entropy(logits, ys_train).item()

    mo.md(
        f"""
        **Step 1: Diagnose initial loss**

        Expected initial loss (uniform predictions): **{expected_loss:.4f}**

        Actual initial loss: **{actual_loss:.4f}**
        """
    )
    return (h,)


@app.cell
def _(h, mo, torch):
    import matplotlib.pyplot as plt

    # --- Histogram of tanh activations ---
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    h_flat = h.detach().view(-1).numpy()
    ax1.hist(h_flat, bins=50, color="steelblue", edgecolor="black")
    ax1.set_title("Hidden layer activation distribution")
    ax1.set_xlabel("tanh output")
    ax1.set_ylabel("count")

    # --- Saturation map ---
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    saturated = (torch.abs(h.detach()) > 0.99).float().numpy()
    ax2.imshow(saturated[:10, :50].T, cmap="gray_r", aspect="auto")
    ax2.set_title("Saturation map (black = |tanh| > 0.99)")
    ax2.set_xlabel("example")
    ax2.set_ylabel("neuron")

    frac_saturated = saturated.mean() * 100
    mo.vstack([
        fig1,
        fig2,
        mo.md(f"**{frac_saturated:.1f}%** of activations are saturated (|tanh| > 0.99)"),
    ])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
