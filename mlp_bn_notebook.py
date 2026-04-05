import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn.functional as F
    import random
    import matplotlib.pyplot as plt

    return F, mo, plt, random, torch


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
    return block_size, vocab_size, xs_train, xs_val, ys_train, ys_val


@app.cell
def _(torch):
    # --- Module classes ---

    class Linear:
        def __init__(self, fan_in, fan_out, bias=True, generator=None):
            self.weight = torch.randn((fan_in, fan_out), generator=generator) * fan_in**-0.5
            self.bias = torch.zeros(fan_out) if bias else None

        def __call__(self, x):
            self.out = x @ self.weight
            if self.bias is not None:
                self.out += self.bias
            return self.out

        def parameters(self):
            return [self.weight] + ([] if self.bias is None else [self.bias])

    class BatchNorm1d:
        def __init__(self, dim, eps=1e-5, momentum=0.001):
            self.eps = eps
            self.momentum = momentum
            self.training = True
            # Learnable parameters
            self.gamma = torch.ones(dim)
            self.beta = torch.zeros(dim)
            # Running stats (not learned)
            self.running_mean = torch.zeros(dim)
            self.running_var = torch.ones(dim)

        def __call__(self, x):
            if self.training:
                mean = x.mean(0, keepdim=True)
                var = x.var(0, keepdim=True)
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            self.out = self.gamma * x_hat + self.beta

            # Update running stats
            if self.training:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            return self.out

        def parameters(self):
            return [self.gamma, self.beta]

    class Tanh:
        def __call__(self, x):
            self.out = torch.tanh(x)
            return self.out

        def parameters(self):
            return []

    return BatchNorm1d, Linear, Tanh


@app.cell
def _(BatchNorm1d, Linear, Tanh, block_size, mo, torch, vocab_size):
    # --- Build deep network ---
    emb_dim = 10
    n_hidden = 200
    n_layers = 3

    g_init = torch.Generator().manual_seed(2147483647)
    C = torch.randn((vocab_size, emb_dim), generator=g_init)

    layers = []
    # First hidden layer: input is flattened embeddings
    layers.append(Linear(block_size * emb_dim, n_hidden, bias=False, generator=g_init))
    layers.append(BatchNorm1d(n_hidden))
    layers.append(Tanh())
    # Additional hidden layers
    for _ in range(n_layers - 1):
        layers.append(Linear(n_hidden, n_hidden, bias=False, generator=g_init))
        layers.append(BatchNorm1d(n_hidden))
        layers.append(Tanh())
    # Output layer
    layers.append(Linear(n_hidden, vocab_size, generator=g_init))

    # Collect all parameters
    C.requires_grad = True
    parameters = [C]
    for layer_init in layers:
        for p_init in layer_init.parameters():
            p_init.requires_grad = True
            parameters.append(p_init)

    n_params = sum(p.numel() for p in parameters)
    mo.md(f"**Deep MLP: {n_layers} hidden layers, {n_params} parameters**")
    return C, emb_dim, layers, parameters


@app.cell
def _(
    C,
    F,
    block_size,
    emb_dim,
    layers,
    mo,
    parameters,
    plt,
    torch,
    xs_train,
    ys_train,
):
    # --- Training loop ---
    _max_steps = 2000
    _batch_size = 32
    _lr = 0.012
    _lossi = []

    # Track update-to-data ratios per layer over time
    _linear_layers = [_l for _l in layers if hasattr(_l, "weight")]
    ud_ratios = {_i: [] for _i in range(len(_linear_layers))}

    _g = torch.Generator().manual_seed(42)

    for _i in range(_max_steps):
        # Mini-batch
        _ix = torch.randint(0, xs_train.shape[0], (_batch_size,), generator=_g)
        _xb, _yb = xs_train[_ix], ys_train[_ix]

        # Forward pass
        _emb = C[_xb]
        _x = _emb.view(-1, block_size * emb_dim)
        for _layer in layers:
            _x = _layer(_x)
        _loss = F.cross_entropy(_x, _yb)

        # Backward pass
        for _p in parameters:
            _p.grad = None
        _loss.backward()

        # Update — decay learning rate after 10k steps
        _lr_current = _lr if _i < 10000 else 0.01
        for _p in parameters:
            _p.data += -_lr_current * _p.grad

        _lossi.append(_loss.log10().item())

        # Log update-to-data ratio
        with torch.no_grad():
            for _j, _ll in enumerate(_linear_layers):
                _ud = (_lr_current * _ll.weight.grad).std() / _ll.weight.data.std()
                ud_ratios[_j].append(_ud.log10().item())

    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.plot(_lossi)
    _ax.set_xlabel("step")
    _ax.set_ylabel("log10(loss)")
    _ax.set_title("Training loss")

    mo.vstack([_fig, mo.md(f"**Final training loss: {_lossi[-1]:.4f}**")])
    return (ud_ratios,)


@app.cell
def _(C, F, block_size, emb_dim, layers, mo, xs_val, ys_val):
    # --- Evaluate on val set using running stats ---
    for _layer in layers:
        if hasattr(_layer, "training"):
            _layer.training = False

    _emb = C[xs_val]
    _x = _emb.view(-1, block_size * emb_dim)
    for _layer in layers:
        _x = _layer(_x)
    _val_loss = F.cross_entropy(_x, ys_val).item()

    # Set back to training mode
    for _layer in layers:
        if hasattr(_layer, "training"):
            _layer.training = True

    mo.md(f"**Validation loss: {_val_loss:.4f}**")
    return


@app.cell
def _(layers, mo, plt, torch, ud_ratios):
    # --- Activation & gradient diagnostics ---
    _tanh_layers = [_l for _l in layers if isinstance(_l, type(layers[2]))]

    # Plot activation distributions
    _fig1, _axes1 = plt.subplots(1, len(_tanh_layers), figsize=(4 * len(_tanh_layers), 4))
    for _i, _tl in enumerate(_tanh_layers):
        _ax = _axes1[_i] if len(_tanh_layers) > 1 else _axes1
        _h = _tl.out.detach()
        _ax.hist(_h.view(-1).numpy(), bins=50, color="steelblue", edgecolor="black")
        _sat_pct = (torch.abs(_h) >= 0.95).float().mean().item() * 100
        _ax.set_title(f"Layer {_i}: {_sat_pct:.1f}% saturated")
        _ax.set_xlabel("tanh output")

    # Plot gradient distributions
    _linear_layers = [_l for _l in layers if hasattr(_l, "weight")]
    _fig2, _axes2 = plt.subplots(1, len(_linear_layers), figsize=(4 * len(_linear_layers), 4))
    for _i, _ll in enumerate(_linear_layers):
        _ax = _axes2[_i] if len(_linear_layers) > 1 else _axes2
        _grad = _ll.weight.grad.detach()
        _ax.hist(_grad.view(-1).numpy(), bins=50, color="salmon", edgecolor="black")
        _ratio = _grad.std() / _ll.weight.data.std()
        _ax.set_title(f"Linear {_i}: grad/data = {_ratio:.4f}")
        _ax.set_xlabel("gradient value")

    # Plot update-to-data ratio over time
    _fig3, _ax3 = plt.subplots(figsize=(10, 4))
    for _i in ud_ratios:
        _ax3.plot(ud_ratios[_i], label=f"Linear {_i}")
    _ax3.set_xlabel("step")
    _ax3.set_ylabel("log10(update/data ratio)")
    _ax3.set_title("Update-to-data ratio over training")
    _ax3.axhline(y=-3, color="black", linestyle="--", label="~1e-3 target")
    _ax3.legend()

    _fig1.suptitle("Activation distributions per Tanh layer")
    _fig2.suptitle("Gradient distributions per Linear layer")
    _fig1.tight_layout()
    _fig2.tight_layout()

    mo.vstack([_fig1, _fig2, _fig3])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
