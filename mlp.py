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
import random

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

xs_train, ys_train = build_dataset(words[:n1])
xs_val, ys_val = build_dataset(words[n1:n2])
xs_test, ys_test = build_dataset(words[n2:])

print(f"Train: {xs_train.shape[0]} examples")
print(f"Val:   {xs_val.shape[0]} examples")
print(f"Test:  {xs_test.shape[0]} examples")


# --- Training and evaluation ---


def train_and_evaluate(emb_dim=2, n_hidden=100, lr=0.1, epochs=100):
    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((27, emb_dim), generator=g, requires_grad=True)
    W1 = torch.randn((block_size * emb_dim, n_hidden), generator=g, requires_grad=True)
    b1 = torch.randn(n_hidden, generator=g, requires_grad=True)
    W2 = (torch.randn((n_hidden, 27), generator=g) * 0.01).requires_grad_(True)
    b2 = torch.zeros(27, requires_grad=True)

    parameters = [C, W1, b1, W2, b2]
    n_params = sum(p.numel() for p in parameters)

    train_losses = []
    for epoch in range(epochs):
        # Forward pass
        emb = C[xs_train]
        emb_cat = emb.view(-1, block_size * emb_dim)
        h = torch.tanh(emb_cat @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, ys_train)

        # Backward pass
        loss.backward()

        # Update weights
        for p in parameters:
            p.data -= lr * p.grad

        # Zero gradients
        for p in parameters:
            p.grad = None

        train_losses.append(loss.item())

    # Evaluate on validation set
    emb = C[xs_val]
    emb_cat = emb.view(-1, block_size * emb_dim)
    h = torch.tanh(emb_cat @ W1 + b1)
    logits = h @ W2 + b2
    val_loss = F.cross_entropy(logits, ys_val)

    return train_losses, val_loss.item(), n_params, parameters


# --- Train best config ---

train_losses, val_loss, n_params, params = train_and_evaluate(emb_dim=10, n_hidden=200)
C, W1, b1, W2, b2 = params
print(f"Train loss: {train_losses[-1]:.4f} | Val loss: {val_loss:.4f}")

# --- Step 10: Sample names from the trained MLP ---

g = torch.Generator().manual_seed(2147483647)
for _ in range(10):
    context = [0] * block_size
    name = []
    while True:
        emb = C[torch.tensor(context)]
        emb_cat = emb.view(1, -1)
        h = torch.tanh(emb_cat @ W1 + b1)
        logits = h @ W2 + b2
        probs = torch.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        if ix == 0:
            break
        name.append(itos[ix])
        context = context[1:] + [ix]
    print("".join(name))
