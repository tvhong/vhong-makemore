# Study Plan

## Lecture 1: Makemore — Character-level Language Model

**Video**: [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) (1h57m)
**Linear Issue**: [VY-391](https://linear.app/aisi/issue/VY-391)
**Due**: Mar 15, 2026
**Est. Hours**: 5h

### Key Concepts

- Bigram character-level language model
- PyTorch tensors and basic operations
- Language modeling fundamentals (predicting next character)
- Training on a dataset of names
- Counting-based approach vs neural network approach
- Negative log likelihood loss

### What We'll Build

A character-level language model that learns to generate name-like strings by training on a dataset of names. Two approaches:

1. **Bigram counting model** — count character pair frequencies, normalize to get probabilities
2. **Neural network model** — single-layer neural net that learns the same distribution

### Study Method: Watch-then-Implement Blocks

Interleave watching and coding. Watch a section, pause, then:

1. **Quiz** — answer 3-7 questions to check understanding before touching code
2. **Implement** — code it up yourself

This avoids passive watching while giving you enough context to code with productive struggle. The quiz step catches gaps in understanding early, before they become debugging headaches.

### Time Log

| Date | Hours | What |
|------|-------|------|
| Mar 13 | 2h | Block 1: watched bigram section + implemented counting model |
| Mar 23 | 1.75h | Block 2: quiz + implemented forward pass & training loop (steps 5–6) |
| Mar 24 | 2h | Bug exercises, Block 2 steps 7–8, planned Lecture 2 |

#### Block 1: Bigram Counting Model (~40 min watch, ~1.5h implement)

**Watch** the first ~40 minutes, covering the counting-based bigram approach. Then pause and implement:

1. [x] Download and explore the names dataset
2. [x] Build bigram frequency table from the dataset
3. [x] Implement sampling from the bigram model
4. [x] Compute loss (negative log likelihood) on the dataset

#### Block 2: Neural Network Model (~1h watch, ~1.5h implement)

**Watch** the next ~1 hour, covering the neural net reframing. Then pause and implement:

5. [x] Reframe as a neural network with one-hot encoding
6. [x] Train the neural net with gradient descent
7. [x] Compare neural net results to the counting approach
8. [x] Sample from the trained neural net

#### Block 3: Wrap-up (~15 min watch)

**Watch** the remaining wrap-up and review.

---

## Lecture 2: Makemore — MLP (Bengio et al. 2003)

**Video**: [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) (1h15m)
**Linear Issue**: [VY-392](https://linear.app/aisi/issue/VY-392)
**Due**: Mar 28, 2026
**Est. Hours**: 5h

### Key Concepts

- Character-level embeddings (lookup table)
- Multi-layer perceptron (MLP) architecture
- Context window: using more than one previous character
- Hidden layer with tanh activation
- Train/validation/test splits
- Hyperparameter tuning (embedding size, hidden layer size, learning rate)
- Overfitting and underfitting

### What We'll Build

An MLP that takes a fixed-size context window of previous characters (e.g., 3) and predicts the next character. This is a big step up from bigrams — the model can now use more context to make better predictions.

### Time Log

| Date | Hours | What |
|------|-------|------|
| Mar 24 | 1h | Blocks 1–2: quiz, implemented embedding + MLP training (steps 1–7) |
| Mar 25 | 1.5h | Block 3: train/val/test split, grid search over emb_dim and n_hidden (steps 8–9) |
| Mar 26 | 0.5h | Sampling + completed Lecture 2 (steps 10–11) |

#### Block 1: Embeddings & Dataset (~25 min watch, ~1h implement)

**Watch** the first ~25 minutes, covering the embedding lookup and dataset construction. Then pause and implement:

1. [x] Build the dataset with a context window (e.g., block_size=3)
2. [x] Implement the embedding lookup table (C matrix)
3. [x] Verify shapes: input indices → embedded vectors → concatenated context

#### Block 2: MLP Forward Pass & Training (~25 min watch, ~1.5h implement)

**Watch** the next ~25 minutes, covering the MLP architecture and training. Then pause and implement:

4. [x] Build the hidden layer (W1, b1) with tanh activation
5. [x] Build the output layer (W2, b2) and softmax
6. [x] Implement the training loop with cross-entropy loss
7. [x] Train and check that loss decreases

#### Block 3: Splitting, Tuning & Sampling (~25 min watch, ~1h implement)

**Watch** the remaining section on evaluation and tuning. Then pause and implement:

8. [x] Split data into train/val/test sets
9. [x] Experiment with hyperparameters (embedding size, hidden size, learning rate)
10. [x] Sample names from the trained MLP
11. [x] Compare MLP loss to bigram model loss

---

## Lecture 3: Makemore — Activations & Gradients

**Video**: [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) (1h55m)
**Linear Issue**: [VY-393](https://linear.app/aisi/issue/VY-393)
**Due**: Mar 29, 2026
**Est. Hours**: 6h

### Key Concepts

- Why initialization matters (too-confident initial predictions destroy learning)
- Kaiming initialization (scaling weights by fan-in)
- Activation statistics: monitoring distribution of hidden layer outputs
- Gradient flow: saturated tanh → dead gradients → slow learning
- Batch normalization: normalize pre-activations to be unit Gaussian
- BN during training vs inference (running mean/variance)
- Diagnostic tools: plotting activation/gradient distributions per layer

### What We'll Build

Starting from our lecture 2 MLP, we'll diagnose and fix training pathologies. By the end we'll have a deeper MLP with proper initialization, batch normalization, and tools to visualize what's happening inside the network during training.

### Time Log

| Date | Hours | What |
|------|-------|------|
| Mar 26 | 0.5h | Pre-lecture discussion: activation/gradient diagnostics, saturation, residual connections |
| Apr 4 | 1.5h | Block 1 review, Block 2 quiz + implemented BN with gamma/beta and running stats (steps 5–8) |
| Apr 5 | 2h | Block 3: quiz, refactored into module classes, built 3-layer MLP with training loop + diagnostics (steps 9–11) |

#### Block 1: Initialization & Activation Statistics (~40 min watch, ~1.5h implement)

**Watch** the first ~40 minutes, covering why the initial loss is too high and how initialization affects training. Then pause and implement:

1. [x] Diagnose the initial loss problem — compute what the loss *should* be at initialization (uniform predictions → -log(1/27)) and check how far off we are
2. [x] Fix output layer initialization so initial loss is close to -log(1/27)
3. [x] Fix hidden layer initialization using Kaiming init (scale W1 by `(fan_in)**-0.5`)
4. [x] Add instrumentation: plot activation distributions (histograms of tanh outputs per layer)

#### Block 2: Batch Normalization (~40 min watch, ~1.5h implement)

**Watch** the next ~40 minutes, covering batch normalization mechanics. Then pause and implement:

5. [x] Implement batch normalization layer: normalize pre-activations to zero mean, unit variance
6. [x] Add learnable scale (gamma) and shift (beta) parameters to BN
7. [x] Track running mean and running variance during training (for inference)
8. [x] Switch to running stats at eval time and verify val loss is consistent

#### Block 3: Gradient Flow & Deeper Networks (~35 min watch, ~1.5h implement)

**Watch** the remaining section on gradient diagnostics and going deeper. Then pause and implement:

9. [x] Add instrumentation: plot gradient distributions and gradient-to-data ratios per layer
10. [x] Diagnose tanh saturation — identify layers where activations are stuck at ±1
11. [x] Build a deeper MLP (e.g., 3+ hidden layers) with BN and proper init
12. ~~Train the deep model and compare loss to the single-hidden-layer MLP from lecture 2~~ (skipped)

---

## Lecture 4: Makemore — Becoming a Backprop Ninja

**Video**: [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) (1h56m)
**Due**: Apr 5, 2026
**Est. Hours**: 6h

### Key Concepts

- Manual backpropagation through tensor operations (not scalar like micrograd)
- Deriving gradients for cross-entropy loss, matrix multiply, tanh, batch norm
- Understanding shapes: each gradient has the same shape as its corresponding tensor
- Bessel's correction in batch norm (dividing by N-1 vs N)
- Fusing operations: collapsing many atomic gradients into a single efficient formula
- Building intuition for how gradients flow through a compute graph

### What We'll Build

Starting from the forward pass of our 2-layer MLP with batch norm, we'll manually implement the backward pass for every operation — replacing `loss.backward()` with our own gradient computations. We verify correctness by comparing against PyTorch's autograd at each step.

### Format Note

This lecture is exercise-centric. Karpathy recommends working through the exercises yourself and using the video when stuck, rather than watching passively. Our approach: attempt each exercise first, then check the video if stuck.

### Time Log

| Date | Hours | What |
|------|-------|------|
| Apr 5 | 3h10m | Exercise 1: completed all manual backprop gradients (cross-entropy → logits → linear layer 2 → tanh → batchnorm affine → bnraw → bnvar → bndiff → linear layer 1 → embedding) |

#### Block 1: Setup & Cross-Entropy Backward (~30 min watch, ~1.5h implement)

**Watch** the intro and starter code setup. Then work through the exercises:

1. [x] Set up the forward pass with atomic operations (matching Karpathy's starter code)
2. [x] Backprop through cross-entropy loss: gradient of logits from softmax + negative log likelihood
3. [x] Verify gradients match `loss.backward()` using `torch.allclose`

#### Block 2: Batch Norm Backward (~40 min watch, ~2h implement)

**Watch** the batch norm section and Bessel's correction digression. Then implement:

4. [x] Backprop through the second linear layer (W2, b2)
5. [x] Backprop through tanh activation
6. [x] Backprop through batch normalization (the hardest part — mean, variance, normalization, gamma, beta)
7. [x] Understand Bessel's correction: why `torch.var` uses N-1 by default and how it affects the gradient

#### Block 3: First Layer & Fused Operations (~30 min watch, ~1h implement)

**Watch** the remaining section on completing the backward pass and fusing operations. Then implement:

8. [x] Backprop through the first linear layer (W1, b1) and embedding lookup
9. [ ] Fuse the cross-entropy backward pass into a single efficient formula
10. [ ] Fuse the batch norm backward pass into a single formula
11. [ ] Run a full training loop using only manual gradients and verify it trains correctly
