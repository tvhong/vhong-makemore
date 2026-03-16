# Study Plan

## Lecture 2: Makemore — Character-level Language Model

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

#### Block 1: Bigram Counting Model (~40 min watch, ~1.5h implement)

**Watch** the first ~40 minutes, covering the counting-based bigram approach. Then pause and implement:

1. [x] Download and explore the names dataset
2. [x] Build bigram frequency table from the dataset
3. [x] Implement sampling from the bigram model
4. [x] Compute loss (negative log likelihood) on the dataset

#### Block 2: Neural Network Model (~1h watch, ~1.5h implement)

**Watch** the next ~1 hour, covering the neural net reframing. Then pause and implement:

5. [ ] Reframe as a neural network with one-hot encoding
6. [ ] Train the neural net with gradient descent
7. [ ] Compare neural net results to the counting approach
8. [ ] Sample from the trained neural net

#### Block 3: Wrap-up (~15 min watch)

**Watch** the remaining wrap-up and review.
