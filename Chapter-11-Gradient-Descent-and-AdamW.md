# Chapter 11 — Training III: Gradient Descent, AdamW and the Training Loop

## Introduction

Chapter 10 showed how backpropagation produces a gradient for every parameter, without changing any of them. This chapter covers what happens next: how those gradients become actual updates (gradient descent, and the more capable AdamW), how training data is fed to the model in mini-batches and epochs, and finally the complete training loop that ties all three training chapters together.

---

## Part 1 — Gradient descent: actually updating the weights

People often blur three different things together, so let's separate them:

```text
Loss              →  "How wrong am I?"
Gradient          →  "Which direction should each weight move?"
Gradient descent  →  "How far should I actually move each weight?"
```

An analogy: a GPS computes the route, and *you* decide how far to drive. The gradient is the GPS. Gradient descent is the "drive 100 meters" part. Notably, **gradient descent doesn't compute gradients; it only uses them.**

### What a gradient really means

Back to our example: weight `w = 2`, gradient `−12`. That does **not** mean "change the weight by 12". A gradient is a **slope**, not an instruction for distance. Picture standing on a hillside. The gradient tells you *"this slope tilts downward to the right"*; it doesn't tell you to walk eight kilometers.

### The update rule

How far to move is controlled by a small number called the **learning rate**, `η`:

```text
New weight = Old weight − η × Gradient
```

With `η = 0.1`, `w = 2` and gradient `−12`:

```text
New weight = 2 − 0.1 × (−12) = 2 + 1.2 = 3.2
```

The gradient was negative, so subtracting it **increased** the weight, which is exactly what we wanted. If the gradient had been positive (say `+8`), the weight would have *decreased*: `2 − 0.1 × 8 = 1.2`. That's all the minus sign means: **always walk downhill.** Many people memorize the formula without ever understanding it; it's just "old weight, plus a tiny correction toward lower loss".

### Choosing the learning rate

The learning rate is a **hyperparameter**: a number the architect chooses before training, like the number of heads. It is not learned. And it matters a lot:

```text
Too large (η = 10):   2 − 10 × (−12) = 122     → we jumped clear across the valley; training becomes unstable
Too small (η = 10⁻⁹): 2 → 2.000000012          → almost no movement; training would take years
```

Same direction, different distance: like driving 1 meter versus 100 kilometers.

### Watching a weight walk downhill

Let's actually run a few steps of our tiny network (target `8`, input `3`, `η = 0.1`). Each step, we recompute the prediction, the loss, the gradient, and then update:

```text
Step   Weight   Prediction   Loss      Gradient   New weight
 1     2.000      6.000     4.0000     −12.00      3.200
 2     3.200      9.600     2.5600      +9.60      2.240
 3     2.240      6.720     1.6384      −7.68      3.008
 4     3.008      9.024     1.0486      +6.14      2.394
```

The loss falls every single step (`4 → 2.56 → 1.64 → 1.05`), which is the proof that the whole mechanism works. Notice something else, though: the weight *bounces* back and forth around its ideal value (`8 ÷ 3 ≈ 2.667`), overshooting a little each time. That's a sign `η = 0.1` is a bit large for this problem. With a gentler `η = 0.05`, the same weight would approach smoothly: `2 → 2.6 → 2.66 → ...`. This is the learning-rate tradeoff in action.

### The complete loop so far

```text
Input sentence → Forward pass → Prediction → Cross-entropy loss
      → Backpropagation → a gradient for EVERY parameter
      → Gradient descent → a tiny update to EVERY parameter → repeat
```

Notice that gradient descent never sees sentences, embeddings, attention, or the Transformer. All it ever sees is *a current weight* and *a current gradient*. For a model with 10 billion parameters, its entire job is one line:

```python
for every parameter:
    parameter = parameter - learning_rate * gradient
```

No intelligence, no optimization tricks, just one line of mathematics.

---

## Part 2 — Why modern models use AdamW instead

Plain gradient descent is the right idea, but it has real weaknesses, and every large LLM today (GPT, Llama, Gemma, and others) uses a smarter version called **AdamW**. Here is why.

### Three problems with plain gradient descent

Imagine hiking down a real mountain, not a smooth bowl.

1. **Zig-zagging.** In a narrow valley, the gradient points left, then right, then left. You waste steps bouncing across the valley instead of walking down it.
2. **Flat regions.** The slope is almost zero, so the gradient is almost zero, so steps are almost zero, and training nearly stalls.
3. **Very steep regions.** The gradient is huge, so the step is huge, and you leap over the minimum.

It's like driving at exactly 20 km/h whether you're on a highway, in traffic, or entering a parking lot: the same speed everywhere, which is rarely ideal. Researchers wanted something that remembers what's been happening, not just today's slope.

### Idea 1: momentum (remember the direction)

Picture pushing a heavy shopping cart. The first push barely moves it. The second moves it a bit more easily. By the third, it rolls smoothly. **Momentum remembers previous pushes.** If the last gradients were `−2, −3, −2.5`, they all point roughly the same way, so momentum says *"I trust this direction"* and moves faster. If they were `−2, +2, −2, +2`, the direction keeps flipping, so momentum says *"don't overreact"* and smooths the update.

Plain gradient descent has no memory: every step starts fresh, like a driver who forgets which way they were heading a second ago. Adam keeps a running **weighted average** of recent gradients, weighting recent ones more:

```text
m = β₁ · (previous m)  +  (1 − β₁) · (today's gradient)
```

With `β₁ = 0.9`, previous `m = 10` and today's gradient `4`:

```text
m = 0.9 × 10 + 0.1 × 4 = 9.4
```

Today's gradient didn't replace history, it only nudged it. `β₁` simply controls how long the memory is: `β₁ = 0` means no memory (plain gradient descent), and `β₁ = 0.99` means history dominates. The typical value is `0.9`.

### Idea 2: adapt the step size to each parameter

Some weights always receive huge gradients (say around `100`), others always tiny ones (around `0.00001`). Giving both the same step size makes no sense: the first needs tiny steps, the second needs much bigger ones. So Adam keeps a second memory, of how **large** each parameter's gradients usually are, by averaging the *squared* gradient:

```text
v = β₂ · (previous v)  +  (1 − β₂) · (today's gradient)²
```

Why square it? If gradients alternate `+5, −5, +5, −5`, a direct average is `0`, wrongly suggesting nothing is happening. Squaring makes both `25`, so `v` measures the **size** of the gradients regardless of direction.

### The Adam update

Putting both memories together:

```text
New weight = Old weight − η × m / (√v + ε)
```

- `m` supplies the preferred direction (momentum).
- `√v` normalizes the step: if a parameter's gradients are usually huge, `√v` is large, so its step shrinks; if they're usually tiny, `√v` is small, so its step grows relatively larger.
- `ε` is a tiny number like `10⁻⁸`, there purely so we never divide by zero.

(Real implementations also apply a small "bias correction" to `m` and `v` during the first few steps, because both start at zero. It doesn't change the idea.)

A precision worth making: it's tempting to say "Adam gives every parameter its own learning rate". More exactly, the learning rate `η` is still a single global value chosen by the architect. What Adam changes is the **effective step size** per parameter, by dividing by `√v`:

```text
Global learning rate → adaptive scaling (different for each parameter) → final step for that parameter
```

### AdamW: adding weight decay

Researchers found that weights tended to grow larger than necessary, which often hurts how well the model generalizes to new text. So they added **weight decay**: on every step, in addition to following the gradient, the optimizer also pulls each weight very slightly toward zero, like a mild spring attached to zero. It's not strong enough to stop learning, but it discourages weights from growing without good reason. **Adam + weight decay = AdamW.**

```text
Gradient descent:  "Where is downhill? Walk."
Momentum:          "Where has downhill been lately? Smooth my direction."
Variance:          "How rough is the terrain? Adjust my step size."
Weight decay:      "Don't wander too far unless there's a good reason."
```

### Training costs far more memory than running a model

Here's a consequence people often miss. Adam stores two extra numbers for every parameter. So during training, each parameter needs its **weight**, its **gradient**, its **momentum `m`**, and its **variance `v`**: roughly four values per parameter. During inference, you only need the weights. This is one of the main reasons training a large LLM takes dramatically more GPU memory than merely running one.

---

## Part 3 — Mini-batches, epochs, and when to stop

Suppose the training set holds 1 trillion tokens. We can't load all of it, run a forward pass, then a backward pass, then update. No computer can hold that. So how do we actually feed data to the model?

### Three ways to use the data

**Update after every example (stochastic gradient descent, SGD).** Fast to react, but noisy. One sentence says "increase these weights", the next ("Dogs are wonderful") says "increase *different* weights", and the one after that, about quantum physics, pulls somewhere else entirely. The model keeps changing direction.

**Update after the entire dataset.** Average the gradient over all 1,000 examples, then make one update. Very stable, but very slow, and for a real dataset, impossible.

**Mini-batches: the compromise.** Take a modest batch (say 100 examples), average their gradients, update, then take the next 100:

```text
Examples 1–100   →  forward → average loss → backward → update
Examples 101–200 →  forward → average loss → backward → update
...
```

Think of learning to cook. Changing the recipe after every single meal is unstable. Cooking 10,000 meals before changing it once is far too slow. Cooking 50 meals, looking at the average feedback, and *then* adjusting is what chefs do. That's mini-batching. **Averaging the gradients** means the optimizer follows the consensus of a batch, not the noise of a single sentence.

Batch size is another hyperparameter the architect chooses.

### Epochs

If 1,000 examples are split into batches of 100, that's 10 batches per pass. After all ten, we've completed one **epoch**:

> **One epoch = one complete pass through the entire training dataset.**

Training usually runs several epochs, each starting from the weights the previous one left behind. Before each epoch the data is **shuffled**. Otherwise the model could accidentally pick up the order (cats, then dogs, then birds), and shuffling improves how well it generalizes.

### When does training stop?

Three common answers:

1. **A fixed number of epochs.** Simple, though not always optimal.
2. **Early stopping on validation loss.** Researchers hold back a **validation set** that the model never trains on. After each epoch they compare training loss with validation loss. If validation loss stops improving, the model has begun memorizing rather than learning, so training stops.
3. **A fixed number of tokens.** Modern LLMs usually don't think in epochs at all. Their datasets are so large that a full pass may never even be completed, so the budget is stated as "train on 10 trillion tokens."

(Real training also usually *changes* the learning rate over time, commonly starting small, rising, and then decaying, rather than holding it fixed. That detail is an engineering refinement on top of everything above.)

---

## The complete picture

```text
Initialize all weights randomly
        │
        ▼
Shuffle the dataset, take a mini-batch
        │
        ▼
Forward pass → prediction (probabilities)
        │
        ▼
Cross-entropy loss ("how surprised was the model?")
        │
        ▼
Backpropagation → one gradient for every parameter
        │
        ▼
AdamW → a tiny, adaptive update to every parameter
        │
        ▼
Next mini-batch ... repeat for billions of steps
        │
        ▼
Trained model  →  inference (predict the next token)
```

And this connects back to something from earlier chapters. In Chapter 4 we described the numbers inside `W_Q` being nudged `0.83 → 0.84 → 0.85`. Now you know *how* each nudge is computed: the loss gives a gradient for that exact entry (Chapter 10), and the optimizer turns that gradient into a small step (Parts 1 and 2 of this chapter). Nothing mystical is left, only linear algebra, probability, calculus, optimization, and a very large amount of compute.

One more point of scope: what we described here is training a model **from scratch on raw text** to predict the next word, which produces a general-purpose "base" model. Teaching a model to follow instructions, or adapting it to a specialty, builds on this same machinery, and we cover it in Chapter 13.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "A gradient of `−12` means change the weight by 12." | A gradient is a *slope*: it gives direction and sensitivity, not a distance. The learning rate decides how far to move. |
| "Gradient descent calculates the gradients." | It only *uses* them. Backpropagation computes gradients; gradient descent (or AdamW) turns them into updates. |
| "Adam gives every parameter its own learning rate." | The learning rate is one global value. Adam adapts each parameter's *effective step size* by dividing by `√v`. |
| "Training and running a model need the same memory." | Training also stores gradients plus Adam's `m` and `v`, roughly four values per parameter instead of one. |
| "An epoch is how modern LLMs measure training." | Large LLMs usually budget in *tokens*; their datasets are so large that they may never complete many full epochs. |

---

## Quick reference

```text
Gradient descent: w ← w − η·gradient         (η = learning rate, chosen by the architect)
AdamW:           adds momentum (m), per-parameter step scaling (√v), and weight decay
Mini-batch:      average gradients over a small batch, then update
Epoch:           one full pass over the dataset (modern LLMs budget in tokens)

Forward pass = prediction.  Loss = feedback.  Backward pass = blame.  Optimizer = correction.
```

---

## What's next

We now understand, end to end, how a GPT-style model is built and how its random numbers become useful. Training produces a fixed set of weights, and a model is only useful once you *use* it. Chapter 12 turns to **inference**: how a trained GPT writes text one token at a time, why the KV cache exists, and how sampling picks the actual word. After that, Chapter 13 shows how a base model is turned into an assistant (fine-tuning), and Chapter 14 how its weights are compressed (quantization).
