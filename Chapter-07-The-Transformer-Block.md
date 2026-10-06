# Chapter 7 — The Transformer Block: Residual Connections and LayerNorm

## Introduction

We now have both halves of a Transformer block fully explained: Chapters 4–5 covered Attention in depth — how a token **gathers** relevant information from every other token — and Chapter 6 covered the Feed-Forward Network, or FFN — how a token then *processes* that gathered information on its own. This chapter is pure assembly: **how do Attention and the FFN actually get wired together into one repeatable unit** — the Transformer Block — and what stops that unit from quietly breaking as we stack dozens of them on top of each other?

Two ideas answer that: **residual connections** and **Layer Normalization**. Neither adds any new "intelligence" to the model — both exist purely to make a very deep stack of blocks actually trainable at all. That might sound like a minor plumbing detail. It isn't — without them, deep Transformers simply don't train.

---

## Part 1 — A question worth asking: what happens to the original input?

Suppose we've built everything through the FFN:

```text
Input Token → Attention → W_O → FFN → Output Token
```

Take our running example. Suppose the token `"it"` arrives as the vector `[4, 9]`. After the attention and FFN computations have both transformed it, it comes out as, say, `[12, 8]`.

Stop and ask: **where did `[4, 9]` go?** It's gone. Completely overwritten.

Is that actually a problem? Consider an analogy: imagine an AI editor rewrites your paragraph —

```text
Original:  "The animal didn't cross the road because it was tired."
Rewritten: "The creature remained stationary due to exhaustion."
```

Maybe the rewrite is better. Maybe the AI misunderstood something subtle and the rewrite is actually worse. Either way, once the original is gone, there's no way back, and no way to blend the good parts of both. Wouldn't it be safer to keep a copy of the original around, and only *add* improvements to it, rather than blindly replacing it?

---

## Part 2 — Residual connections: don't replace, add

That question is exactly why **residual connections** exist. Instead of this:

```text
Input → Transformer computation → Output
```

we do this:

```text
        Input
          │
          ├──────────────────────┐
          ▼                      │
   Transformer computation       │
          │                      │
          ▼                      │
   Add the input back  ◄─────────┘
          │
          ▼
    Final Output
```

The original input takes a "shortcut" around the computation, and gets **added back** at the end. Concretely:

```text
Input:          [4, 9]
After FFN:       [12, 8]
Residual output: [4,9] + [12,8] = [16, 17]
```

That's the entire mechanism. In formula form:

```text
Output = x + F(x)
```

where `x` is the original input, and `F(x)` is whatever the block just computed (attention, or the FFN). No new matrices, no new learned parameters — residual connections are pure addition. **Nothing is learned here at all.**

### Why is this such a good idea?

Think about training a new employee. Every week they learn something new. Would you rather they *forget everything and relearn from scratch each week*, or *keep everything they already know and add the new lesson on top*? Obviously the second. That's precisely what a residual connection does for a token's representation:

```text
Before attention:   "I am the word 'it.'"
After attention:    "I know I probably refer to the animal."
After the FFN:       "I also understand the causal relationship."

Residual connections say: don't throw away "I am the word 'it,'"
just keep adding new understanding on top of it.
```

Or think of sculpting a statue: you don't start over from a fresh block of stone every day. Day 1, you shape the stone. Day 2, you take *yesterday's statue* and make small improvements. Day 3, the same. The Transformer doesn't reinvent each token's representation from scratch at every block — it **iteratively refines** it, one small improvement layered on top of the last.

```text
Without residuals:  Old representation → REPLACED
With residuals:      Old representation + improvement → Better representation
```

### Why researchers actually needed this

Before residual connections were invented, people tried building very deep networks — dozens or even a hundred layers stacked on top of each other — and training them went badly. Information from early layers gradually got diluted or lost entirely as it passed through so many transformations, and the gradients used to train the network (Chapter 10) got weaker and weaker the further back they had to travel — a well-known problem called the **vanishing gradient**. Residual connections create a direct, unobstructed path for both information and gradients to flow through, which makes training dramatically more stable, even with very deep stacks of layers. This single idea is a large part of what made truly deep neural networks — including modern LLMs with dozens of Transformer blocks — trainable at all.

### There are two residual connections per block, not one

Inside a real Transformer block, this shortcut-and-add pattern happens **twice** — once around attention, once around the FFN:

```text
Input
   │
   ├─────────────────────┐
   ▼                      │
Attention → W_O           │
   │                      │
   ▼                      │
Add original input  ◄─────┘
   │
   ├─────────────────────┐
   ▼                      │
FFN                       │
   │                      │
   ▼                      │
Add previous result  ◄────┘
```

### Why not just keep the original forever, and never let it change?

Because each block is supposed to contribute *something*. After Block 1 adds its improvement, the *result* — not the original raw embedding — becomes the input to Block 2, which gathers more context and adds its own refinement on top of that. The representation keeps evolving, block after block:

```text
Block 0 input:        [4, 9]
After Block 1:         [16, 17]
After Block 2:         [21, 24]
After Block 3:         ...
```

Every block contributes another layer of refinement on top of everything that came before — that's the whole appeal of stacking many blocks (Chapter 8 covers exactly what deeper stacks buy you).

---

## Part 3 — A new problem appears: the numbers keep growing

Residual addition solves the "don't throw away information" problem, but it quietly introduces a new one. Since every block **adds** something to what it received, and real models stack dozens of blocks, the numbers involved tend to keep getting larger as they pass through the network:

```text
Token starts as:        [2, 5]
After Block 1:            [8, 11]
After Block 2:            [15, 20]
After Block 3:            [31, 42]
...
After Block 20:           [840, 1021]
```

Why is that a problem? Imagine grading two students, where one answers on a scale of 0–10 and the other on a scale of 0–10,000 — comparing them directly is awkward, and any downstream calculation that expects "normal-sized" numbers starts behaving unpredictably. Or imagine three different shops weighing fruit — one in grams, one in kilograms, one in tons — you'd want to convert everything to a common scale before comparing. Left unchecked, ever-growing numbers make training increasingly unstable the deeper the network gets.

### Layer Normalization: put every vector back on a common scale

**Layer Normalization (LayerNorm)** fixes this by rescaling each token's vector, individually, back to a well-behaved range — every single time it's applied. Suppose after a residual step we have:

```text
[100, 110, 90, 95]
```

LayerNorm asks two questions about this vector:

**1. What's the average (mean)?**

```text
Mean = (100 + 110 + 90 + 95) ÷ 4 = 98.75
```

**2. How spread out are the values (standard deviation)?** You don't need the exact formula to follow along — just think of it as "how much variation exists." Suppose it comes out to `7.4`.

Then every value gets rescaled using the same rule: **subtract the mean, then divide by the standard deviation.**

```text
100 → (100 − 98.75) ÷ 7.4 ≈  0.17
110 → (110 − 98.75) ÷ 7.4 ≈  1.52
 90 → ( 90 − 98.75) ÷ 7.4 ≈ −1.18
 95 → ( 95 − 98.75) ÷ 7.4 ≈ −0.51
```

```text
Result: [0.17, 1.52, -1.18, -0.51]
```

### Did we just lose information?

This is the question almost everyone asks the first time they see this. Look closely at the ordering:

```text
Before:  110 > 100 > 95 > 90
After:   1.52 > 0.17 > -0.51 > -1.18
```

**The ordering, and the relative relationships between the numbers, stayed exactly the same.** Only the *scale* changed. It's like printing a map at A4 size and then enlarging it to A1 — the cities don't move relative to each other, the roads don't change, only the scale of the paper does. LayerNorm does the same thing to a vector: it keeps the shape of the information intact, and only fixes the scale.

### LayerNorm's two learned adjustments: γ and β

The normalization itself (subtract the mean, divide by the standard deviation) is completely fixed math, with nothing to learn. But real implementations then apply two small **learned** adjustments right after it, one value of each per embedding dimension:

```text
output  =  γ (gamma, a learned "scale")  ×  normalized value  +  β (beta, a learned "shift")
```

One naming note: this "scale" has nothing to do with the *learning rate* from training (Chapter 11). `γ` is simply a number that multiplies each dimension.

Are `γ` and `β` matrices, like `W_Q` or `W_1`? No, they're much smaller. If the embedding dimension is 4096, then `γ` is just 4096 individual numbers, and so is `β`: one scale value and one shift value per dimension, not a matrix mixing dimensions together. LayerNorm never learns *relationships between* dimensions. It only learns how much to rescale and shift each dimension, independently.

### A numerical example: what γ and β actually do

Take the vector from above. After the fixed normalization step, `[100, 110, 90, 95]` became:

```text
x̂ = [0.17, 1.52, -1.18, -0.51]      mean = 0, spread (standard deviation) = 1
```

This part is fixed math, and it's recomputed from each token's own numbers, so every token gets its own mean and spread. Now apply `γ` and `β`. Suppose training has settled on these values:

```text
γ = [1.0,  2.0,  0.5,  1.0]
β = [0.0,  1.0, -1.0,  0.5]

y₀ = 1.0 × ( 0.17) + 0.0 =  0.17
y₁ = 2.0 × ( 1.52) + 1.0 =  4.04
y₂ = 0.5 × (-1.18) − 1.0 = -1.59
y₃ = 1.0 × (-0.51) + 0.5 = -0.01

y = [0.17, 4.04, -1.59, -0.01]
```

The output is no longer mean 0 and spread 1: its mean is about 0.65, and its spread is larger. And each dimension was treated differently:

- **Dimension 1** was amplified (`γ = 2`) and lifted (`β = 1`).
- **Dimension 2** was shrunk (`γ = 0.5`) and pushed down (`β = −1`).
- **Dimension 0** was left alone.

### How is this different from the zero-mean step?

The two steps do different jobs at different levels:

| | Normalization step | `γ` and `β` |
|---|---|---|
| Learned? | No, fixed math | Yes, trained |
| Computed from | Each token's own numbers | Nothing. They're stored constants |
| Same for every token? | No, each token gets its own mean and spread | Yes, the same values for every token |
| Applies to | The vector as a whole | Each dimension separately |

Normalization removes a token's *own* overall offset and scale. `γ` and `β` then apply a fixed, learned "house style" to each dimension.

### Why is this an advantage?

**1. Normalizing doesn't force you to lose information.** Plain normalization throws away the original overall scale and mean. But the model can undo it exactly if that turns out to be useful. Set `γ` to the original spread (7.4) and `β` to the original mean (98.75), and you get the original vector back (up to rounding):

```text
0.17 × 7.4 + 98.75 ≈ 100
1.52 × 7.4 + 98.75 ≈ 110
-1.18 × 7.4 + 98.75 ≈ 90
-0.51 × 7.4 + 98.75 ≈ 95
```

So the model gets the stability of normalizing, and it can recover as much of the original scale as it actually needs. In practice `γ` starts at 1 and `β` at 0, which is pure normalization, and training moves them only as far as it helps.

**2. Different dimensions can matter by different amounts.** After normalization every dimension sits on equal footing. But the next matrix multiplication treats a larger number as a stronger signal. `γ` lets the model say "dimension 1 should carry more weight from here on, and dimension 2 less." Without it, all dimensions would be forced to the same strength.

**3. `β` sets where each dimension sits relative to zero.** That position can matter for what comes next. For example, whether a value is above or below zero affects how a function like GELU (Chapter 6) treats it. A learned shift lets the model place values where the steps after it work well.

**4. It's cheap.** For an embedding size of 4096, that's only 4096 + 4096 extra numbers per LayerNorm, compared with millions in a weight matrix.

One more point: many recent models, such as Llama, use a simplified variant called **RMSNorm**. It drops the mean subtraction and `β` and keeps only a learned scale. So `γ` is the more essential of the two.

---

## Part 4 — Assembling the complete Transformer Block

We now have every ingredient. Here is the full, original Transformer block, exactly as described in the 2017 paper that introduced this architecture:

```text
Input
   │
   ▼
Multi-Head Attention
   │
   ▼
W_O
   │
   ▼
Residual Add
   │
   ▼
LayerNorm
   │
   ▼
Feed Forward Network
   │
   ▼
Residual Add
   │
   ▼
LayerNorm
   │
   ▼
Output
```

This single block — attention to gather information, a residual connection to preserve what came before, LayerNorm to keep the numbers well-behaved, then the FFN to process the gathered information, another residual connection, another LayerNorm — is the fundamental repeating unit of a Transformer. Chapter 8 covers what happens when you stack many of these on top of each other, which is essentially what "GPT" actually is.

### An important correction for modern LLMs

The diagram above — LayerNorm placed *after* the residual addition — is called **Post-LayerNorm**, and it's how the original 2017 Transformer paper did it. Most modern decoder-only LLMs (GPT-2, GPT-3, Llama, Gemma, Qwen, and most others you'll encounter) actually use a rearranged version called **Pre-LayerNorm** instead:

```text
Post-LayerNorm (original):      Pre-LayerNorm (most modern LLMs):

Attention                        LayerNorm
   ↓                                ↓
Residual                        Attention
                                     ↓
                                 Residual
```

...and the same swap happens around the FFN. Researchers found that normalizing *before* each computation, rather than after, makes training dramatically more stable for very deep, modern-scale models. The conceptual role of every component — what attention does, what the FFN does, why residuals exist, why LayerNorm exists — is identical either way; only the *order* changes. We'll point out this distinction again when we look at real GPT and Llama architectures directly.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "Residual connections are a special learned mechanism." | They introduce zero new parameters — it's literally just `Output = x + F(x)`, plain addition. |
| "Adding the original input back means nothing really changes." | The *addition* is fixed, but `F(x)` — attention or the FFN — still does real, learned computation. Residuals add refinements on top of the original; they don't cancel the refinement out. |
| "LayerNorm learns how the dimensions of a vector relate to each other." | It doesn't — the normalization step is fixed math, and its only learned parameters (`γ`, `β`) adjust each dimension *independently*, not relationships between dimensions. |
| "If `γ` and `β` can undo the normalization, the normalization is pointless." | The normalization step fixes the numbers' scale at the start, so training is stable. `γ` and `β` then start at 1 and 0 (no change) and only move as far as training finds useful, so the model gets stability *and* flexibility, rather than being locked into a fixed zero-mean, unit-spread shape. |
| "Normalizing a vector changes what it represents." | It changes the *scale* only. The relative ordering and relationships between the numbers are preserved — nothing about the underlying information is discarded. |
| "There's one 'correct' place to put LayerNorm in a block." | Both Post-LN (original Transformer) and Pre-LN (most modern LLMs) are used in real, successful models — the placement is a stability-driven design choice, not a fixed law. |

---

## Quick reference

```text
Residual connection:   Output = x + F(x)
   - No learned parameters.
   - Preserves information and gradient flow through deep stacks of blocks.
   - Appears TWICE per block: around attention, and around the FFN.

LayerNorm:              Output = (x − mean) / std_dev  × γ + β
   - Normalization itself is fixed math (no parameters).
   - γ (scale) and β (shift) ARE learned — one value per embedding
     dimension, not a full matrix.
   - Keeps vector magnitudes stable across many stacked blocks.
   - Preserves ordering/relationships; only rescales.

One Transformer Block =
  Attention → Residual → LayerNorm → FFN → Residual → LayerNorm

Post-LN (original paper) vs Pre-LN (most modern LLMs):
  same components, different order, chosen for training stability.
```

---

## The architecture so far

The diagram so far. This chapter adds the double-lined boxes, and the skip lines on the right show the residual connections. With them, the Transformer block is complete.

```text
             Raw text: "The cat slept"
                          │
                          ▼
┌──────────────────────────────────────────────────┐
│ TOKENIZER                                        │
│  text → token IDs                                │
└──────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────┐
│ EMBEDDINGS                                       │
│  ID → vector, plus position vector (learned)     │
└──────────────────────────────────────────────────┘
                          │
                          ├─────────────────────────────┐
                          ▼                             │
┌──────────────────────────────────────────────────┐    │
│ MULTI-HEAD ATTENTION                             │    │
│  each head: softmax(Q·Kᵀ ÷ √dₖ + mask) · V       │    │
│  concatenate heads, then × W_O                   │    │
└──────────────────────────────────────────────────┘    │
                          │                             │
                          ▼                             │
╔══════════════════════════════════════════════════╗
║ ADD (residual)                             ◄ NEW ║◄───┘
║  output = input + attention output               ║
╚══════════════════════════════════════════════════╝
                          │
                          ▼
╔══════════════════════════════════════════════════╗
║ LAYERNORM                                  ◄ NEW ║
║  rescale each vector: (x − mean) ÷ std, × γ + β  ║
╚══════════════════════════════════════════════════╝
                          │
                          ├─────────────────────────────┐
                          ▼                             │
┌──────────────────────────────────────────────────┐    │
│ FFN                                              │    │
│  x · W1  →  GELU  →  · W2                        │    │
└──────────────────────────────────────────────────┘    │
                          │                             │
                          ▼                             │
╔══════════════════════════════════════════════════╗
║ ADD (residual)                             ◄ NEW ║◄───┘
║  output = input + FFN output                     ║
╚══════════════════════════════════════════════════╝
                          │
                          ▼
╔══════════════════════════════════════════════════╗
║ LAYERNORM                                  ◄ NEW ║
║  rescale each vector: (x − mean) ÷ std, × γ + β  ║
╚══════════════════════════════════════════════════╝
                          │
                          ▼
    Block output: same size as the input vectors
```

Learned so far: embedding and position tables; per block: W_Q, W_K, W_V per head, W_O, W1, W2, and LayerNorm γ and β (two sets).

This is the complete Transformer block, drawn in the original (post-norm) order. Most modern LLMs move each LayerNorm to just before attention and just before the FFN (pre-norm). The parts are the same; only the order changes.

---

## What's next

We now have one complete, self-contained Transformer Block — Attention, residual, LayerNorm, FFN, residual, LayerNorm. Chapter 8 covers what happens when you stack many of these blocks on top of each other, which is essentially what turns this single block into an actual GPT-style model.
