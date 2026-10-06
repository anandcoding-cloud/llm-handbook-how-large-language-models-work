# Chapter 16 — Modern Building Blocks: RoPE, RMSNorm and SwiGLU

## Introduction

Chapter 15 explained why decoder-only models dominate. This chapter and the next look at how they have been improved. The GPT-style model we built in Chapters 2 to 8 is the classic recipe. It learns a table of position vectors (Chapter 3), normalizes with LayerNorm (Chapter 7), and uses a two-matrix FFN with a GELU in the middle (Chapter 6). Open models such as Llama, Qwen, DeepSeek and Gemma have, over the last few years, converged on a noticeably different recipe. They didn't change the big picture, which is still stacked decoder blocks predicting the next token. They swapped out individual parts for better ones.

Attention itself has also been reworked, and Chapter 17 covers those variants. This chapter covers the other parts:

```text
Positions:     learned position table     →   RoPE (rotary position embeddings)
Normalization: LayerNorm (γ and β)        →   RMSNorm
FFN:           W1 → GELU → W2             →   SwiGLU (a gated FFN)
Small things:  biases, attention scores   →   no biases, QK-Norm
```

Each is a small, understandable idea, and by this point in the handbook you have all the background needed for each one.

---

## Part 1 — RoPE: rotating Q and K instead of adding a position vector

### What's wrong with a table of position vectors?

In Chapter 3, we gave the model a word's meaning plus a **learned position vector**, one table row per position (position 0, position 1, position 2, and so on), added to the embedding. It works, but it has two awkward properties:

1. **A hard length limit.** The table has a fixed number of rows. A model trained with 1,024 rows has no vector at all for position 1,025.
2. **It learns "position 7", not "three tokens apart".** Yet what attention really needs to know is mostly *how far apart two tokens are*. "The word right before me" matters wherever it happens to appear in the text.

Researchers wanted position information that depends on the **relative distance** between two tokens, and that doesn't require a table with a fixed number of rows.

### The RoPE idea

**Rotary position embedding (RoPE)** does something quite different. It doesn't add anything to the embedding. Instead, it **rotates the Query and Key vectors** by an angle that depends on the token's position. The farther along in the text, the larger the rotation.

Here is the smallest possible example. Use a 2-number vector and, to keep the arithmetic easy, an exaggerated rotation of 90° per position. Let `q = [1, 0]` and `k = [1, 0]` before rotation. Rotating `[1, 0]`:

```text
position 0 → rotate 0°     → [ 1,  0]
position 1 → rotate 90°    → [ 0,  1]
position 2 → rotate 180°   → [-1,  0]
position 3 → rotate 270°   → [ 0, -1]
```

Now compute the attention score (the dot product from Chapter 4) between a query at position `m` and a key at position `n`:

```text
query at position 2, key at position 0:   [-1, 0] · [ 1, 0] = -1      (distance 2)
query at position 3, key at position 1:   [ 0,-1] · [ 0, 1] = -1      (distance 2)
query at position 1, key at position 0:   [ 0, 1] · [ 1, 0] =  0      (distance 1)
query at position 4, key at position 3:   [ 1, 0] · [ 0,-1] =  0      (distance 1; 4 × 90° = 360° is a full turn)
query and key at the same position:       identical rotation  = +1     (distance 0)
```

Look at the pattern: **the score depends only on the distance `m − n`, never on the absolute positions.** Positions 2 and 0 give the same score as positions 3 and 1, because both pairs are 2 apart. That is exactly the property we wanted: attention automatically "knows" relative distance, without any position table.

### How it works in a real model

A real Query or Key has many numbers, not two. RoPE splits them into **pairs** and rotates each pair, with each pair turning at its **own speed**:

```text
Some pairs rotate quickly   →  good at telling nearby positions apart
Some pairs rotate slowly    →  good at telling far-apart positions apart
```

Think of the hands of a clock: the second hand distinguishes moments a few seconds apart, the minute hand distinguishes minutes, the hour hand distinguishes hours. Together they pin down time precisely. Several speeds together let the model sense both short-range and long-range distance.

Three points are worth being clear on, because each connects to something earlier in the handbook:

- **RoPE has no learned parameters.** The rotations are fixed math. It's another "pure math, nothing to learn" step like softmax, so the position-embedding table from Chapter 3 simply goes away.
- **It's applied to Q and K only, not V,** in every layer, right before the scores are computed. Positions only matter for deciding *where to look*, not for the content that gets copied.
- **It still goes through the causal mask and softmax as before.** Nothing else in attention changes.

### Why this helps with longer text

Because the rotation is a formula and not a table, nothing stops you from asking for position 100,000. But a model that only ever saw positions up to 4,000 during training has never seen rotations that large, and quality drops. So long-context models are typically **trained at a short length first, then extended**, adjusting the rotation speeds so that long distances look familiar to the model. Qwen3, for example, was trained at 4,096 tokens, then extended to 32,768 with these techniques (YaRN and Dual Chunk Attention), and DeepSeek-V3 was extended in two stages, from 4K to 32K and then to 128K.

### Variations on the idea

Some newer designs apply RoPE to only *part* of each head's numbers ("partial RoPE"). RoPE also needs special handling when K is stored in compressed form, which you'll see with DeepSeek's MLA in Chapter 17.

---

## Part 2 — RMSNorm: normalization with one less step

Recall LayerNorm from Chapter 7. For each token's vector it did three things: subtract the mean, divide by the standard deviation, then apply a learned scale `γ` and a learned shift `β`.

**RMSNorm** (root-mean-square normalization) is a simplification: **skip the mean subtraction, and skip the shift.** Just divide by the vector's root-mean-square, and apply a learned scale `γ`:

```text
RMS = √( average of the squares )
output = γ × (x / RMS)
```

Using the same vector from Chapter 7, `[100, 110, 90, 95]`:

```text
RMS = √( (100² + 110² + 90² + 95²) / 4 ) = √(39,225 / 4) ≈ 99.03

x / RMS = [1.010, 1.111, 0.909, 0.959]

Compare, for the same vector:
  LayerNorm:  [ 0.17,  1.52, -1.18, -0.51]    (centered on zero, spread of 1)
  RMSNorm:    [ 1.010, 1.111, 0.909, 0.959]   (just rescaled; not centered)
```

Both put the numbers on a manageable scale, which is the whole reason normalization exists (Chapter 7). RMSNorm does it with less work: one fewer statistic to compute, and no `β` to store. Researchers found that the mean-centering step contributed little in practice, and models trained with RMSNorm do as well as with LayerNorm. Since normalization is applied twice in every block of dozens of blocks, small savings add up. Almost all modern open models use RMSNorm, in the **pre-norm** position (before attention and before the FFN), as described in Chapter 7.

---

## Part 3 — SwiGLU: a gated FFN

Recall the FFN from Chapter 6: expand with `W1`, apply an activation (GELU), compress with `W2`. Modern models use a variant called **SwiGLU**, which adds a **gate**.

### The idea of a gate

Instead of one expansion followed by an activation, there are **two parallel expansions** of the same input. One produces the *content*. The other produces a *gate*, whose job is to decide how much of each piece of content to let through. The two are multiplied together, number by number:

```text
              ┌──►  × W_up    ──────────────► content ─┐
x (one token) │                                          × (multiply, element by element) ──►  × W_down  ──►  output
              └──►  × W_gate  ──► SiLU ──────► gate  ───┘
```

In formula form:

```text
FFN(x) = ( SiLU(x · W_gate)  ⊙  (x · W_up) ) · W_down
```

where `⊙` means multiply element by element, and `SiLU(z) = z × sigmoid(z)` is a smooth activation similar in spirit to GELU (it's sometimes called "Swish", which is where the "Sw" in SwiGLU comes from; "GLU" stands for gated linear unit).

### A worked example: the gate opening and closing

Take one feature, where the content branch produced `4`. The gate branch is passed through SiLU:

```text
Gate pre-activation = +2    →  SiLU(2)  = 2 × 0.881 =  1.76   →  4 × 1.76  =  7.05   (gate open: content passes)
Gate pre-activation = −3    →  SiLU(−3) = −3 × 0.047 = −0.14   →  4 × (−0.14) = −0.57  (gate nearly closed: content suppressed)
```

The same content value gets through strongly or is nearly shut off, depending on what the *gate* says, and the gate is itself computed from the input by a learned matrix. In the mango analogy from Chapter 6, the activation used to be a fixed grading rule applied to each reading. Here the model also learns, through `W_gate`, *which* readings deserve to be passed along. It's a more flexible kind of decision.

### Keeping the size comparable

SwiGLU uses **three** matrices (`W_gate`, `W_up`, `W_down`) instead of two. To avoid simply making the model bigger, the hidden size is reduced from the usual `4d` to roughly `8d/3`:

```text
Classic FFN:  2 matrices of size d × 4d       →  2 × 4d²    =  8d²  parameters
SwiGLU:       3 matrices of size d × (8d/3)   →  3 × 8d²/3  =  8d²  parameters
```

The same parameter count, in a more expressive shape. (Llama 7B has `d = 4096` and an FFN hidden size of 11,008, about 2.7 times `d`, which is this idea in practice.) Researchers found that gated FFNs perform better than the plain two-matrix version at equal size, and SwiGLU is now the standard choice in open models.

---

## Part 4 — A few small refinements

**No biases.** A linear layer can add a bias term, `Y = XW + b`, which our examples in Chapters 6 and 8 left out for simplicity, as many implementations do. Modern models mostly drop biases in their linear layers, which saves a few parameters with no loss in quality. (Earlier Qwen versions kept a bias in the Q, K and V projections; Qwen3 removed it.)

**QK-Norm.** Attention scores come from the dot product of Q and K. If those vectors grow large during training, the scores explode, and softmax collapses into near-one-hot choices (Chapter 4 explained why). **QK-Norm** applies a normalization to Q and to K *before* the dot product, keeping the scores in a stable range as an added safeguard on top of the `√dₖ` scaling. Qwen3 introduced it in place of the old QKV bias to keep training stable.

**Tied embeddings in smaller models.** Chapter 8 explained weight tying, reusing the embedding table as the LM head. Smaller models often do this to save parameters, since the embedding table is a large share of a small model, while the biggest models more often keep them separate.

---

## The complete picture

Here is the "classic" block next to the typical modern open-model block:

| Component | Classic (GPT-2 era) | Modern open model (Llama / Qwen / DeepSeek style) |
|---|---|---|
| Position information | Learned position table, added to embeddings | **RoPE**: rotate Q and K |
| Attention heads | Multi-head (MHA) | **GQA** (or **MLA**), see Chapter 17 |
| Normalization | LayerNorm (`γ`, `β`) | **RMSNorm** (`γ` only) |
| Norm placement | Post-norm (original paper) | **Pre-norm** |
| FFN | `W1` → GELU → `W2` | **SwiGLU** (gated, 3 matrices) |
| Biases | Yes | Mostly removed |
| Attention stability | Just `√dₖ` scaling | Plus **QK-Norm** in some |

The overall flow is unchanged: embeddings, N stacked blocks of attention then FFN with residuals, an LM head, softmax. Each row above is a *drop-in replacement* for one part, which is why the changes are easy to describe, and why you could swap them into the GPT diagram of Chapter 8 without changing its shape.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "RoPE adds a position vector to the embedding, just done differently." | It adds nothing. It rotates Q and K, and the score ends up depending on the *distance* between tokens. |
| "RoPE has learned parameters like the old position table." | RoPE is fixed math: nothing to train. |
| "RoPE alone lets a model handle any length." | Models still need to be trained, or extended, for long context, because they haven't seen very large rotations. |
| "RMSNorm is a different technique from LayerNorm." | It is LayerNorm with the mean subtraction and the shift removed: a simplification that works as well and costs less. |
| "SwiGLU is just a different activation function." | It changes the FFN's structure: a second, parallel branch (the gate) multiplied with the content branch. The activation is only one part. |
| "SwiGLU makes the FFN much bigger." | The hidden size is reduced to about 8d/3 so the parameter count stays about the same as the classic FFN. |

---

## Quick reference

```text
RoPE      rotate Q and K by an angle ∝ position; score depends on relative distance;
          no parameters; applied to Q, K only
RMSNorm   x / √(mean of squares) × γ        (no mean subtraction, no β)
SwiGLU    FFN(x) = ( SiLU(x·W_gate) ⊙ (x·W_up) ) · W_down      (hidden ≈ 8d/3)
QK-Norm   normalize Q and K before the dot product (stability)
No biases in most linear layers; pre-norm; GQA or MLA for attention
```

---

## What's next

Attention has been reworked as well as these parts. Chapter 17 covers the attention variants that modern models use to keep long text affordable: multi-query, grouped-query, latent, sliding-window, sparse and hybrid attention.
