# Appendix A — TinyGPT by Hand: One Complete Training Step with Real Numbers

## Introduction

Throughout this handbook we have described a GPT in words, diagrams and small fragments of arithmetic. This appendix does something different: it runs **one complete training step on a miniature GPT, with every number visible**. We follow a single training sentence from raw text, through the forward pass, to a loss, back through every operation to get gradients, and finally through an AdamW update. Then we run the forward pass a second time and watch the loss go down.

The model is tiny, but it is a *real* GPT in structure. It has token and position embeddings, a causal multi-head attention layer, residual connections, LayerNorm, a feed-forward network, an LM head, a cross-entropy loss, backpropagation and AdamW. Every operation is exactly the one from Chapters 2 to 11, just with 4 numbers per token instead of 4,096.

A few notes on how to read it:

- **Every number was computed by a program, not invented.** Each value comes from the previous one, in the same order a framework like PyTorch would compute them. The tables show numbers rounded to three decimals (the program keeps full precision), so if you redo a step by hand with the rounded numbers, the last digit may differ slightly. The program itself, `Appendix-A-tinygpt.js`, sits next to this file. Run it with `node Appendix-A-tinygpt.js` to print every number again.
- **The gradients were checked.** The backward pass was done with the manual formulas of Chapter 10, and then verified against an independent method: nudging each of the 204 parameters by a tiny amount and measuring how the loss changes. The two agree to about 1 part in 10 million (largest relative difference 4.5e-7).
- **One simplification:** the FFN has a hidden width of 8 (twice the model width) instead of the usual 4× (which would be 16), only so that the matrices fit on a page. Biases are left out, as in many modern models.
- **The starting values are arbitrary.** A real model starts from random numbers, and so does this one. The numbers were chosen by hand or filled in with small one-decimal values. What matters is not the particular values but that every later number follows from them.

---

## Part 1 — The model and its 204 numbers

### The configuration

```text
Vocabulary size        = 6 words
Embedding dimension    = 4        (call it E)
Attention heads        = 2
Head dimension         = 2        (heads × head dimension = 2 × 2 = 4 = E)
Transformer blocks     = 1        (original "post-norm" order of Chapter 7)
FFN hidden width       = 8
Sequence length        = 3 tokens
```

### The vocabulary and the training sentence

| ID | Word |
|---:|---|
| 0 | The |
| 1 | cat |
| 2 | sat |
| 3 | dog |
| 4 | ran |
| 5 | END |

We train on a single sentence, **"The cat sat"**, whose token IDs are `[0, 1, 2]`. Recall from Chapter 9 that one forward pass over this sentence produces *three* training examples at once, one per position, thanks to the causal mask:

| Position | The model has seen | It must predict |
|---:|---|---|
| 1 | The | cat |
| 2 | The cat | sat |
| 3 | The cat sat | END |

### Every learned parameter

| Parameter | Shape | Count |
|---|---|---:|
| Token embedding table | 6 × 4 | 24 |
| Position embedding table | 3 × 4 | 12 |
| W_Q, W_K, W_V for 2 heads (6 matrices) | 6 × (4 × 2) | 48 |
| W_O | 4 × 4 | 16 |
| LayerNorm 1 (γ and β) | 4 + 4 | 8 |
| FFN W1 (expand) | 4 × 8 | 32 |
| FFN W2 (compress) | 8 × 4 | 32 |
| LayerNorm 2 (γ and β) | 4 + 4 | 8 |
| LM head | 4 × 6 | 24 |
| **Total** | | **204** |

Here are all of them. These are the only numbers the model "knows". Everything else in this appendix is computed from them and the input.

**Token embedding table** (Chapter 3), one row per word:

```text
The     0.20   0.60   0.40   0.10
cat     0.70   0.10   0.20   0.80
sat     0.50   0.80   0.90   0.20
dog     0.90   0.20   0.40   0.60
ran     0.30   0.40   0.70   0.10
END     0.10   0.90   0.30   0.50
```

**Position embedding table** (Chapter 3), one row per position:

```text
pos 0   0.10   0.00   0.20   0.10
pos 1   0.00   0.30   0.10   0.00
pos 2   0.20   0.10   0.00   0.40
```

**Attention matrices** (Chapters 4 and 5), each 4 × 2. Head 1 and head 2 have different matrices, as Chapter 5 explained:

```text
W_Q (head 1)          W_K (head 1)          W_V (head 1)
  0.4   0.1            0.3  -0.1            0.5   0.1
  0.3  -0.2            0.2   0.4           -0.2   0.3
 -0.2   0.6           -0.3   0.2            0.4  -0.1
  0.5   0.2            0.1   0.5            0.2   0.6

W_Q (head 2)          W_K (head 2)          W_V (head 2)
 -0.3   0.2            0.4   0.3            0.2  -0.3
  0.5   0.1           -0.1   0.2            0.4   0.5
  0.1  -0.4            0.3  -0.2           -0.1   0.2
  0.2   0.3            0.2   0.1            0.3   0.1
```

**Output projection W_O** (4 × 4), **LayerNorm** parameters (γ = 1 and β = 0 at the start, for both LayerNorms), and the **FFN** and **LM head** matrices:

```text
W_O (4 × 4)
  0.2   0.1  -0.1   0.3
  0.5  -0.2   0.4   0.1
 -0.3   0.6   0.2  -0.2
  0.1   0.2   0.3   0.4

W1 (4 × 8)
  0.1  -0.2   0.1   0.4   0.1   0.1  -0.3   0.3
  0.2   0.3   0.2  -0.1   0.2  -0.2   0.1  -0.2
 -0.2   0.2   0.3   0.3  -0.3   0.0   0.2  -0.1
 -0.2  -0.1   0.3   0.3  -0.1  -0.1   0.3  -0.3

W2 (8 × 4)
 -0.3  -0.1  -0.1   0.0
  0.4   0.0   0.2  -0.2
 -0.1   0.3   0.3  -0.2
  0.0   0.3   0.2  -0.3
  0.1  -0.3   0.2  -0.1
  0.2  -0.2  -0.1  -0.1
 -0.1   0.4  -0.1  -0.4
 -0.2  -0.3   0.1  -0.4

LM head (4 × 6)
 -0.2  -0.2  -0.1  -0.2   0.2   0.0
  0.0   0.3   0.1   0.2  -0.4   0.0
  0.0   0.0   0.1  -0.2   0.0   0.4
 -0.2   0.0   0.3   0.1  -0.4   0.1
```

---

## Part 2 — The forward pass

We follow the order of the architecture diagram from Chapter 8, using the post-norm block of Chapter 7.

### Step 1 — Tokenize and look up embeddings (Chapters 2 and 3)

`"The cat sat"` becomes `[0, 1, 2]`. Each ID selects a row of the token table, and each position selects a row of the position table. They are **added**, and the sum is the first real input of the Transformer. From now on the token IDs are gone, and the model only sees vectors:

```text
Token    Word embedding              + Position embedding         =  X (input to the block)
The      0.20  0.60  0.40  0.10       0.10  0.00  0.20  0.10       0.30  0.60  0.60  0.20
cat      0.70  0.10  0.20  0.80       0.00  0.30  0.10  0.00       0.70  0.40  0.30  0.80
sat      0.50  0.80  0.90  0.20       0.20  0.10  0.00  0.40       0.70  0.90  0.90  0.60
```

`X` is a 3 × 4 matrix: 3 tokens, 4 numbers each.

### Step 2 — Head 1: Q, K and V (Chapter 4)

Each head projects every token through its own three matrices: `Q = X·W_Q`, `K = X·W_K`, `V = X·W_V`. As one worked multiplication, **Q for "The"** in head 1 is the row `[0.30, 0.60, 0.60, 0.20]` times the 4 × 2 matrix `W_Q`:

```text
first number:   0.30×0.4 + 0.60×0.3 + 0.60×(−0.2) + 0.20×0.5 = 0.12 + 0.18 − 0.12 + 0.10 = 0.28
second number:  0.30×0.1 + 0.60×(−0.2) + 0.60×0.6 + 0.20×0.2 = 0.03 − 0.12 + 0.36 + 0.04 = 0.31
So Q(The) = [0.28, 0.31]
```

Doing this for every token and all three matrices gives (head 1):

```text
Token        Q                  K                  V
The       0.280   0.310       0.050   0.430       0.310   0.270
cat       0.740   0.330       0.280   0.550       0.550   0.640
sat       0.670   0.550       0.180   0.770       0.650   0.610
```

### Step 3 — Head 1: scores, causal mask, scaling, softmax

The raw score for query token *i* and key token *j* is the dot product `Q(i)·K(j)`. For example, "cat" asking about "cat" is `0.74×0.28 + 0.33×0.55 = 0.389`. The whole matrix `Q·Kᵀ`:

```text
              The      cat      sat
The      0.147    0.249    0.289
cat      0.179    0.389    0.387
sat      0.270    0.490    0.544
```

Now three things happen, in the order a typical implementation does them. **Scale** every score by `1/√2 ≈ 0.707` (the head dimension is 2). **Mask** the future (replace the upper triangle by −∞). Then **softmax** each row:

```text
After scaling and masking:
              The      cat      sat
The      0.104     -inf     -inf
cat      0.127    0.275     -inf
sat      0.191    0.347    0.385

After softmax (attention weights, each row adds up to 1):
              The      cat      sat
The      1.000    0.000    0.000
cat      0.463    0.537    0.000
sat      0.296    0.345    0.359
```

Read the "cat" row: when the model builds a new representation for *cat*, it uses **46% of "The"'s information and 54% of "cat"'s own**, and **0% of "sat"**, which is the future. The attention weights are close to uniform because the model has not been trained yet. That will matter in the backward pass.

### Step 4 — Head 1: the weighted sum of V

Each token's output is its attention row times the V vectors. For "cat": `0.463×V(The) + 0.537×V(cat)` = `[0.439, 0.469]`.

```text
Head 1 output (3 × 2):
The      0.310    0.270
cat      0.439    0.469
sat      0.515    0.520
```

### Step 5 — Head 2, concatenation and W_O (Chapter 5)

Head 2 repeats Steps 2 to 4 with its own matrices. Its queries, keys and values are:

```text
Token        Q                  K                  V
The       0.310  -0.060       0.280   0.110       0.300   0.350
cat       0.180   0.300       0.490   0.310       0.510   0.130
sat       0.450   0.050       0.580   0.270       0.590   0.480
```

After the same scale, mask and softmax steps, its attention weights and outputs are:

```text
Head 2 attention weights:                      Head 2 output (3 × 2):
              The      cat      sat
The      1.000    0.000    0.000             0.300    0.350
cat      0.483    0.517    0.000             0.409    0.236
sat      0.314    0.338    0.348             0.472    0.321
```

The two heads' outputs are placed side by side (2 + 2 = 4 numbers per token), and `W_O` lets the heads mix:

```text
Concatenated C (3 × 4):                        After W_O (Attn):
The     0.310   0.270   0.300   0.350            0.142   0.227   0.242   0.200
cat     0.439   0.469   0.409   0.236            0.223   0.243   0.296   0.191
sat     0.515   0.520   0.472   0.321            0.253   0.295   0.347   0.240
```

### Step 6 — Residual connection and LayerNorm 1 (Chapter 7)

The attention output is **added back** to the block's input (the residual connection), and then each row is **normalized**:

```text
R1 = X + Attn                                   H1 = LayerNorm(R1)
The     0.442   0.827   0.842   0.400           -0.896   0.961   1.033  -1.098
cat     0.923   0.643   0.596   0.991            0.787  -0.850  -1.121   1.184
sat     0.953   1.195   1.247   0.840           -0.628   0.809   1.120  -1.301
```

A worked LayerNorm for the row of "The": the mean of `[0.442, 0.827, 0.842, 0.400]` is 0.6278, the variance is 0.0430, so the standard deviation is 0.2074 (a tiny ε = 10⁻⁵ is added inside the square root). Each number becomes `(x − 0.628) / 0.207`, giving `[-0.896, 0.961, 1.033, -1.098]`, which has mean 0 and standard deviation 1. With γ = 1 and β = 0, that is the output.

### Step 7 — The feed-forward network (Chapter 6)

Each token is processed **independently** (no mixing between tokens here). Expand 4 → 8 with `W1`, apply GELU, and compress 8 → 4 with `W2`:

```text
Z = H1·W1  (3 × 8)
  0.116   0.784   0.083  -0.474  -0.098  -0.172   0.242  -0.235
 -0.104  -0.755  -0.072   0.419   0.126   0.130  -0.190   0.163
  0.135   0.723   0.045  -0.387  -0.107  -0.095   0.103  -0.072

G = GELU(Z)   (for example GELU(0.784) = 0.614)
  0.063   0.614   0.044  -0.151  -0.045  -0.074   0.144  -0.096
 -0.048  -0.170  -0.034   0.277   0.070   0.072  -0.081   0.092
  0.075   0.553   0.023  -0.135  -0.049  -0.044   0.056  -0.034

F = G·W2  (3 × 4)
  0.208   0.076   0.074  -0.094
 -0.039  -0.017   0.040  -0.061
  0.184   0.015   0.069  -0.074
```

### Step 8 — Residual connection and LayerNorm 2

```text
R2 = H1 + F                                     H2 = LayerNorm(R2)
The    -0.688   1.037   1.107  -1.192           -0.738   0.950   1.019  -1.231
cat     0.747  -0.867  -1.081   1.123            0.793  -0.877  -1.098   1.182
sat    -0.444   0.824   1.189  -1.376           -0.482   0.759   1.117  -1.394
```

`H2` is the block's output: one **contextual embedding** (Chapter 8) per token. It is a 3 × 4 matrix, the same shape as the block's input, which is what lets blocks be stacked.

### Step 9 — LM head, softmax and the loss (Chapters 8 and 9)

The LM head turns each 4-number vector into 6 scores (**logits**), one per word, and softmax turns each row into probabilities:

```text
Logits (3 × 6):
          The     cat     sat     dog     ran     END
The     0.394   0.433  -0.099   0.011  -0.035   0.284
cat    -0.395  -0.422   0.078   0.004   0.037  -0.321
sat     0.375   0.324  -0.182  -0.114   0.157   0.307

Probabilities (softmax):
          The     cat     sat     dog     ran     END
The     0.205   0.213   0.125   0.140   0.133   0.184
cat     0.130   0.127   0.209   0.194   0.200   0.140
sat     0.205   0.195   0.117   0.126   0.165   0.192
```

The untrained model is nearly clueless: every word gets roughly 1/6 ≈ 0.167 of the probability. Now compare each row with its target, using cross-entropy `−log(probability of the correct word)`:

| Position | Target | Probability given to it | Loss = −log(p) |
|---:|---|---:|---:|
| 1 (after "The") | cat | 0.213 | 1.546 |
| 2 (after "The cat") | sat | 0.209 | 1.567 |
| 3 (after "The cat sat") | END | 0.192 | 1.652 |
| **Average** | | | **1.588** |

A sanity check: a model that guessed uniformly among 6 words would have loss `ln 6 = 1.792`. Ours is 1.588, about the same. This single number, **1.588**, is what the backward pass starts from.

### The same pass, as tensor shapes

Real implementations carry an extra *batch* dimension `B` (here `B = 1`). Here is the pass above written the way framework code writes it, with `T = 3` tokens, `E = 4`, `H = 2` heads and head dimension `D = 2`:

| Step | Shape | Our numbers |
|---|---|---|
| Input embeddings `X` | `(B, T, E)` | (1, 3, 4) |
| After `X·W_Q` (all heads at once) | `(B, T, H·D)` | (1, 3, 4) |
| Split into heads and move heads forward | `(B, H, T, D)` | (1, 2, 3, 2) |
| Attention scores `Q·Kᵀ` | `(B, H, T, T)` | (1, 2, 3, 3) |
| After softmax | `(B, H, T, T)` | (1, 2, 3, 3) |
| Weighted sum with `V` | `(B, H, T, D)` | (1, 2, 3, 2) |
| Concatenate heads | `(B, T, E)` | (1, 3, 4) |
| After `W_O`, residual, LayerNorm | `(B, T, E)` | (1, 3, 4) |
| FFN expand / compress | `(B, T, 8)` / `(B, T, E)` | (1, 3, 8) / (1, 3, 4) |
| Logits | `(B, T, vocab)` | (1, 3, 6) |

The rule that makes the block stackable: **it takes `(B, T, E)` and returns `(B, T, E)`**. A line of real code such as `q = self.wq(x).view(B, T, n_heads, head_dim).transpose(1, 2)` now reads as a sentence: *"project the tokens into queries, then split the embedding dimension into heads and move the head dimension forward."*

---

## Part 3 — The backward pass

Now everything runs in reverse (Chapter 10). Remember what flows backward: not the loss itself, but the **gradient of the loss** with respect to each intermediate value. Every operation does two jobs: it computes the gradient for its own **parameters** (if it has any), and the gradient for its **inputs**, which it hands to the operation before it.

### Step B1 — Where it starts: the gradient of the logits

For softmax followed by cross-entropy, the gradient of the loss with respect to the logits is wonderfully simple: **the predicted probabilities minus the one-hot target**, divided by the number of positions (3), because the loss is an average. For position 1, whose target is *cat*:

```text
probabilities:   [0.205, 0.213, 0.125, 0.140, 0.133, 0.184]
minus one-hot:   [0, 1, 0, 0, 0, 0]
divided by 3  →  [0.0683, -0.2623, 0.0417, 0.0466, 0.0445, 0.0612]
```

The correct word gets a **negative** gradient ("raise this logit") and every other word a small positive one ("lower these"). For all three positions:

```text
d(loss)/d(logits):
          The      cat      sat      dog      ran      END
The     0.0683  -0.2623   0.0417   0.0466   0.0445   0.0612
cat     0.0434   0.0422  -0.2637   0.0646   0.0668   0.0467
sat     0.0684   0.0650   0.0392   0.0419   0.0550  -0.2694
```

Each row adds up to zero, as it must.

### Step B2 — The LM head

The LM head computed `logits = H2 · W_head`. For a matrix multiplication, the gradient of the weights is `H2ᵀ · d(logits)` and the gradient passed back to its input is `d(logits) · W_headᵀ`. As one worked entry, the gradient of `W_head[0][1]` (the weight from the first embedding number to the *cat* logit) is the sum over the three positions of *(that embedding number) × (that position's logit gradient for cat)*:

```text
dW_head[0][1] = -0.738 × -0.2623  +  0.793 × 0.0422  +  -0.482 × 0.0650  =  0.1957
```

The full gradient matrix for the LM head:

```text
dW_head (4 × 6):
          The      cat      sat      dog      ran      END
e1     -0.0490   0.1957  -0.2589  -0.0033  -0.0064   0.1218
e2      0.0788  -0.2369   0.3007   0.0194   0.0254  -0.1874
e3      0.0983  -0.2410   0.3758   0.0233   0.0334  -0.2898
e4     -0.1281   0.2822  -0.4176  -0.0394  -0.0525   0.3554
```

### Steps B3 to B7 — back through the block

The gradient now travels back through the block in the exact reverse order of the forward pass. Each step uses a rule you already know:

| Backward step | What it does |
|---|---|
| **LayerNorm 2** | Computes gradients for γ₂ and β₂, and passes a gradient to `R2`. (The formula is longer than for a plain multiplication, because every number in a row affects that row's mean and variance, and so every other number.) |
| **Residual connection** | An addition **copies** the incoming gradient to both of its inputs: one copy goes down into the FFN, and one goes straight to `H1`. This "shortcut" is why gradients reach the early layers so easily. |
| **FFN, `W2`** | `dW2 = Gᵀ · dF`; the gradient passed back is `dF · W2ᵀ`. |
| **GELU** | Multiplies the gradient by GELU's slope at each `Z` value (a fixed function with no parameters, Chapter 6). |
| **FFN, `W1`** | `dW1 = H1ᵀ · dZ`; the gradient to `H1` is `dZ · W1ᵀ`, **added** to the shortcut copy from the residual. |
| **LayerNorm 1, then residual** | Same as before: gradients for γ₁ and β₁, then the gradient is copied to both the attention path and the shortcut to `X`. |

### Step B8 — W_O and the attention heads

`dW_O = Cᵀ · dAttn`, and the gradient passed back, `dC`, is split into one piece per head. Inside each head, the weighted sum `O = A·V` sends one gradient to `V` (and to `W_V`) and one to the attention weights `A`. The gradient on `A` then passes backward through **softmax** (each row: `dS = A ⊙ (dA − Σ A·dA)`), through the **scaling** by 1/√2, and into `Q·Kᵀ`, which splits into gradients for `Q` and `K` and, from there, for `W_Q` and `W_K`. The masked (future) positions had attention weight exactly 0, so they receive exactly zero gradient.

### Step B9 — The embeddings

The input `X` was the **sum** of a token embedding and a position embedding, and the gradient of a sum goes to both parts. Each token's row is collected into the embedding table. Notice what that means: **only the rows for words that were actually used get a gradient.** "The", "cat" and "sat" receive gradients. **"dog" and "ran" receive exactly zero**, because they were never looked up in this sentence. Position rows 0, 1 and 2 each receive a gradient.

### What the gradients look like

Every one of the 204 parameters now has a gradient. A summary of how big they are (the overall size, or norm, of each gradient matrix):

| Parameter | Size of its gradient |
|---|---:|
| LM head | 1.00e+0 |
| Token embeddings | 7.54e−1 |
| Position embeddings | 7.54e−1 |
| W_O | 5.68e−1 |
| LayerNorm 2 β / γ | 2.20e−1 / 1.07e−1 |
| W_V, head 2 | 1.63e−1 |
| W_V, head 1 | 1.20e−1 |
| LayerNorm 1 β / γ | 1.31e−1 / 2.98e−2 |
| FFN W1 | 5.32e−2 |
| FFN W2 | 2.54e−2 |
| W_Q, head 2 | 3.12e−3 |
| W_K, head 1 | 2.92e−3 |
| W_Q, head 1 | 1.92e−3 |
| W_K, head 2 | 1.85e−3 |

(The token and position embeddings have identical norms because each word appears exactly once in this sentence, so each position's gradient flows into exactly one token row.)

Two things stand out. The gradients are largest **near the loss** (the LM head) and smaller further back, which is the usual picture. And the gradients for `W_Q` and `W_K` are **tiny**, hundreds of times smaller than the LM head's. That is not a bug: because the untrained attention weights are nearly uniform (Step 3), changing the query–key scores barely changes the outputs yet, so the loss is not sensitive to them. They will matter once the model is a little less clueless. This is a nice example of why **AdamW**, which adapts the step to each parameter's own gradient size, is useful, as the next part shows.

### The gradient check

To make sure none of this is wrong, the program also measured each parameter's effect on the loss by brute force: nudge the parameter up by 0.00001, then down by 0.00001, recompute the loss both times, and take the slope. The slope from backpropagation and the slope from this brute-force method **agree for all 204 parameters**, with a largest relative difference of 4.5e-7. That is the whole content of "backpropagation computes exact gradients, only much faster".

---

## Part 4 — The AdamW update

We now have a gradient for every parameter, and we apply one **AdamW** step (Chapter 11) to all of them. The settings:

```text
learning rate  η  = 0.05       (large, so the effect is visible in one step)
β₁ = 0.9      β₂ = 0.999    ε = 10⁻⁸
weight decay   λ  = 0.01
```

For each parameter, with gradient `g`, and `m` and `v` both starting at 0:

```text
m  = β₁·m + (1 − β₁)·g              (momentum: smoothed gradient)
v  = β₂·v + (1 − β₂)·g²             (smoothed squared gradient)
m̂  = m / (1 − β₁ᵗ)     v̂ = v / (1 − β₂ᵗ)   (bias correction, t = step number)
w ← w − η · m̂ / (√v̂ + ε)  −  η·λ·w          (the step, plus weight decay)
```

### The first step has a special shape

On the very first step, `m = 0.1·g` and `v = 0.001·g²`. After bias correction, `m̂ = g` and `v̂ = g²`, so `m̂ / √v̂ = g / |g|`, which is just **+1 or −1**. So on step 1, **every parameter moves by about η = 0.05, in the direction opposite to its gradient's sign, no matter how large or small the gradient is.** Three examples:

| Parameter | Old value | Gradient | m̂ | v̂ | New value | Change |
|---|---:|---:|---:|---:|---:|---:|
| `W_head[0][1]` | -0.2000 | 1.96e-1 | 1.96e-1 | 3.83e-2 | -0.2499 | -0.0499 |
| `W_Q head 1 [0][0]` | 0.4000 | 5.07e-4 | 5.07e-4 | 2.57e-7 | 0.3498 | -0.0502 |
| `embedding of 'dog' [0]` | 0.9000 | 0.00e+0 | 0.00e+0 | 0.00e+0 | 0.8996 | -0.0004 |

- **`W_head[0][1]`** has a large gradient (0.196) and moves by about 0.050.
- **`W_Q` of head 1** has a gradient several hundred times smaller (5.1e-4), yet it also moves by about 0.050. That is Adam's per-parameter scaling (dividing by √v̂) at work. Plain gradient descent would barely have touched it.
- **The embedding of "dog"** has a gradient of exactly 0, so `m̂ = 0` and the Adam part contributes nothing. It only shrinks slightly through weight decay (0.9000 to 0.8996). A word that doesn't appear in the data isn't learned.

(Real training doesn't take such a bold first step. Chapter 11 mentioned that the learning rate usually starts small and "warms up".)

---

## Part 5 — Did it learn? The second forward pass

We now run the forward pass again with the updated parameters, on the same sentence:

| Position | Target | P(target) before | P(target) after | Loss before | Loss after |
|---:|---|---:|---:|---:|---:|
| 1 (after "The") | cat | 0.213 | 0.302 | 1.546 | 1.197 |
| 2 (after "The cat") | sat | 0.209 | 0.276 | 1.567 | 1.289 |
| 3 (after "The cat sat") | END | 0.192 | 0.289 | 1.652 | 1.243 |
| **Average loss** | | | | **1.588** | **1.243** |

One update, using nothing but the arithmetic above, moved the loss from **1.588** to **1.243**, and each of the three target words became more probable. Repeat this for millions of sentences, and the random matrices become a language model.

Don't over-read the size of the drop: we trained on a single sentence with a deliberately large learning rate, so the model is memorizing it. The point is the *mechanism*. Every step in the loop is exactly what happens at full scale.

---

## Part 6 — What TinyGPT teaches

### What is random, and what is mathematics?

The first question many people have is whether a GPT is "magically" learning from random numbers. Now we can answer it precisely. The only random things are the **starting parameters**: the token and position embeddings, `W_Q`, `W_K`, `W_V`, `W_O`, `W1`, `W2`, the LM head, and (in a real model) the LayerNorm parameters, plus things like the order of the training data. Everything else is **deterministic mathematics**: embedding lookup, matrix multiplication, dot products, scaling, masking, softmax, weighted sums, residual adds, LayerNorm, GELU, cross-entropy, backpropagation and AdamW. There is no hidden step.

### Where each step is explained

| In this appendix | Chapter |
|---|---|
| Tokenizing | 2 |
| Embeddings and position embeddings | 3 |
| Q, K, V, scores, the causal mask, softmax | 4 |
| Two heads, concatenation, `W_O` | 5 |
| FFN and GELU | 6 |
| Residuals and LayerNorm; the block | 7 |
| Stacking, the LM head, logits | 8 |
| Next-token targets and cross-entropy | 9 |
| The backward pass | 10 |
| AdamW | 11 |

### TinyGPT versus a real model

| | TinyGPT | A modern LLM |
|---|---|---|
| Parameters | 204 | billions |
| Embedding width | 4 | 4,096 or more |
| Blocks | 1 | 30 to 100+ |
| Heads | 2, each of size 2 | 32 or more, each of size 128 |
| FFN hidden width | 8 | about 4× the width (or the gated SwiGLU form, Chapter 16) |
| Vocabulary | 6 words | 30,000 to 250,000 tokens |
| Context | 3 tokens | thousands to a million |
| Batch | 1 sentence | thousands of sequences |
| Position information | learned table | usually rotary (RoPE, Chapter 16) |
| Norm placement | after (post-norm) | before (pre-norm), usually RMSNorm |
| Attention | full multi-head | GQA, MLA, sliding window and more (Chapter 17) |

Everything in the right-hand column is a change to *one* part of the left-hand model. The overall loop is the same: **embed, attend, think, repeat, predict, measure the error, send it back, nudge every weight.**

The sources behind the whole handbook (papers, documentation and the study material it was built on) are listed in **References**, the next section.
