# Chapter 17 — Attention Variants: MQA, GQA, MLA, Sliding Window and FlashAttention

## Introduction

In Chapters 4 and 5 we built attention exactly as the original Transformer defined it: many heads, each with its own `W_Q`, `W_K`, `W_V`, with every token comparing itself against every earlier token. That design is elegant and powerful. It is also expensive, and the expense gets worse as text gets longer.

Real models don't all use that original design. Llama, Qwen, DeepSeek, Gemma and others each use their own **attention variant**, and each variant is an answer to one specific cost problem. This chapter explains the problems first, then the variants, so that each one makes sense as a solution rather than a list of names to memorize.

One thing stays constant through all of this: attention is still *"compare queries with keys, softmax, blend values"* (Chapter 4). The variants change **how many keys and values are stored**, **which tokens are compared**, or **how the computation is carried out**, never the basic idea.

---

## Part 1 — The two costs of attention

### Cost 1: compute grows with the square of the text length

In attention, every token is scored against every earlier token. That's a grid of scores:

```text
1,000 tokens     →  about 1 million scores  (per head, per layer)
10,000 tokens    →  about 100 million
100,000 tokens   →  about 10 billion
```

Ten times more text means a hundred times more scores. This is why long context is hard.

### Cost 2: the memory that generation needs (the KV cache)

Recall from Chapter 12 how a model writes text: one token at a time, with each new token needing the Keys and Values of **all the earlier tokens**. Because an earlier token's Key and Value never change once computed (the causal mask of Chapter 4 guarantees it), the model stores them instead of recomputing them. That store is the **KV cache**.

The catch is that the KV cache grows with every token and has to be kept for every layer. Its size is:

```text
KV cache per token  =  2 (K and V)  ×  layers  ×  KV heads  ×  head dimension  ×  bytes per number
```

Take a model shaped like Llama 3 8B: 32 layers, 8 KV heads, head dimension 128, stored in 16-bit (2 bytes):

```text
2 × 32 × 8 × 128 × 2 bytes  =  131,072 bytes  =  128 KiB per token

   8,000 tokens of context  →  about  1 GiB
 128,000 tokens of context  →  about 16 GiB      (for ONE conversation)
```

At large context lengths, and when serving many users at once, this cache can take more memory than the model's own weights. Most of the variants below exist to shrink it, or to cut the compute cost, or both. Keep this formula in mind: every factor in it is something a variant can reduce.

---

## Part 2 — The baseline: multi-head attention (MHA)

This is Chapter 5's design. With `H` heads, there are `H` query heads **and** `H` separate sets of Keys and Values, one per head:

```text
Head 1:  Q₁  K₁  V₁
Head 2:  Q₂  K₂  V₂
  ...
Head H:  Q_H K_H V_H          ← H sets of K and V must be cached
```

It works well, and it's the reference point for quality. GPT-2 uses it. Its weakness is the one in the formula above: `KV heads = H`, which makes the cache large.

---

## Part 3 — Multi-query attention (MQA): share one K and V

The first idea is blunt. **Keep all the query heads, but let them all share a single set of K and V.**

```text
Q₁ ─┐
Q₂ ─┤
Q₃ ─┼──►  one shared K, V
Q₄ ─┘
```

Each head still asks its *own* question (its own `Q`), but they all search the same keys and read the same values. The KV cache shrinks by a factor of `H`; with 32 heads, it's 32 times smaller. The cost is some loss of quality, since every head now works from the same K and V, which gives the model less flexibility. MQA appeared in models such as PaLM.

---

## Part 4 — Grouped-query attention (GQA): the middle path

MHA keeps one K/V set per head (best quality, biggest cache). MQA keeps one K/V set in total (smallest cache, some quality loss). **GQA splits the difference: heads are divided into groups, and each group shares one K/V set.**

```text
8 query heads, 2 KV groups:

Q₁ Q₂ Q₃ Q₄  ──►  K₁, V₁   (group 1)
Q₅ Q₆ Q₇ Q₈  ──►  K₂, V₂   (group 2)

Cache: 2 K/V sets instead of 8   →   4× smaller than MHA
```

GQA is really a dial:

```text
KV groups = number of heads   →  this IS multi-head attention
KV groups = 1                 →  this IS multi-query attention
anything in between           →  grouped-query attention
```

In the matrices, the only change is the shape. `W_Q` still produces all `H` query heads, but `W_K` and `W_V` produce only `G` heads' worth of output (one per group), so they're smaller too. At attention time, each query head simply uses the K and V of its own group.

This is the default choice in most modern open models. Llama 3 8B uses 32 query heads with 8 KV groups (4 query heads per group), which is exactly where the `8` came from in the cache example above. Had it used plain MHA (32 KV heads), that cache would be 4 times larger: 512 KiB per token, or 64 GiB for a 128K-token conversation. Qwen3 and many others also use GQA.

---

## Part 5 — Multi-head latent attention (MLA): compress instead of share

GQA and MQA shrink the cache by *sharing* K and V between heads. DeepSeek took a different route: **keep every head's own K and V, but store them in compressed form.**

The idea is to cache one small vector per token, called a **latent** vector, instead of the full K and V:

```text
Standard:    token → K (all heads) and V (all heads)  →  cache both in full

MLA:         token → ONE small latent vector c   →  cache only c
                              │
                              └─► when needed, expand c back into K and V
                                   using learned "up-projection" matrices
```

Both the compression (a learned down-projection) and the expansion (learned up-projections) are ordinary matrices, trained along with everything else, exactly like `W_K` and `W_V` in Chapter 4. Implementations can even fold the expansion into neighboring matrices so the full K and V never need to be built at all.

The numbers for DeepSeek-V3 show how large the saving is. It has 128 heads with a per-head dimension of 128, so a standard cache would hold `128 × 128 × 2 = 32,768` numbers per token, per layer. MLA instead caches a latent of 512 numbers plus a small extra piece for position information (rotary position embeddings, from Chapter 16; see the note below), about 576 in total:

```text
Standard MHA (this model's size):     32,768 numbers per token per layer
GQA with 8 groups:                      2,048
MLA (DeepSeek-V3):                       ~576        →  roughly 57× smaller than MHA,
                                                         and about 3.5× smaller than GQA
```

A note on that small extra piece. RoPE (Chapter 16) rotates K by an amount that depends on position, and that rotation can't pass cleanly through MLA's compress-and-expand step. DeepSeek's fix is **decoupled RoPE**: K is split into a large compressed part that carries no position information, and a small separate part (64 numbers per token in DeepSeek-V3) that carries the rotation. That small part is the extra piece counted in the 576 above.

DeepSeek reported that, in their experiments, MLA kept quality competitive with full MHA while GQA and MQA lost some. The trade-off is complexity: MLA is considerably harder to implement than GQA. It's used by DeepSeek-V3 and several other large models, such as Kimi K2.

---

## Part 6 — Sliding-window attention: only look at recent tokens

The variants so far shrink the *memory*, but each token still looks at every earlier token. **Sliding-window attention** attacks the compute cost directly: each token attends only to the most recent `w` tokens.

```text
Full causal attention          Sliding window (w = 3)

      1 2 3 4 5 6                    1 2 3 4 5 6
  1   ✓                          1   ✓
  2   ✓ ✓                        2   ✓ ✓
  3   ✓ ✓ ✓                      3   ✓ ✓ ✓
  4   ✓ ✓ ✓ ✓                    4     ✓ ✓ ✓
  5   ✓ ✓ ✓ ✓ ✓                  5       ✓ ✓ ✓
  6   ✓ ✓ ✓ ✓ ✓ ✓                6         ✓ ✓ ✓
```

The cost per token is now fixed by `w` instead of growing with the text length, and the cache only needs to keep the last `w` tokens. Mistral 7B used a window of 4,096 tokens.

You might worry the model "forgets" anything older than the window. It's less severe than it sounds, because layers stack. A token at layer 1 sees `w` tokens back, but what it sees has already absorbed information from `w` tokens before *that*. After several layers, information can travel much farther than one window.

Still, some information needs a direct, long-range lookup. So many models **mix** local and global layers. Gemma 3, for example, alternates 5 sliding-window layers (window of 1,024 tokens) for every 1 full-attention layer, keeping most of the savings while preserving a global view in a fraction of the layers.

---

## Part 7 — Sparse attention: choose which tokens are worth looking at

A sliding window picks tokens by a fixed rule: "the most recent ones". But the tokens that matter most are not always the latest. **Sparse attention** lets each token attend only to a *selected subset* of earlier tokens, chosen more intelligently.

Early versions used fixed patterns (a few local tokens plus a few global ones), as in models like Longformer and BigBird. DeepSeek's recent approach, **DeepSeek Sparse Attention (DSA)**, introduced in DeepSeek-V3.2, makes the selection *learned*:

```text
1. A cheap "lightning indexer" scores how relevant each earlier token is to the current one.
2. Only the top-k highest-scoring tokens are kept (k = 2,048 in V3.2).
3. Full attention then runs only over those selected tokens.
```

So the expensive attention step covers roughly 2,048 tokens no matter how long the text is, while the cheap indexer decides which 2,048. DSA is used together with MLA, so the cache is compressed *and* the attention is selective. DeepSeek's newer V4 models go further, with a hybrid of compressed and sparse attention for a context length of one million tokens; the team reports it needs only about a quarter of the per-token compute and a tenth of the cache of V3.2 at that length.

---

## Part 8 — Linear attention and hybrids: replace the cache with a running summary

Everything so far still keeps (some form of) the K and V of earlier tokens, so memory grows with length. A more radical idea is to give attention a **fixed-size memory** instead. These designs are called **linear attention** (or recurrent/state-space designs). Instead of storing every earlier token and comparing against all of them, the layer keeps a single fixed-size *state*, a running summary, and updates it as each new token arrives.

```text
Full attention:    memory grows with every token (KV cache)
Linear attention:  memory stays the same size however long the text gets
```

The compute cost grows only in proportion to length, and the memory doesn't grow at all. The trade-off is that a fixed-size summary is lossy: recalling a *specific* earlier detail precisely is harder than looking it up directly in a full cache.

Because of that, the practical design is a **hybrid**. Qwen's recent models (Qwen3-Next and the Qwen3.5 family) use a layer called **Gated DeltaNet**, a gated linear-attention layer, for most layers and keep a full-attention layer (a "gated attention" layer) every fourth layer:

```text
Layer pattern, repeated:    [Gated DeltaNet] [Gated DeltaNet] [Gated DeltaNet] [full attention]
                              cheap, fixed memory ×3                  exact recall ×1
```

Three cheap layers do most of the work with constant memory, and one full-attention layer in four preserves the precise long-range lookup. Other model families have explored similar hybrids, with different linear layers (state-space designs such as Mamba among them).

---

## Part 9 — FlashAttention: not a new variant, a faster way to compute the same thing

It would be a mistake to leave out **FlashAttention**, because almost every modern system uses it, for both training and serving. But it's different in kind from the rest of this chapter. It doesn't change what attention computes. It changes **how it's computed on the hardware.** The numbers that come out are exactly the same as standard attention.

### The real bottleneck is memory traffic, not arithmetic

A GPU has two kinds of memory that matter here:

```text
HBM  (the GPU's main memory)    large (tens of GB)       slower
SRAM (on-chip, next to the      tiny  (a few MB to        very fast
      compute units)                  tens of MB in total)
```

You can think of SRAM as the countertop in a kitchen and HBM as the warehouse down the road. Cooking is fast, but every trip to the warehouse is slow.

A straightforward implementation of attention makes a lot of trips. Following the formula `softmax(Q·Kᵀ / √dₖ) · V` literally:

```text
1. compute the score grid  S = Q·Kᵀ          →  write the whole N × N grid to HBM
2. read S back, apply the mask and softmax   →  write the whole N × N grid P to HBM
3. read P back, multiply by V                →  write the output
```

The score grid is the problem. For a 8,192-token prompt, one head in one layer has 8,192 × 8,192 ≈ 67 million scores, which is about 134 MB in 16-bit numbers. With 32 heads, that's roughly 4.3 GB for a *single layer*, written out and read back several times. The GPU spends more time hauling this grid to and from memory than doing the multiplications. In other words, attention is usually **memory-bound**, the same kind of bottleneck we met in decoding (Chapter 12) and quantization (Chapter 14), just for a different reason.

### The idea: bring a tray, do all the steps, never store the grid

FlashAttention rearranges the work so the big grid is **never written out at all**. It cuts Q, K and V into small blocks (**tiles**) that fit in SRAM, and for each tile of queries it loops over the tiles of keys and values, doing the scores, the softmax and the multiplication by V entirely on-chip, and only the final output goes back to HBM. In kitchen terms: instead of walking to the warehouse after every step, you carry a tray of ingredients to the counter, cook the whole dish, and send back only the finished plate.

### The obstacle: softmax seems to need the whole row

There is a catch. Softmax turns a row of scores into percentages, and to do that it must divide by the sum over the **whole row**. If we only see one tile of keys at a time, how can we get the right answer?

The trick is **online softmax**: keep a *running maximum* and a *running sum* as tiles arrive, and whenever a new tile reveals a bigger maximum, rescale what you have accumulated so far. Here is a tiny example, with one query whose scores against four keys are `[1, 3, 5, 2]`, and whose values are `[10, 20, 30, 40]`. We process two keys at a time.

```text
Tile 1: scores [1, 3]
   running max      m = 3
   running sum      l = e^(1−3) + e^(3−3)          = 0.135 + 1        = 1.135
   running output   o = 0.135·10 + 1·20            = 21.35

Tile 2: scores [5, 2]      (the max is now 5, so rescale the old state by e^(3−5) = 0.135)
   running max      m = 5
   running sum      l = 1.135·0.135 + e^(5−5) + e^(2−5)   = 0.154 + 1 + 0.050   = 1.203
   running output   o = 21.35·0.135 + 1·30 + 0.050·40     = 2.89 + 30 + 1.99    = 34.88

Final answer:  o / l = 34.88 / 1.203 = 28.98
```

Computing it the ordinary way, with the full row at once, gives softmax weights `[0.015, 0.112, 0.831, 0.041]` and `0.015·10 + 0.112·20 + 0.831·30 + 0.041·40 = 28.98`. **The same answer**, but we never needed the whole row at once. That is the mathematical heart of FlashAttention. (For the backward pass during training, it also avoids storing the grid: it recomputes the scores tile by tile. That is extra arithmetic, but far less memory traffic, so it is a net win.)

### What you get, and what you don't

- **Exact results.** No approximation: the output is the same as standard attention (up to normal floating-point rounding).
- **Far less extra memory.** The memory for the score grid drops from growing with N² to growing only with N, which is a large part of what makes 100,000-token contexts practical.
- **Much faster in practice.** Because it cuts the trips to HBM, it typically runs several times faster than the straightforward version. Later versions (FlashAttention-2 and -3) improve how the work is divided up and use newer GPU features.
- **It works alongside everything else in this chapter.** It supports the causal mask (it can skip tiles that lie entirely in the future), and it combines with GQA, MLA, sliding windows and the rest. Frameworks such as PyTorch, vLLM and llama.cpp use FlashAttention-style kernels.

And what it does **not** do, which is easy to miss. It does not reduce the **arithmetic**: the number of score calculations still grows with the square of the text length (Cost 1, at the top of this chapter). And it does not shrink the **KV cache** (Cost 2); that is what MQA, GQA and MLA are for. FlashAttention attacks a third problem, the *traffic* of moving the score grid through memory, which is why it is a different kind of fix from everything else in the chapter.

---

## The complete picture

The variants change different things, so they're not competitors in one list. They act on different axes, and real models combine them:

| Variant | What it changes | Saves | Trade-off | Seen in |
|---|---|---|---|---|
| **MHA** | Baseline: K/V for every head | (reference) | Biggest cache | GPT-2 |
| **MQA** | All heads share 1 K/V | Cache (÷ number of heads) | Some quality loss | PaLM |
| **GQA** | Heads share K/V in groups | Cache (e.g. ÷4) | Small quality cost | Llama 3, Qwen3, Gemma |
| **MLA** | Cache a compressed latent | Cache (large) | More complex | DeepSeek-V3, Kimi K2 |
| **Sliding window** | Each token sees only recent tokens | Compute and cache | No direct long-range lookup | Mistral, Gemma 3 (mixed with global) |
| **Sparse (DSA)** | Each token sees a chosen subset | Compute | Needs an indexer | DeepSeek-V3.2 onward |
| **Linear / hybrid** | Fixed-size running state | Memory and compute | Lossy recall; keep some full layers | Qwen3-Next, Qwen3.5 |
| **FlashAttention** | How the computation runs on hardware | Time and memory, **exact** | None on accuracy | Nearly everything |

A way to organize them:

```text
Which tokens can each token see?     →  full, causal, sliding window, sparse
How are K and V stored and shared?   →  MHA, MQA, GQA, MLA
Is it still softmax attention?       →  yes (all above)   /   no (linear, state-space hybrids)
How is it computed on hardware?      →  FlashAttention and similar
```

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "GQA and MQA make attention compute much cheaper." | Their main saving is the **KV cache memory** (and the memory traffic during generation). Each query head still computes its own scores. |
| "MLA is just GQA with a different name." | GQA shrinks the cache by *sharing* K/V between heads. MLA keeps per-head K/V but *stores a compressed version* and expands it when needed. |
| "With a sliding window, the model can't use anything older than the window." | Stacked layers let information travel farther than one window, and many models add occasional global layers on top. |
| "FlashAttention is an approximation that trades accuracy for speed." | It is exact: the same result as standard attention, computed in a hardware-friendlier order. |
| "FlashAttention shrinks the KV cache." | It doesn't. It avoids writing the N × N score grid to memory. The KV cache is shrunk by MQA, GQA and MLA. |
| "FlashAttention removes the quadratic cost of attention." | The number of score calculations still grows with the square of the length. It removes the *memory traffic* and score-grid storage, not the arithmetic. |
| "Linear attention replaces full attention outright." | Its fixed-size memory is lossy, so today's practical designs are hybrids that keep some full-attention layers. |
| "There's one best attention variant." | They attack different costs, and real models combine several (for example MLA with sparse attention, or GQA with sliding-window layers). |

---

## Quick reference

```text
Two costs:     compute ∝ (length)²     and     KV cache memory ∝ length

KV cache per token = 2 × layers × KV heads × head dim × bytes

MQA   all query heads share 1 K/V              cache ÷ H
GQA   query heads share K/V in G groups        cache ÷ (H/G)
MLA   cache a small latent, expand on demand   cache ≈ tiny (DeepSeek-V3: ~576 numbers/token/layer)
SWA   attend to last w tokens only             compute and cache bounded by w
Sparse  attend to a learned top-k subset       compute bounded by k
Linear/hybrid  fixed-size running state        memory constant (+ a few full layers for recall)
FlashAttention  exact, tiled, online softmax; the N×N score grid never goes to GPU memory
                same math, far less memory traffic; does NOT cut arithmetic or the KV cache
```

---

## What's next

That completes the tour of how attention has been reworked. The next change goes deeper inside the block. Chapter 18 covers **Mixture of Experts**, where the FFN is replaced by many expert FFNs and only a few of them are used for each token. Chapter 19 then extends the model to images and audio, Chapter 20 covers reasoning models, and Chapter 21 puts everything together by looking inside Qwen and DeepSeek, which combine several of the attention variants from this chapter with Mixture of Experts.
