# Chapter 23 — Speed and Scale: Speculative Decoding, Parallelism and Hardware

## Introduction

Chapter 22 showed how to package a model, run it, and serve it to many users. Two stubborn limits remain, and they are what this chapter is about:

1. **Generation produces one token per forward pass.** However cleverly requests are batched, each *individual* answer still grows one token at a time, and every one of those steps has to read all the model's weights (Chapter 12). How can a single request go faster?
2. **A big model doesn't fit on one GPU.** A 70-billion-parameter model in 16-bit numbers is about 140 GB, far more than the 80 GB of a high-end GPU. How do you run a model that is bigger than the machine?

The chapter has three parts:

| Part | Question | Answer |
|---|---|---|
| **Speculative decoding** | How can one request generate faster? | A small model drafts several tokens, and the big model checks them all in one pass |
| **Parallelism** | How can a model (and its data) be spread over many GPUs? | Split the matrices, split the layers, or split the data |
| **Hardware** | Why are GPUs used, and why are new chips being built for LLMs? | Because LLM inference is limited by memory movement as much as by arithmetic |

As in Chapters 14 and 22, **nothing here changes what the model computes.** Speculative decoding gives exactly the same output distribution as ordinary decoding. Parallelism splits the same arithmetic across machines. The Transformer is untouched.

---

## Part 1 — Speculative decoding: a small model drafts, a big model verifies

### The problem

A large model is slow at generating one token at a time. Each step is a full forward pass:

```text
Large model  →  token 1
Large model  →  token 2
Large model  →  token 3
Large model  →  token 4
```

And remember why each step is slow. In the decode phase (Chapter 12) the GPU spends its time *reading the weights from memory*, and does very little arithmetic with them. The big model is mostly waiting for data.

### The key observation

If a step is limited by reading the weights rather than by arithmetic, then **checking several tokens in one pass costs almost the same as producing one**. The weights are read once either way. This is the same fact that made batching nearly free in Chapter 22, used differently: instead of serving several *users* per weight-read, serve several *positions of one user*.

There is also a second fact, from Chapter 9. When we trained the model, a single forward pass over a sentence produced a next-token prediction at **every position at once**, thanks to the causal mask. The model can do the same at inference. If we hand it a prompt plus some *guessed* extra tokens, one forward pass tells us what the model would have predicted after each of them.

So here is the plan:

> Get *cheap guesses* for the next several tokens from somewhere, then ask the big model to check all of them in a single pass.

### The idea

Use a **small, fast "draft" model** to guess several tokens ahead:

```text
Draft model (small, fast):
   "I think the next tokens are:   the → capital → of → France → is → London"
```

Then ask the large model to **verify** them in one forward pass, giving it the prompt followed by all the drafted tokens. Because of the causal mask, that single pass produces the large model's own prediction after every position:

```text
Large model input:   The capital of France is  London
Large model's view:  ✓ "capital"   ✓ "of"   ✓ "France"   ✓ "is"   ✗ "London" (it wanted "Paris")
```

The first four drafted tokens match what the large model would have said, so they are accepted. The fifth is wrong, so it is rejected, and the large model supplies the correct token instead. In one large-model pass we advanced **five tokens** (four accepted plus the corrected one) instead of one:

```text
WITHOUT speculation:    Large → Large → Large → Large → Large          (5 passes)

WITH speculation:       Draft (cheap, several steps)
                              ↓
                        Large: verify all, in ONE pass                  (1 pass)
                              ↓
                        accept the good prefix, fix the first mistake
```

In one line:

> **Small model = guess. Large model = judge.**

The small model never *replaces* the large one. And it is an optimization in a very specific sense: **the model weights themselves don't change, and nothing is retrained.**

### But is the output the same? Why this is exactly correct

An important worry: surely the answer depends on whose guesses we accept? It doesn't. The acceptance rule is designed so that the **final text follows exactly the large model's probability distribution**, as if the small model had never been involved. The idea (from the original papers) is a form of rejection sampling.

For each drafted token, with `q` the draft model's probability for it and `p` the large model's probability for it:

```text
Accept the drafted token with probability   min(1, p / q)

If it is rejected, sample the replacement from the "leftover" distribution:
   (what the large model wanted more than the draft did), re-normalized.
```

A worked example. After the prompt `The capital of France is`, suppose the next-token probabilities are:

| Token | Draft model `q` | Large model `p` |
|---|---:|---:|
| Paris | 0.30 | 0.90 |
| London | 0.50 | 0.05 |
| Rome | 0.20 | 0.05 |

The draft model happens to pick **London**. The large model thinks London is unlikely (`p = 0.05` against `q = 0.50`), so the acceptance probability is `min(1, 0.05 / 0.50) = 0.10`. Most of the time it is rejected. Then the replacement is drawn from what the large model wants *more* than the draft did: `p − q` is positive only for Paris (`0.90 − 0.30 = 0.60`), so the leftover distribution is simply Paris.

Now add up every way the process can end:

```text
End with Paris:   draft says Paris (0.30) → always accepted                       = 0.30
                  draft says London (0.50) → rejected (0.90) → replaced by Paris  = 0.45
                  draft says Rome   (0.20) → rejected (0.75) → replaced by Paris  = 0.15
                                                                           total  = 0.90  ✓
End with London:  draft says London (0.50) × accepted (0.10)                      = 0.05  ✓
End with Rome:    draft says Rome   (0.20) × accepted (0.25)                      = 0.05  ✓
```

The result is **Paris 0.90, London 0.05, Rome 0.05**: exactly the large model's own distribution, `p`. The draft model only affects *how fast* we get there, never *what* comes out. (With greedy decoding, temperature 0, the rule simplifies to "accept a token if it is the large model's top choice".)

### How much faster?

Two numbers decide the speed-up: the **acceptance rate** `α` (how often a drafted token is accepted) and the **cost ratio** `c` (how expensive one draft step is compared with one large-model step). If the draft proposes `K` tokens per round, the expected number of tokens produced per large-model pass is:

```text
expected tokens per pass  =  (1 − α^(K+1)) / (1 − α)
```

and the overall speed-up is that number divided by the cost of the round, `1 + K·c`.

| Acceptance rate α | Tokens per large pass (K = 4) | Speed-up if the draft is 20× cheaper (c = 0.05) |
|---:|---:|---:|
| 0.5 | 1.9 | ≈ 1.6× |
| 0.8 | 3.4 | ≈ 2.8× |
| 0.9 | 4.1 | ≈ 3.4× |

So the speed-up depends heavily on **how often the draft guesses right**. The original paper reported 2× to 3× speed-ups on a large model with no change to its outputs. Text that is predictable (code, boilerplate, quoted passages, formulaic answers) has a high acceptance rate, while open-ended creative text has a lower one.

### Where the draft comes from: variants

A separate small model is only one way to get the guesses:

- **A smaller model from the same family**, such as a 1B model drafting for a 70B one. It must use the same tokenizer and vocabulary so the tokens line up.
- **Extra prediction heads on the model itself** (methods such as Medusa and EAGLE), so no second model is needed. A related idea is the **multi-token prediction** trained into models like DeepSeek-V3, whose extra prediction can later be reused to speed up generation (Chapter 21).
- **No model at all**: simply look for text in the prompt that the answer is likely to repeat (an "n-gram" or prompt-lookup draft). It works well when summarizing, editing, or answering about a document.

### When it helps less

- **When the server is already busy.** Chapter 22's batching already uses the GPU's spare capacity. If a large batch makes the GPU compute-bound, there is less free capacity for the extra verification work.
- **When acceptance is low**, since wrong guesses are wasted work.
- It also needs memory for the draft model (or heads) and its cache.

It is best thought of as another example of the pattern that runs through this handbook: **don't make the expensive model do work that a cheaper source can propose first.**

---

## Part 2 — Parallelism: running a model bigger than one GPU

### The problem

Suppose a model needs 80 GB for its weights and your GPU has 24 GB. It obviously doesn't fit. Even with Chapter 14's quantization, the biggest models don't fit on one device, and the KV cache (Chapter 12) needs room as well. So we have to **split the work across several GPUs**. There are several ways to split, and each answers a different question:

```text
Tensor parallelism    →  split the MATRICES inside each layer
Pipeline parallelism  →  split the LAYERS into groups
Data parallelism      →  keep a full COPY of the model, split the DATA
Expert parallelism    →  split the EXPERTS of a Mixture-of-Experts model
```

### 1. Tensor parallelism: split each layer across GPUs

Take the basic operation `Y = X·W`. Instead of one GPU holding all of `W`, each GPU holds a *piece*, computes its share of the answer, and the pieces are combined. There are two ways to cut a matrix, and a Transformer block uses both.

**Cutting by columns.** Each GPU holds some of the *columns* of `W` and produces some of the *output numbers*. Combining is just placing them side by side:

```text
X = [1 2 3 4]        W (4×2) = [1 0]
                               [0 1]
                               [1 1]
                               [2 0]

GPU 1 holds column 1:   Y₁ = 1·1 + 2·0 + 3·1 + 4·2 = 12
GPU 2 holds column 2:   Y₂ = 1·0 + 2·1 + 3·1 + 4·0 = 5

Combine:   Y = [12  5]          (just concatenate, no arithmetic)
```

**Cutting by rows.** Each GPU holds some of the *rows* of `W` and receives the matching part of `X`. Each produces a *partial sum* of the same output, and the pieces have to be **added** together:

```text
GPU 1 holds rows 1–2 of W and X[1, 2]:   partial = [1·1 + 2·0,  1·0 + 2·1] = [1  2]
GPU 2 holds rows 3–4 of W and X[3, 4]:   partial = [3·1 + 4·2,  3·1 + 4·0] = [11 3]

Combine:   Y = [1 2] + [11 3] = [12  5]      (an "all-reduce": add the partial results)
```

Same answer, `[12 5]`, as computing it on one GPU.

**How a Transformer block uses these.** The block is designed so that the two cuts chain neatly:

- **The FFN** (Chapter 6): cut the first matrix `W1` by **columns**. Each GPU computes its own slice of the hidden units. The activation function (GELU or SiLU) works on each number independently, so *no communication is needed* between the two matrices. Then cut the second matrix `W2` by **rows**, so each GPU produces a partial sum, and a single all-reduce adds them up.
- **Attention** (Chapter 5): the heads are independent of each other, so give each GPU a **group of heads**. Each GPU computes its heads (and keeps the KV cache for just those heads). Then `W_O` is cut by rows, and one all-reduce adds the partial results.

So each block needs just **two all-reduces**, one after attention and one after the FFN.

The big advantage is that tensor parallelism **cuts the latency of each token**. In decode, every GPU reads only *its share* of the weights, so a step is faster. For a 70B model in 16-bit (140 GB): split across 2 GPUs, each reads 70 GB per step; at roughly 3.35 TB/s of memory bandwidth that's about 21 ms per step instead of about 42 ms if the whole thing had fit on one GPU (communication time comes on top).

The cost is that those all-reduces happen **twice per block, for every block, for every token**, so the GPUs must talk to each other constantly and quickly. Compare the links that connect them:

| Link | Speed (as NVIDIA quotes it for the H100, total both directions) |
|---|---|
| GPU memory (HBM) | about 3,350 GB/s |
| NVLink between GPUs in one server | about 900 GB/s |
| PCIe Gen 5 | about 128 GB/s |
| Network between servers | far lower still |

So tensor parallelism is normally used **inside one server**, across 2, 4 or 8 GPUs connected by a fast link like NVLink.

> **Mental model: split one layer across GPUs.**

### 2. Pipeline parallelism: split the layers into stages

Instead of cutting inside each layer, give each GPU a **consecutive group of whole Transformer blocks**:

```text
GPU 1:  Blocks  1 – 8     ──►   activations   ──►
GPU 2:  Blocks  9 – 16    ──►   activations   ──►
GPU 3:  Blocks 17 – 24    ──►   activations   ──►
GPU 4:  Blocks 25 – 32  + LM head
```

The data flows through the GPUs like items along an assembly line. The communication is light: only the activation vectors at the boundaries travel between stages, and only once per stage rather than twice per block. That makes pipeline parallelism suitable for **linking separate servers**, where the network is slower.

The catch is that, naively, only one GPU is busy at a time while the others wait. This is called the **pipeline bubble**. The fix is to cut the batch into **micro-batches** so every stage always has something to work on:

```text
NAIVE: one batch moves through four stages                time →
GPU 1   ██ ·· ·· ··
GPU 2   ·· ██ ·· ··
GPU 3   ·· ·· ██ ··
GPU 4   ·· ·· ·· ██          each GPU busy 1 slot in 4  (25%)

WITH 4 MICRO-BATCHES  (a, b, c, d)                        time →
GPU 1   a  b  c  d  ·  ·  ·
GPU 2   ·  a  b  c  d  ·  ·
GPU 3   ·  ·  a  b  c  d  ·
GPU 4   ·  ·  ·  a  b  c  d  busy 16 slots of 28  (57%)
```

With `m` micro-batches and `p` stages, the fraction of time the GPUs are busy is roughly `m / (m + p − 1)`: 25% for one batch, 57% for four, and about 84% for sixteen (with four stages). One more thing to remember: pipeline parallelism helps **throughput** and **memory**, but it does *not* shorten the time for one token, because that token still has to pass through every stage in turn.

> **Mental model: split the model vertically, by layers.**

### 3. Data parallelism: copy the model, split the data

Here every GPU holds a **full copy** of the model, and each handles a *different part of the workload*:

```text
GPU 1 → full model copy → batch 1
GPU 2 → full model copy → batch 2
GPU 3 → full model copy → batch 3
GPU 4 → full model copy → batch 4
```

Its two uses are different:

- **In training**, each GPU computes gradients on its own slice of a large mini-batch, and then the gradients are **averaged across GPUs** (Chapters 10 and 11: a mini-batch gradient is already an average) before every copy applies the same update. This is how training scales to thousands of GPUs. Because the optimizer state (the momentum and variance from Chapter 11) is large, training systems often also **shard** it, and the weights, across the data-parallel GPUs instead of duplicating it.
- **In inference**, data parallelism is simply **more replicas**: two copies of the model serve twice as many users. It does nothing to make a too-big model fit.

> **Mental model: copy the model, split the data.**

### 4. Expert parallelism: split the experts

In a Mixture-of-Experts model (Chapter 18), the FFN of each block is replaced by many separate expert FFNs. **Expert parallelism** places different experts on different GPUs:

```text
GPU 1 → Experts 1, 2
GPU 2 → Experts 3, 4
GPU 3 → Experts 5, 6
```

A router decides which experts each token uses, and the token's vector is sent to the GPU that holds them. Chapter 18 covers this in full.

### 5. One more: splitting the sequence

For extremely long contexts, even the attention itself can be spread out. **Context (or sequence) parallelism** splits the *tokens* of a long sequence across GPUs, so that no GPU has to hold the whole sequence's activations or attention at once.

### Putting them together

Real deployments combine these. A typical large-model recipe:

```text
Within one server (fast NVLink):        tensor parallelism across 8 GPUs
Across servers (slower network):        pipeline parallelism between groups of servers
To serve more users, or to train:       data parallelism, i.e. several copies of all of the above
For MoE models:                         expert parallelism as well
```

| Method | What gets split | Communication | Typical use |
|---|---|---|---|
| Tensor parallel | The matrices inside each layer | Heavy (2 all-reduces per block per token) → needs fast links | Large-model inference and training, within a server |
| Pipeline parallel | Groups of layers | Light (activations between stages) | Very large models, across servers |
| Data parallel | The data / requests | Gradient averaging (training) or none (inference) | Training; more serving capacity |
| Expert parallel | MoE experts | Tokens routed to experts | MoE models |
| Context parallel | The sequence | Between attention chunks | Very long contexts |

A last reminder from earlier chapters: parallelism is the **last** resort, not the first. A 70B model that needs 140 GB in 16-bit needs only about 35 GB in 4-bit (Chapter 14), which fits on a single 80 GB GPU. Quantization often turns a multi-GPU problem into a one-GPU problem, and engines like vLLM (Chapter 22) support both.

---

## Part 3 — Hardware: why GPUs, and why new chips?

A natural question, and one that comes up as soon as you see how repetitive an LLM is: a Transformer is the same few operations over and over (matrix multiplications, attention, normalization, repeated across dozens of identical blocks). **Why run it on a general-purpose GPU? Why not build a dedicated "Transformer chip"?**

### What a GPU is, and why it fits

A GPU is a highly **parallel, programmable** machine: thousands of small compute units that can perform the same operation on huge amounts of data at once. That matches matrix multiplication perfectly. It is also *programmable*, so the same chip runs a new attention variant, a new activation function or a new quantization format tomorrow, just by running new software.

### The spectrum of chips

```text
CPU                       general purpose, few powerful cores, flexible, slowest at this
 ↓
GPU                       programmable, massively parallel
 ↓
AI accelerator / TPU      built around matrix multiplication, less general
 ↓
LLM-specialized chip      designed around LLM workloads in particular
```

**TPUs** (Google's Tensor Processing Units) are purpose-built AI accelerators, designed around large matrix units rather than as a general graphics-derived design. And specialization is accelerating: the newest generation of chips is explicitly aimed at **LLM inference** (see below).

### Why not hard-wire today's Transformer into silicon?

Because **Transformers keep changing**. A chip takes years to design and build. Today's model might use attention, an FFN and RMSNorm. A few years on, the same field has produced:

```text
MoE layers  +  different attention (MLA, sparse, linear hybrids)  +  new position encodings
+  new quantization formats  +  reasoning-specific workloads
```

(Chapters 16, 17, 18 and 21 are a record of exactly that churn.) A chip hard-wired for last year's recipe could become an expensive paperweight. So the trade-off is fundamental:

```text
GPU    →  programmability, at the cost of some efficiency
ASIC   →  efficiency, at the cost of flexibility
```

### The bigger issue: memory, not arithmetic

Here is the most important point of this section. You've seen it in Chapter 12 (decode is memory-bound), Chapter 14 (fewer bytes per weight means faster generation), and Chapter 17 (FlashAttention reduces memory traffic). For LLM inference, the question is often not *"can I multiply these matrices quickly?"* but:

> **"Can I move all these weights and KV-cache values to the compute units fast enough?"**

That is sometimes called the **memory wall**. It is why accelerators are now designed around **compute, memory and interconnect together**, rather than raw arithmetic alone. For scale, an H100 has about 80 GB of main memory (HBM) at about 3.35 TB/s, and the tiny on-chip SRAM beside the compute units is much faster still. Whatever can be kept close to the compute is cheap to read, and that is the principle behind both FlashAttention (Chapter 17) and the new chips.

### What the industry is doing (as announced in 2026)

These are recent announcements, so treat the details as a snapshot:

- **Google's eighth-generation TPUs**, announced in April 2026, were split into two chips: **TPU 8t** for training and **TPU 8i** for inference. Google describes the 8i as designed for low-latency inference, with **288 GB of high-bandwidth memory and 384 MB of on-chip SRAM** (three times the previous generation), so that large KV caches can sit close to the compute.
- **OpenAI and Broadcom** unveiled **Jalapeño** in June 2026, described as OpenAI's first chip designed from the ground up for LLM inference, built around the memory movement, kernels and serving patterns that matter for models like the ones it serves. Initial deployment was announced for late 2026.

Both are examples of the same idea: **an inference chip designed around the real bottlenecks of serving LLMs**, which are memory capacity and bandwidth, the KV cache, and the speed of communication between chips, rather than just peak arithmetic. The same logic explains why **reasoning models** (Chapter 20), which generate very long outputs and so keep large KV caches alive for longer, put even more pressure on memory.

A caution about the specifics: product details, names and availability change quickly, and the numbers above come from the companies' own announcements. The *principle* (programmability vs efficiency, and the memory wall) is the lasting part.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "Speculative decoding changes the model's answers." | The accept/reject rule makes the final distribution exactly the large model's. The draft only affects speed. |
| "The small model's guesses are trusted." | Every drafted token is checked by the large model. Wrong ones are rejected and replaced. |
| "It needs retraining or a modified model." | The standard form needs neither: any suitably small model with the same tokenizer can draft. |
| "Verifying 5 tokens costs 5× as much as generating 1." | Decode is memory-bound: the weights are read once, so verifying several tokens in one pass costs about the same as one. |
| "Speculative decoding always gives the same speed-up." | It depends on the acceptance rate (predictable text gains more) and on how busy the GPU already is. |
| "More GPUs always means faster." | Splitting adds communication. Tensor parallelism needs very fast links, and pipeline parallelism has bubbles. |
| "Tensor parallelism and pipeline parallelism are the same thing." | Tensor splits the matrices *inside* each layer. Pipeline gives each GPU a group of whole layers. |
| "Pipeline parallelism makes one token faster." | It helps memory and throughput, but a token still passes through every stage in turn. |
| "Data parallelism lets a bigger model fit." | Every GPU holds a full copy, so the model must already fit. It adds throughput (inference) or training speed. |
| "You need multiple GPUs for any big model." | Often quantization (Chapter 14) removes the need: 70B in 4-bit is about 35 GB. |
| "A GPU is used because it's the best possible Transformer chip." | It's used because it is fast *and programmable* while models keep changing. Specialized chips trade flexibility for efficiency. |
| "LLM speed is all about arithmetic." | Decode is largely limited by memory bandwidth, which is why chips are being designed around memory and interconnect. |

---

## Quick reference

```text
Speculative decoding   draft model guesses K tokens; large model verifies all in ONE pass
                       (causal mask → predictions at every position at once)
   accept drafted token with prob min(1, p/q); on rejection sample from normalized max(0, p−q)
   → output distribution is EXACTLY the large model's
   expected tokens per pass = (1 − α^(K+1)) / (1 − α);  speed-up ≈ that / (1 + K·c)
   variants: smaller model, extra heads (Medusa/EAGLE), multi-token prediction, n-gram lookup

Tensor parallel    split matrices: columns (concatenate) then rows (all-reduce add)
                   FFN: W1 by columns, W2 by rows;  attention: by heads, W_O by rows
                   2 all-reduces per block per token → needs NVLink, used within a server;
                   cuts per-token latency (each GPU reads a share of the weights)
Pipeline parallel  groups of layers per GPU; light communication; micro-batches fill the
                   bubble (busy ≈ m / (m + p − 1)); helps memory/throughput, not one token's latency
Data parallel      full copy per GPU; training: average gradients; inference: more replicas
Expert parallel    experts spread across GPUs; tokens routed to them (Chapter 18)
Context parallel   split a very long sequence across GPUs
Combine:  tensor inside a server + pipeline across servers + data replicas (+ experts)
Try quantization first: 70B → ~35 GB in 4-bit.

Hardware   GPU: parallel + programmable.  ASIC/TPU: efficient, less flexible.
           LLM inference is memory-bound → "memory wall" → chips designed around memory + interconnect
           (TPU 8i: 288 GB HBM + 384 MB SRAM; OpenAI/Broadcom Jalapeño; announced 2026)
```

---

## What's next

We have now covered the model itself and how it is served and scaled: built, trained, adapted, compressed, redesigned, packaged, served, sped up and spread across GPUs. What remains is the layer on top. Chapter 24 steps back from the model to the **systems built around it**: routing between models, calling tools, memory, retrieval and agents, and keeping all of it observable and affordable. After that, Appendix A returns to the very small to make everything concrete, following one complete forward and backward pass through a tiny GPT with real numbers.
