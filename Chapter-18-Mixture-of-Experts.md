# Chapter 18 — Mixture of Experts: A Huge Model That Only Uses Part of Itself

## Introduction

In Chapters 16 and 17 we saw how modern models swapped out parts of the plain GPT: new position encodings, new norms, new attention. All of those changed *how* something is computed, but the block kept the same shape. This chapter covers a change that goes deeper. It alters the **structure** of the Transformer block itself, and it answers a question that every scaling effort runs into:

> **Can we make a model enormous, without making every token pay for all of it?**

In a normal ("dense") model, making the model smarter by making it bigger has a built-in cost. Every parameter you add is a parameter that **every token** has to be multiplied by, so the compute per token grows in step with the model. That puts a ceiling on how big a model we can afford to train and run.

**Mixture of Experts (MoE)** breaks that link. The idea is simple to state:

```text
                 Token
                   │
                 Router
              ┌────┼────┐
              ▼    ▼    ▼
          Expert  Expert  Expert        (many experts; only a few are used per token)
              └────┬────┘
                Output
```

Instead of one big FFN that every token uses, the block has **many FFNs ("experts") and a small router** that picks a few of them for each token. The model can then hold a huge number of parameters in total, but each token only uses a small slice of them.

This chapter explains what changes inside the block, how the router works and how it is trained, how many parameters such a model really has, and why the idea is harder to run than it sounds. As always, we'll build it from pieces you already know: the FFN (Chapters 6 and 16), softmax (Chapter 4), the training loss (Chapter 9), and backpropagation (Chapter 10).

---

## Part 1 — The idea: don't make every token pay for everything

### Dense models: everyone takes every course

In a dense Transformer, every token goes through every block, and in every block through the **same** FFN and the same attention:

```text
Token
  ↓
Block 1   ← every token uses all of it
  ↓
Block 2   ← every token uses all of it
  ↓
Block 3   ← every token uses all of it
  ↓
 ...
```

So if you want a smarter model by adding parameters, every token pays for the extra parameters, whether or not they are relevant to that token. The compute per token grows with the model:

```text
Model grows  →  every token uses more parameters  →  every token costs more compute
```

A university analogy helps. In a **dense** university, every student takes every course: maths, physics, programming, history, languages. A programming student pays for all five, including the ones they don't need.

### MoE: visit an advisor first

In an **MoE** university, every student first visits an **advisor** (the router), who sends them only to the relevant specialists:

```text
Student → Advisor (router) → the relevant specialists only

   a programming student →  Programming expert + Maths expert
   a language student    →  Language expert + History expert
```

The university can now have **100 specialist departments**, while each student takes just two or three courses. Adding a new department makes the university more capable overall, but doesn't increase what any individual student has to pay.

That is the MoE trade, in the language of the model:

```text
Model grows  →  add more experts  →  total capacity increases
                                  →  each token still uses only a few experts
```

### "But the blocks already specialize. Why a new design?"

A natural objection. Earlier we said different layers learn different things: early blocks pick up basics, later ones more abstract relationships. Why not simply rely on that, instead of inventing something new?

First, a correction to that earlier intuition. It was a *useful picture*, not a guarantee: Block 7 does not become "the maths block". More importantly, even if Block 7 *did* learn features that help with maths, **every token still pays for Block 7**, because in a dense model every token goes through every block. Specialization by training doesn't give you **sparse computation**.

MoE gives you both things at once:

> **Different parts of the model can specialize, *and* the model can switch on only the parts that are useful for each token.**

---

## Part 2 — What actually changes inside the block

Recall the Transformer block from Chapter 7:

```text
Input
  ↓
Attention  →  Add & Norm
  ↓
FFN        →  Add & Norm
  ↓
Output
```

In an MoE block, **only the FFN is replaced.** The single FFN becomes a router plus many expert FFNs:

```text
                Transformer block
                       │
                 ┌─────┴─────┐
                 ▼           ▼
             Attention      MoE layer
                 │            │
                 │          Router
                 │            │
                 │      ┌─────┼─────┐
                 │      ▼     ▼     ▼
                 │     E1    E2    E3 ... E64       (each is a complete FFN)
                 │      │     │
                 │      └──┬──┘     (only the chosen experts run)
                 │      Combine
                 │         │
                 └────┬────┘
                      ▼
                   Output
```

Each **expert is a complete FFN**: the same `W1 → activation → W2` of Chapter 6, or the SwiGLU version of Chapter 16, just smaller or one among many. So MoE genuinely **changes the internal structure** of the block. It is not a runtime trick.

But nearly everything else is untouched:

```text
Unchanged:   tokenizer, embeddings, attention, Q/K/V, softmax attention,
             residual connections, normalization, the LM head

Changed:     Attention → ONE FFN        becomes        Attention → Router → SELECTED FFNs
```

It helps to compare with the other techniques we've met. Some change *how a model is run* or *stored*, and one changes the architecture:

| Technique | Does it change the architecture? |
|---|---|
| KV cache | No |
| Quantization | No |
| LoRA | No (the base architecture is unchanged) |
| Reasoning models | Usually no |
| **Mixture of Experts** | **Yes** |

---

## Part 3 — The router

### Where it sits and what it does

The router sits **exactly where the FFN used to be**, right after attention. For each token, it answers one question:

> **Which expert(s) should process this token?**

**The router is a single weight matrix. Nothing more exotic than that.** It is a small learned linear layer, the same kind of thing as every other weight matrix in the model.

**Its shape.** One matrix `W_r` with `d` rows (the model's hidden width) and one column per expert:

```text
W_r :  d × N        (N = number of experts)
scores = x · W_r    →   N numbers, one per expert
```

Given the token's current hidden vector `x`, it produces one score per expert. Each column of `W_r` is one expert's **"preference vector"**. The score for an expert is the dot product of the token's vector `x` with that column, the same "how well do these two vectors line up?" operation as the Q·K comparison in attention (Chapter 4). A high score means "this token looks like what this expert likes".

### A worked example

Suppose there are 4 experts, and for one token the router produces these scores. Softmax (Chapter 4) turns them into probabilities:

```text
Expert        1       2       3       4
score        0.2     3.8     1.1     0.4
softmax     0.024   0.887   0.060   0.030         (these add up to 1)
```

The router thinks Expert 2 is by far the best match. With **top-2 routing**, we keep the two highest and ignore the rest:

```text
Selected:  Expert 2 (0.887)   and   Expert 3 (0.060)
```

Often the two kept weights are re-normalized so they add up to 1 (here `0.937` and `0.063`). The token is then sent to just those two experts, and their outputs are **blended using the router's weights**:

```text
y  =  0.937 · Expert2(x)  +  0.063 · Expert3(x)
```

With made-up expert outputs, `Expert2(x) = [1, 2]` and `Expert3(x) = [3, 0]`:

```text
y = 0.937·[1, 2] + 0.063·[3, 0] = [0.937 + 0.189,  1.874 + 0]  =  [1.126, 1.874]
```

Experts 1 and 4 never run for this token. That is the saving.

(Real models differ in details: some keep the softmax weights as they are rather than re-normalizing them, some pick only one expert, and some pick eight. The structure is always the same: score, select a few, blend their outputs.)

### Different tokens choose differently, at every layer

The routing decision is made **per token and per MoE layer**. Different tokens in the same sentence can pick different experts:

```text
Token A  →  Experts 3 + 27
Token B  →  Experts 8 + 12
Token C  →  Experts 3 + 41
Token D  →  Experts 17 + 22
```

And the *same* token can take a different path at each layer, since each layer has its own router and its own set of experts:

```text
Layer 1  →  Experts 3 + 7
Layer 2  →  Experts 1 + 4
Layer 3  →  Experts 6 + 8
```

So the model's huge parameter space is **conditionally activated**: a token only touches the part of it that its router chose.

### One router per MoE layer (not one per matrix)

It's easy to get confused here, because the FFN contains *two* matrices (`W1` and `W2`, or three in SwiGLU). A block still has **one FFN sublayer**, and that FFN is made of two or three linear layers inside it. So in MoE:

```text
One MoE layer  =  ONE router  +  many experts (each expert is a whole FFN: W1, W2, ...)
```

The router sits *outside* the experts. It does not route to "the first linear layer" or "the second linear layer". It routes to whole experts. If a model has 32 blocks and every one is an MoE layer, then it has **32 routers**, one per block, each with its own set of experts.

### Does every block have to be MoE?

No. Many designs mix dense and MoE blocks. Some alternate them, and others, like DeepSeek-V3 (Chapter 21), keep the first few blocks dense and make the rest MoE. It's a design choice with a trade-off:

```text
More MoE layers   → more capacity and specialization,  more routing overhead
Fewer MoE layers  → simpler and cheaper,               less sparse capacity
```

---

## Part 4 — Counting parameters: total, active, and compute

### You're right: the model gets bigger

A fair worry: *"An expert is just an FFN. If the block now has several FFNs instead of one, doesn't the model get bigger, not smaller?"*

Yes. **The total number of parameters goes up.** The trick is that we don't *use* all of them for every token. Take a made-up example. Start with a dense model:

```text
Dense model:
   Attention             2B
   FFN                   1B
   ----------------------------
   Total                 3B       and every token uses all 3B
```

Now turn the FFN into 8 experts of 1B each, with a small router:

```text
MoE model (8 experts, top-2):
   Attention             2B
   Expert 1 … Expert 8   8 × 1B = 8B
   Router                tiny
   ----------------------------
   Total                ~10B

   Used per token:   attention 2B  +  2 chosen experts (2 × 1B)  =  ~4B
```

So the model grew from 3B to about 10B in **total**, but each token only does about 4B worth of work. That is why MoE is called **sparse**.

### But is that actually better? Choosing the fair comparison

Look carefully at those numbers, because they can mislead. Per token, this MoE model does about **4B** of work, while the dense model we started from did only **3B**. So the MoE model costs *more* per token than that 3B model. How can that be an improvement?

The answer is that a 3B dense model is **not the right thing to compare against**. A 3B model and a 10B-total MoE model hold very different amounts of knowledge. MoE is not "cheaper than every dense model". It is better than the *right* dense baselines. There are two fair ones:

| Model | Total parameters (what it stores) | Active per token (what it computes) |
|---|---:|---:|
| Dense, 3B | 3B | 3B |
| **MoE, 8 experts, top-2** | **~10B** | **~4B** |
| Dense, 10B | 10B | 10B |

```text
Baseline 1: a dense model with the same COMPUTE per token  (about 4B)
   MoE:   ~4B active, ~10B stored
   Dense: ~4B active, ~4B stored
   → same cost per token, but the MoE stores about 2.5× more, so it has far more
     capacity and, in practice, noticeably better quality at that compute cost.

Baseline 2: a dense model with the same TOTAL size  (10B)
   MoE:   ~4B active, ~10B stored
   Dense: ~10B active, ~10B stored
   → the MoE does about 2.5× less work per token. Its quality usually lands between
     the dense 4B and dense 10B models, because a parameter in an expert is a little
     less effective than a parameter in a dense layer that every token trains.
```

So the claim of MoE is not "fewer active parameters than a smaller dense model". It is **better quality for a given amount of compute**, or equivalently **the same quality for less compute**. The real-world example, worked out below, is Mixtral 8x7B: it uses about 13B active parameters per token, and its authors report that it matches or beats the dense Llama 2 70B on their benchmarks, at roughly a fifth of the per-token compute.

And the price, as Part 7 explains, is **memory and complexity**: all ~10B parameters still have to be stored, plus the routing, the communication between GPUs and the load balancing. MoE saves *compute*, not memory.

### Three different numbers

It is better not to say "the effective size is the same" or "the effective size is bigger". Separate three things:

| | Question it answers |
|---|---|
| **Total parameters** | How much does the model *store*? (Memory and knowledge capacity) |
| **Active parameters** | How many are actually used for one token? |
| **Compute** | How much arithmetic does one token cost? (Roughly proportional to the active parameters) |

This is why model specifications often read like "671B total, 37B active". Those numbers are not contradictory. They describe different things.

### A real example, worked out: Mixtral 8x7B

**Mixtral 8x7B** has 8 experts per layer, and the router picks 2 for each token. Its published size is about 47B total and 13B active. Here is where those numbers come from. The model has 32 layers, hidden size `d = 4096`, an FFN width of 14,336, 8 K/V heads of size 128 (so K and V are `4096 × 1024`), and a vocabulary of 32,000:

```text
One expert (a SwiGLU FFN: gate, up, down)   3 × 4096 × 14,336            ≈ 176 million
All 8 experts in one layer                   8 × 176M                     ≈ 1.41 billion
Attention in one layer (Q, O, K, V)          2×4096² + 2×4096×1024        ≈  42 million
Embedding table + LM head                    2 × 32,000 × 4096            ≈ 262 million

TOTAL    32 × (1.41B + 0.042B)  +  0.26B                                 ≈ 46.7 billion
ACTIVE   32 × (2 × 0.176B + 0.042B)  +  0.26B                            ≈ 12.9 billion
```

That matches the published figures. It also explains why the model is *not* "8 × 7B = 56B": only the FFNs are replicated eight times. The attention and the embeddings exist **once**.

---

## Part 5 — Training the router

### Who decides which expert is good?

**Nobody tells the router.** We never say "Expert 1 does mathematics, Expert 2 does coding". The router is trained exactly like everything else, and this is the key idea:

> **The router is itself a learned neural layer, and its weights receive gradients like any other.**

Training is the loop from Chapters 9 to 11, with the experts and the router inside it:

```text
Token → ... → Router → selected experts → combine → ... → LM head
                                                            │
                                                     cross-entropy loss
                                                            │
                                                     backpropagation
                                              ┌─────────────┴─────────────┐
                                              ▼                           ▼
                                       Expert weights               Router weights
                                              └─────────────┬─────────────┘
                                                          AdamW
```

At the start, the router knows nothing and routes more or less at random. Suppose the training example is a line of code, `def calculate_total(...)`, and the router sends it to Expert 3, which produces a poor prediction, so the loss is high. Backpropagation flows through the experts and *through the router's scores for the experts that were chosen*, nudging the router toward sending this kind of token somewhere that lowers the loss, say Expert 7. Over billions of tokens, routing patterns emerge.

### Specialization emerges, but messily

As a result, experts often end up **handling different kinds of tokens**: perhaps code-like tokens tend to go to one expert, certain language patterns to another. But resist the neat picture of `Expert 1 = Maths, Expert 2 = Python, Expert 3 = French`. Real experts tend to respond to messy combinations of syntax, language, token type and local context, and the pattern can change from layer to layer. The roles are **never assigned. They emerge from optimization.**

### The problem: expert collapse

Left alone, there's a trap. Suppose Expert 7 happens to be slightly better early on. The router sends it more tokens, so it gets more training, so it gets even better, so it gets even more tokens:

```text
Expert 7   ████████████████████
Expert 1   ██
Expert 2   █
Expert 3   █          ← the rest are barely used
```

This is called **expert collapse** or **routing imbalance**. It wastes most of the model's capacity, and on real hardware it is a performance disaster too (Part 7), because the busy expert's GPU is overloaded while the others sit idle.

### The fix: a load-balancing loss

MoE training adds a second goal to the loss from Chapter 9:

```text
Total loss  =  language-model loss   +   load-balancing loss
                "predict the next       "don't overload a few experts"
                 token correctly"
```

One common form of the balancing term looks at two numbers for each expert `i` over a batch: `f_i`, the fraction of tokens actually sent to it, and `P_i`, the average router probability it received. The penalty is `N × Σ f_i × P_i` (with `N` experts), which is smallest when work is spread evenly. With 4 experts:

```text
Perfectly balanced:  f = [0.25, 0.25, 0.25, 0.25],   P = [0.25, 0.25, 0.25, 0.25]
                     penalty = 4 × (4 × 0.25 × 0.25)                          = 1.00

Imbalanced:          f = [0.75, 0.125, 0.125, 0],    P = [0.6, 0.2, 0.1, 0.1]
                     penalty = 4 × (0.75·0.6 + 0.125·0.2 + 0.125·0.1 + 0)    = 1.95
```

The imbalanced router is penalized almost twice as heavily, so gradient descent pushes it toward spreading its tokens out. (This is the form used by Switch Transformer. Models differ in the details.)

There is a known side effect: this extra term pulls on the model's main objective. Newer designs try to balance load **without** a loss term. DeepSeek-V3, for example, adds a small adjustable bias to each expert's routing score *only when choosing experts*, nudging it up for under-used experts and down for over-used ones (Chapter 21).

### Capacity limits

Some systems also give each expert a fixed **capacity** per batch. If too many tokens choose the same expert, the overflow tokens may be **dropped** from that layer (they simply skip it and carry on through the residual connection) or re-routed. It's another way imbalance shows up as a real cost.

---

## Part 6 — Why the FFN, and not attention?

Two questions come up together here, and both reveal something about MoE.

**Why replace the FFN rather than the attention?**

Because **the FFN is where most of a block's parameters live**. (Recall from Chapter 6 that the FFN expands the vector to roughly four times its width and back. In Mixtral's numbers above, the FFN is about 176M per expert against about 42M for the whole attention in a layer.) Swapping one FFN for 64 experts multiplies the model's capacity enormously while each token still runs only a few of them. It is also relatively easy to make sparse: each token's FFN computation is independent of every other token's.

Attention is different. Its whole job is for **each token to relate to the other tokens** in the context. Making it sparse by routing tokens to different "attention experts" runs into trouble with communication, keeping the KV cache consistent, preserving global relationships between tokens, and GPU efficiency. So the common design is:

> **Keep attention dense. Make the expensive, parameter-heavy FFN sparse.**

**Wouldn't an MoE attention reduce the KV cache?**

It's a sharp idea, but no, not automatically. Recall what the KV cache is (Chapter 12): for every token, the Keys and Values at every layer, so its size is set by

```text
tokens  ×  layers  ×  number of KV heads  ×  head dimension
```

It is not determined by how many "experts" you have. If you routed tokens through several attention experts, each with its own K and V, you could actually make the cache **larger**. Making a token look at less could save compute, but that is not the same as shrinking the cache, and it might reduce the information the model builds about that token.

If the goal is a smaller KV cache, there are techniques built for exactly that: **MQA, GQA and MLA** (Chapter 17). The right way to see it is that each technique attacks a *different* bottleneck:

| Problem | Technique |
|---|---|
| Recomputing previous tokens | KV cache (Chapter 12) |
| Too much weight memory | Quantization (Chapter 14) |
| KV cache too large | MQA / GQA / MLA (Chapter 17) |
| Too many active parameters per token | **MoE** (this chapter) |

---

## Part 7 — Running an MoE model: what really happens to one token

On paper, MoE saves compute. In practice, it creates new engineering problems, because real models are spread across many GPUs (Chapter 23).

### One token, step by step

Take a model with 64 experts and top-2 routing:

```text
1. The token's hidden vector goes to the router → 64 scores
2. The top 2 are chosen          e.g. Expert 3 (0.71) and Expert 27 (0.18)
3. The token is sent to those two experts; each runs its FFN
4. The two outputs are blended with the router weights
5. The result continues to the next block (which has its own router)
```

### The hardware problem: experts on different GPUs

The 64 experts are probably spread across GPUs:

```text
GPU 1 → Experts 1–8        GPU 2 → Experts 9–16       ...       GPU 8 → Experts 57–64
```

A token that picks Expert 3 and Expert 27 has to be **sent to two different GPUs**, processed there, and its results brought back. This is **expert parallelism** (Chapter 23), and the tokens travel between GPUs in an "all-to-all" exchange at every MoE layer.

### Why batching matters even more

Sending a *single* token across GPUs costs more in communication than the computation is worth. But with **10,000 tokens** at once, the router can group them by expert:

```text
Expert 1  ←  1,000 tokens
Expert 2  ←    700 tokens
Expert 3  ←  1,200 tokens
...
```

Now each GPU receives a large batch for its experts, which is the efficient way to use it. That is one reason MoE serving systems rely on sophisticated scheduling (Chapter 22).

It also explains why **load balance matters physically**. If 90% of tokens go to Expert 3, that GPU is overloaded while the others idle, and the whole step waits for the slowest GPU.

### The catch: MoE saves compute, not memory

Here's the point people most often miss. **All the experts have to be stored in GPU memory, even though only a few run for each token.** An MoE model with 47B parameters needs memory for 47B, even though each token only computes with 13B of them.

And during generation (decode), which is memory-bound (Chapter 12), the picture gets subtle. For a single token, only the chosen experts need to be read. But once you serve a *batch*, different tokens choose different experts, so most of the experts get touched in each step:

```text
Mixtral-style (8 experts, top-2), tokens in the batch (if routing is roughly even):
   1 token   → about 2 of 8 experts read
   4 tokens  → about 5.5 of 8
   8 tokens  → about 7.2 of 8

DeepSeek-V3-style (256 routed experts, top-8):
   1 token   → 8 experts      32 tokens → about 163      128 tokens → about 252
```

(These are idealized numbers assuming routing is spread evenly. Real routing is lumpier.) So at serving batch sizes, an MoE model reads **almost all of its weights every step**. Its advantage is mostly in **compute per token**, not in memory traffic. That is why quantization (Chapter 14) and fast memory matter so much for MoE models.

### The full set of trade-offs

```text
Gains:   far more total capacity at about the same compute per token
Costs:   routing complexity, communication between GPUs, load balancing,
         memory for ALL the experts, and implementation difficulty
```

---

## Part 8 — MoE in real models

| Model | What it did | Numbers |
|---|---|---|
| **Switch Transformer** (2021) | Simplified routing to **one expert per token** (top-1); showed MoE could scale to trillions of parameters and train faster than a dense model of equal compute | Reported up to 7× faster pre-training than the dense T5 baseline; trained in bfloat16 |
| **Mixtral 8x7B** (2023) | A well-known open MoE: **8 experts per layer, 2 chosen per token** | ≈ 47B total, 13B active |
| **DeepSeekMoE** (2024) | **Fine-grained experts** (many small ones, so combinations are more flexible) plus **shared experts** that always run, to hold knowledge every token needs | Chapter 21 |
| **Qwen3 MoE** (2025) | 128 experts with 8 active per token, no shared expert | e.g. 235B total, 22B active (Chapter 21) |
| **DeepSeek-V3** (2024) | 1 shared + 256 routed experts, 8 routed active per token, first 3 blocks dense, no auxiliary balancing loss | 671B total, 37B active (Chapter 21) |

The trend is visible: more and smaller experts, a shared expert for common knowledge, and cleverer ways to keep load balanced.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "MoE is a runtime trick that doesn't change the model." | It changes the architecture: the FFN of the block becomes a router plus many expert FFNs. |
| "Each expert is a specialist in one subject." | Specialization emerges during training and is messy. Roles are never assigned, and they vary by layer. |
| "The router follows hand-written rules." | It's a small learned linear layer, trained by backpropagation along with the experts. |
| "An MoE block has one router per linear layer inside the FFN." | One router per MoE layer, which sends the token to whole experts (each expert is a full FFN). |
| "An MoE model is smaller than a dense model of the same quality." | It usually has *more* total parameters. It just uses a small fraction of them for each token. |
| "An MoE model does less work per token than any dense model." | Only compared with a dense model of the same *total* size. Compared with a smaller dense model it may do slightly more work, but it stores far more knowledge. The fair comparison is quality per unit of compute. |
| "671B total, 37B active means the numbers disagree." | Total is what is stored. Active is what one token uses. They measure different things. |
| "Mixtral 8x7B has 8 × 7B = 56B parameters." | Only the FFNs are replicated. Attention and embeddings exist once, giving about 47B. |
| "MoE saves memory." | It saves compute. All experts must be kept in memory, and at serving batch sizes most are read every step. |
| "Every token goes to the same experts at every layer." | Each layer has its own router, so a token can take a different path at each layer. |
| "Making attention MoE would shrink the KV cache." | The cache is set by tokens × layers × KV heads × head size. Shrinking it is the job of MQA, GQA and MLA. |
| "Balanced routing happens by itself." | Left alone, routers collapse onto a few favourites. Training adds a load-balancing incentive (or a bias trick). |
| "Every block of an MoE model is an MoE block." | Not necessarily. Many models mix dense and MoE blocks (DeepSeek-V3 keeps its first three dense). |

---

## Quick reference

```text
MoE block:    Attention (dense)  →  Router  →  top-k of N expert FFNs  →  weighted blend
              (the single FFN of Chapter 6 becomes N experts + a router)

Router:       scores = x · W_r  →  softmax  →  keep top-k (often re-normalized)
              y = Σ (weight_i × Expert_i(x))      one router per MoE layer; per token, per layer
              learned by backprop; sits where the FFN used to be

Counting:     total parameters = stored;  active = used by one token;  compute ≈ active
              Mixtral 8x7B: 32 layers, 8 experts, top-2  →  ≈ 46.7B total, ≈ 12.9B active
              (only FFNs replicated; attention + embeddings once)

Training:     loss = LM loss + load-balancing loss      (penalty N·Σ f_i·P_i, min = 1 when balanced)
              or aux-loss-free: adjustable per-expert bias used only for selection (DeepSeek-V3)
              failure mode: expert collapse;   capacity limits can drop overflow tokens

Why FFN:      most parameters live there; easy to make sparse; keep attention dense
Serving:      experts spread over GPUs (expert parallelism, Chapter 23); tokens travel all-to-all;
              needs big batches; MoE saves COMPUTE, not MEMORY (all experts stored,
              most read each step at serving batch sizes)

Real models:  Switch (top-1), Mixtral (8, top-2), DeepSeekMoE (fine-grained + shared),
              Qwen3 (128, 8 active), DeepSeek-V3 (1 shared + 256 routed, 8 active; 671B / 37B)
```

---

## What's next

We've now seen how the *inside* of a block can change, and how a model can grow far beyond what it computes for any one token. The next change is about what the model can *take in*. So far every input has been text. Chapter 19 covers **multimodal models**: how an image or a sound is turned into vectors that a language model can read alongside words, and how the two spaces are taught to understand each other. You'll revisit the encoders and cross-attention from Chapter 15 in a practical setting.
