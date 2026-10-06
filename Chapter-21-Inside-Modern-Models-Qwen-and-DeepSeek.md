# Chapter 21 — Inside Modern Models: How Qwen and DeepSeek Combine These Ideas

## Introduction

When people say models like Qwen and DeepSeek use a "better" architecture or training recipe, what do they actually mean? This chapter answers that by looking inside two real model families, using their published technical reports and model cards. You'll see that neither is a new kind of network. Both are **decoder-only Transformers** (Chapter 15) built from the pieces you now know: GQA or MLA (Chapter 17), RoPE, RMSNorm and SwiGLU (Chapter 16). What makes them stand out is a set of choices about how attention is stored, how the FFN is organized, and how the model is trained. Because this chapter comes late in the handbook, every ingredient below should already be familiar, so it works as a capstone.

A caveat up front: this field moves quickly, and the details below reflect the published reports and model cards available at the time of writing. Treat the specific numbers as a snapshot, and the *ideas* as the lasting part.

---

## A quick reminder: Mixture of Experts

Both families rely on Mixture of Experts, covered in Chapter 18. Here is a short reminder of the idea.

In the FFN of Chapter 6, every token passes through the same big FFN. In a **Mixture-of-Experts (MoE)** layer, that single FFN is replaced by **many smaller FFNs, called experts**, plus a small **router** that picks a few of them for each token:

```text
                 Token
                   │
                 Router   (picks the best few experts for THIS token)
              ┌────┼────┐
              ▼    ▼    ▼
          Expert 3  Expert 27  ...   (only the chosen few run; the rest sit idle)
              └────┬────┘
                Combine
```

This gives a model with a huge number of **total** parameters but only a small number **active** for any one token. You'll see this written as, for example, "235B total, 22B active": the model stores 235 billion parameters of knowledge, but each token only pays the compute cost of about 22 billion. Attention stays dense; only the FFN becomes sparse. Left alone, a router tends to favor a handful of popular experts and neglect the rest, so MoE training adds an incentive to spread tokens across experts, as an extra term added to the loss from Chapter 9.

---

## Part 1 — Qwen

### Qwen3: the modern recipe, in dense and MoE forms

Qwen3 (released in 2025) comes in six **dense** sizes (0.6B up to 32B) and two **MoE** models. The dense models follow exactly the modern recipe from Chapter 16:

```text
Attention:      GQA  +  RoPE  +  QK-Norm  (the old bias on Q, K, V was removed)
FFN:            SwiGLU
Normalization:  RMSNorm, pre-norm
Embeddings:     tied in the smaller models
```

The two **MoE** models are Qwen3-30B-A3B (30B total, 3B active) and Qwen3-235B-A22B (235B total, 22B active). Their design:

- **128 experts per MoE layer, 8 active per token.**
- **No shared experts.** The earlier Qwen2.5-MoE had them; Qwen3 dropped them.
- A **global-batch load-balancing loss.** This is the balancing incentive mentioned in the primer, which keeps the router from sending everything to a few popular experts. Computing that balance across the whole batch, rather than in small pieces, leaves each expert more room to specialize.

### Qwen3's training recipe

Qwen3 was pre-trained on roughly **36 trillion tokens across 119 languages and dialects**, in three stages. Notice how the training data and the context length change as training proceeds:

```text
Stage 1  General:        over 30 trillion tokens, 4,096-token sequences
Stage 2  Reasoning:      about 5 trillion more tokens, with extra STEM and coding data
Stage 3  Long context:   hundreds of billions of tokens, 32,768-token sequences
                         (using the context-extension techniques from Chapter 16)
```

Training on short sequences first is cheaper (attention cost grows with length, Chapter 17). The model is then extended to longer contexts with a smaller amount of long-text data.

After pre-training, a **four-stage post-training** process shapes the model's behavior:

```text
1. Long chain-of-thought "cold start"   →  teach basic step-by-step reasoning from verified examples
2. Reasoning RL                         →  reinforcement learning (GRPO) on problems with checkable answers
3. Thinking-mode fusion                 →  merge "thinking" and "non-thinking" ability into ONE model
4. General RL                           →  broad improvement across many task types
```

The third stage is a distinctive idea: instead of shipping one model that always reasons at length and another that answers quickly, a single Qwen3 model can do either. A switch in the chat template (`/think` or `/no_think`) selects the mode. (Chapter 20 covers reasoning models and the RL methods behind them.)

For the **smaller** models, Qwen3 used **strong-to-weak distillation**: a small "student" model is trained to imitate a large "teacher", first on text the teacher generated, then by matching the teacher's output probabilities while generating its own text. The report says this took about a tenth of the GPU time that running reinforcement learning on each small model would.

### Newer Qwen: hybrid attention

Qwen's more recent models (Qwen3-Next, and the Qwen3.5 and 3.6 families) changed the attention itself. Following Chapter 17, they use a **hybrid** of cheap linear-attention layers (Gated DeltaNet) and occasional full-attention layers, in a 3:1 pattern. Qwen3-Next, for instance, is an 80B-total, 3B-active MoE model with 48 layers and 512 experts per MoE layer (10 active per token), a native context of 262,144 tokens, and a multi-token-prediction training objective, an idea explained in the DeepSeek section below. The Qwen3.5 family spans models from under a billion parameters up to a roughly 400B-total MoE.

---

## Part 2 — DeepSeek

### DeepSeek-V3: three architectural ideas

DeepSeek-V3 (December 2024) is a **671B-total, 37B-active** MoE model with 61 layers and a hidden size of 7,168. Three architecture choices define it.

**1. MLA for attention** (Chapter 17). 128 heads with a per-head dimension of 128, but the cache stores only a compressed latent of 512 numbers (plus the small RoPE piece) per token per layer.

**2. DeepSeekMoE for the FFN.** In every layer except the first three (which stay as ordinary dense FFNs), the FFN is replaced by an MoE layer with **1 shared expert and 256 routed experts, of which 8 routed experts are active per token**. Two ideas make this design distinctive:

- **Many small experts.** Rather than a few large experts, there are many fine-grained ones, so the router can combine small specialties flexibly.
- **A shared expert that always runs.** Some knowledge is needed by *every* token (basic language patterns, for example). Giving that to an always-on shared expert frees the routed experts to specialize more sharply, instead of each having to relearn the common material.

**3. Auxiliary-loss-free load balancing.** As the primer noted, routers need some way to avoid overloading a few experts. The usual approach adds an extra *auxiliary loss* to the training objective (a second term added to the loss from Chapter 9), but that extra term can pull the model away from simply predicting well. DeepSeek-V3 removes the auxiliary loss and instead keeps a **bias for each expert**:

```text
Routing:   each expert's score gets its bias added, and the top 8 are chosen
           (overloaded expert → its bias is pushed down;  underused expert → pushed up)

Weighting: how much each chosen expert contributes still uses the ORIGINAL score,
           without the bias
```

So the bias only nudges *which* experts get picked, to even out the load, and never distorts *how much* each one contributes to the answer.

### DeepSeek-V3's training choices

**Multi-token prediction (MTP).** In Chapter 9, each position predicted the one next token. With MTP, each position **also predicts one more token beyond that** (V3 uses a depth of 1: the next token and the one after). That gives the model a denser learning signal from every position and encourages it to plan slightly ahead. The extra prediction can also later be used to speed up generation (Chapter 23).

**FP8 mixed-precision training.** Normally, the numbers inside the model are stored and multiplied in 16-bit or 32-bit formats. DeepSeek-V3 does most of its large matrix multiplications in **8-bit floating point**, which cuts memory use and speeds up the math. Precision-sensitive steps stay in higher precision, and the numbers are scaled in small tiles or blocks so that 8-bit's limited range isn't a problem. (Chapter 14 explains these number formats, and why using fewer bits per number is such a powerful idea.)

**Efficient scale.** DeepSeek-V3 was pre-trained on **14.8 trillion tokens using about 2.79 million H800 GPU-hours**, notably less than the compute budgets commonly reported for models of comparable quality at the time, which is a large part of why it drew so much attention. Its context length was extended in two stages: from 4K to 32K, then to 128K, the same short-first-then-extend pattern Qwen3 uses.

### The later DeepSeek models

**DeepSeek-V3.2** made exactly one architectural change to its predecessor: it added **DeepSeek Sparse Attention** (Chapter 17) through continued training, so each token attends to only the 2,048 most relevant earlier tokens, selected by a lightweight "lightning indexer."

**DeepSeek-V4** (April 2026) comes in two sizes: V4-Flash (284B total, 13B active) and V4-Pro (1.6T total, 49B active), with a context length of **one million tokens**. According to its model card, V4 introduces three notable changes:

- **A hybrid of compressed and sparse attention** for long contexts, using about 27% of the per-token compute and 10% of the cache of V3.2 at that length.
- **Manifold-Constrained Hyper-Connections (mHC)**, a strengthened replacement for the plain residual connection of Chapter 7, intended to keep signals stable across layers.
- **The Muon optimizer** instead of AdamW (Chapter 11) for faster, more stable convergence. This is a reminder that even the optimizer, which seems like settled ground, is still being improved.

Training used more than 32 trillion tokens, with a mixed-precision scheme in which the MoE experts use an even smaller 4-bit format and most other parameters use FP8. Post-training had two steps: first, separate specialist models are trained for individual domains (with supervised fine-tuning and reinforcement learning using GRPO), then they are merged into one model through distillation.

---

## Part 3 — The two side by side

| | Classic GPT-2 | Qwen3 | DeepSeek-V3 |
|---|---|---|---|
| Type | Decoder-only | Decoder-only | Decoder-only |
| Attention | Multi-head (MHA) | GQA + QK-Norm | MLA (compressed cache) |
| Positions | Learned table | RoPE | RoPE (with decoupled part for MLA) |
| Normalization | LayerNorm | RMSNorm (pre-norm) | RMSNorm (pre-norm) |
| FFN | GELU, dense | SwiGLU; dense, or MoE with 128 experts (8 active) | SwiGLU; MoE with 1 shared + 256 routed experts (8 active), first 3 layers dense |
| Total / active parameters | Same (dense) | Up to 235B / 22B (MoE) | 671B / 37B |
| Pre-training data | Far smaller | ~36T tokens, 3 stages | 14.8T tokens |
| Training extras | — | Staged data; thinking-mode fusion; distillation for small models | MTP; FP8 training; auxiliary-loss-free balancing |
| Context handling | Fixed table | 4K then 32K (then longer variants) | 4K then 32K then 128K |

---

## Part 4 — So what does "better" actually mean?

Looking across both families, the improvements fall into a few themes:

```text
1. Cheaper to RUN
   GQA, MLA, sliding window, sparse and hybrid attention → a smaller KV cache and less
   compute per token, especially at long context (Chapter 17)

2. More capacity for the same compute
   Mixture of Experts → many parameters stored, few used per token

3. Cheaper and more stable to TRAIN
   FP8 or FP4 arithmetic, QK-Norm, auxiliary-loss-free balancing, better optimizers,
   training on short sequences first and extending later

4. A richer learning signal
   Multi-token prediction; carefully staged and filtered training data

5. Better post-training
   Reasoning-focused reinforcement learning, thinking modes, distillation from larger models
```

It's worth being honest about two things. First, "better" is relative to a goal: these choices mostly buy *efficiency* (the same quality for less cost) and *long-context ability*. Second, we can only describe what has been published. The designs of many closed models aren't public, so comparisons here are among open models. What you can rely on is the pattern: none of it replaces the core idea you learned in Chapters 2 to 11. Every one of these models still takes tokens, embeds them, passes them through stacked attention and FFN blocks with residuals and normalization, and predicts the next token. The progress is in the details of each box.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "Qwen and DeepSeek use a fundamentally different kind of neural network." | They're decoder-only Transformers built from the same blocks. The differences are in attention storage, FFN organization (MoE), and training methods. |
| "A 671B model uses all 671 billion parameters for every token." | In an MoE model, only a small number of experts are active per token (37B in DeepSeek-V3). All parameters must be stored, but few are used at once. |
| "MoE models are smaller and cheaper to store." | They need memory for *all* parameters; the saving is in compute per token, not in stored size. |
| "DeepSeek's trick is a secret new algorithm." | Its pieces (MLA, fine-grained experts, FP8 training, multi-token prediction) are all described in public papers. The achievement is combining them carefully and efficiently. |
| "Multi-token prediction changes how the model generates text." | It's mainly a *training* objective that gives a denser signal. It can also help speed up generation, but the model is still a next-token predictor. |
| "Newer is always better on every measure." | These changes trade off cost, complexity and quality in different ways, and real models combine only the ones that suit their goals. |

---

## Quick reference

```text
Qwen3 dense     GQA + RoPE + QK-Norm + SwiGLU + RMSNorm(pre-norm); 0.6B–32B
Qwen3 MoE       128 experts, 8 active, no shared expert; 30B-A3B and 235B-A22B
Qwen3 training  ~36T tokens: general (4K ctx) → reasoning (~5T) → long context (32K);
                post-training: CoT cold start → reasoning RL → thinking fusion → general RL
Qwen newer      hybrid Gated DeltaNet : full attention = 3 : 1, MoE, multi-token prediction

DeepSeek-V3     671B total / 37B active; MLA; 1 shared + 256 routed experts, 8 active;
                first 3 layers dense; aux-loss-free balancing (per-expert routing bias);
                multi-token prediction; FP8 training; 14.8T tokens; 4K→32K→128K context
DeepSeek-V3.2   adds DeepSeek Sparse Attention (top-2,048 tokens)
DeepSeek-V4     CSA+HCA attention, mHC residuals, Muon optimizer, FP4+FP8, 1M context
```

---

## What's next

That completes our tour of the model itself: how it is built, trained, adapted, compressed and redesigned. The remaining chapters are about everything *around* the model, and they apply to any model. Chapter 22 covers how a model is packaged and served (GGUF, llama.cpp and vLLM), and Chapter 23 covers how generation is made faster and spread across several GPUs. The last chapter, Chapter 24, steps back to the systems built around the model: routing between models, calling tools, memory, retrieval, agents, and keeping all of it observable and affordable.
