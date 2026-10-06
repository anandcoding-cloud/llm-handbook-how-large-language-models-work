# Chapter 5 — Multi-Head Attention: Many Attention Patterns at Once

## Introduction

Chapter 4 built one complete attention computation, end to end: one set of `W_Q`, `W_K`, `W_V`, one pass of compare-scale-mask-softmax-blend, one new representation per word. That's genuinely the hard part. This chapter takes that fully-understood mechanism and simply **runs it several times in parallel**, each copy free to notice a different kind of relationship. If Chapter 4 clicked, this chapter will feel like a natural, almost easy extension — so we'll move a little faster, but without skipping the "why."

---

## Part 1 — Why isn't one attention head enough?

Go back to our running sentence:

```text
"The cat drank the milk because it was thirsty."
```

A single attention head has to answer one comparison question at a time — whatever question its `W_Q` and `W_K` happen to be shaped to ask. But a sentence like this actually needs several *different* questions answered simultaneously to be fully understood:

```text
"Who is the subject?"        →  cat
"What action happened?"      →  drank
"Why did it happen?"         →  because, thirsty
```

One head, with one fixed pair of "what to search for" and "what to be found by" rules, can't easily chase all of these at once — it's built to specialize in one kind of relationship, not several unrelated ones. So the natural fix is almost embarrassingly simple: **run several attention heads side by side**, each with its own separate rules, and let each one focus on whatever relationship it finds most useful.

```text
Sentence
   │
   ▼
Head 1 (say, grammar-ish relationships)
Head 2 (say, meaning-ish relationships)
Head 3 (say, pronoun-ish relationships)
Head 4 (say, cause-and-effect-ish relationships)
   │
   ▼
Combine everything
```

That's Multi-Head Attention: not one smarter attention mechanism, but several *ordinary* attention mechanisms — each exactly like the one you just learned in Chapter 4 — running in parallel and then merged.

*(A quick honest flag before we go further: the "Head 1 does grammar, Head 2 does pronouns" framing above is a teaching simplification, not a guarantee. We'll come back to exactly how literally to take it once we've seen a real numerical example.)*

---

## Part 2 — How do the heads actually end up different?

This is where everything from Chapter 4 pays off directly. Recall the central idea: `W_Q`, `W_K`, and `W_V` aren't values — they're *learned rules* that get applied to an embedding. Multi-head attention simply gives **each head its own separate rules to work with**:

```text
A single attention head uses one set of rules:

    W_Q, W_K, W_V

Multi-head attention instead gives every head its own
independent set, each starting from its own separate
random values — not copies of one another:

    Head 1:  W_Q₁, W_K₁, W_V₁
    Head 2:  W_Q₂, W_K₂, W_V₂
    Head 3:  W_Q₃, W_K₃, W_V₃
```

Every head starts from **exactly the same embeddings** — nothing about the input changes per head. What differs is purely which matrices get applied to those embeddings:

```text
                 SAME EMBEDDINGS
                        │
      ┌─────────────────┼─────────────────┐
      ▼                 ▼                 ▼
   Head 1            Head 2            Head 3
  W_Q₁ W_K₁ W_V₁    W_Q₂ W_K₂ W_V₂    W_Q₃ W_K₃ W_V₃
```

All of these matrices are randomly initialized independently, and all of them get reshaped by training, exactly as described in Chapter 4. Because each head's matrices start at different random values and get nudged by slightly different gradient signals over the course of training, they naturally *end up* specializing in different things — not because anyone assigned them a job, but because having heads specialize in different, complementary relationships is a better strategy for reducing prediction error than having every head converge on the same thing. Nobody tells one head "you're the grammar head." That division of labor **emerges** purely from training.

---

## Part 3 — How many heads, and who decides?

This raises an obvious engineering question: how many heads should a model have?

**Short answer: the number of heads is chosen by the model's designer, before training even begins. It is not learned.** Think of it like staffing an investigation team: you, the architect, decide whether to hire 2 investigators, 8, or 32 — the investigators themselves don't get a vote. In a Transformer, the engineer decides things like:

```text
Embedding size = 768,   Attention heads = 12
Embedding size = 4096,  Attention heads = 32
```

What **is** learned is the content of every head's own `W_Q`, `W_K`, `W_V` — never how many heads exist in the first place.

### Why not just use one huge head instead?

You could — a single head is free to use the *entire* embedding dimension. But there's a real tradeoff, well captured by a medical analogy: one generalist doctor looking at everything, versus a heart doctor, a brain doctor, an eye doctor, and a skin doctor, each looking closely at one thing and then combining opinions. Multiple focused specialists often catch more than one generalist spreading their attention thin across everything at once.

Concretely, the embedding dimension gets **split** evenly across the heads:

```text
Embedding size 768, split across 12 heads
        ↓
    64  64  64  64  64  64  64  64  64  64  64  64
   (each head works on its own 64-dimensional "view")
```

### Why not use hundreds of heads, then?

Every additional head costs real compute: more parameters, more GPU memory, slower inference, higher training cost — and past a certain point, extra heads stop adding meaningfully new information. That's diminishing returns, and it's why real models settle on a specific number rather than maximizing it. There's no single formula for the "correct" number — researchers arrive at it experimentally — but real models cluster around a similar pattern:

| Model | Embedding size | Heads | Dimension per head |
|---|---:|---:|---:|
| GPT-2 Small | 768 | 12 | 64 |
| GPT-2 Medium | 1024 | 16 | 64 |
| Llama 2 7B | 4096 | 32 | 128 |
| GPT-3 175B | 12288 | 96 | 128 |

Notice the per-head dimension tends to land somewhere around 64–128 across very different model sizes — not a strict law, but a common, empirically-successful design pattern.

### An honest correction to the "grammar head / pronoun head" story

It's a genuinely useful *first* intuition to say "one head learns grammar, another learns pronouns, another learns cause and effect" — but don't take it too literally. In real trained models:

- Some heads do specialize cleanly, in ways researchers can identify and name.
- Some heads stay fairly generic, contributing a little to many things.
- Some heads turn out almost redundant.
- Some heads are so unimportant that researchers have removed them from trained models with barely any drop in quality.

A more accurate mental model:

```text
Multi-Head Attention
        ↓
Many independent attention mechanisms
        ↓
Each learns whatever turns out to be
most useful for reducing prediction error —
no more, no less.
```

---

## Part 4 — A worked example: three real heads, one sentence

Let's watch this play out concretely. We're back at our sentence, updating the word `"it"`:

```text
"The animal didn't cross the road because it was tired."
```

Suppose, after training, three heads produce these attention weight distributions over the earlier words:

**Head 1**

| Word | Weight |
|---|---:|
| animal | 80% |
| road | 5% |
| because | 15% |

*Interpretation:* this head has become good at figuring out **who** "it" refers to — mostly `"animal"`. Nobody programmed this; it emerged because attending this way reliably reduced prediction error during training.

**Head 2**

| Word | Weight |
|---|---:|
| animal | 10% |
| road | 5% |
| because | 85% |

*Interpretation:* this head discovered something different — it's picked up on `"because"` as a signal that this sentence expresses causality. Head 2 isn't wrong or worse than Head 1; it's solving a different problem entirely.

**Head 3**

| Word | Weight |
|---|---:|
| animal | 20% |
| road | 70% |
| because | 10% |

*Interpretation:* this one is less obviously interpretable. Maybe it's picked up on something about physical locations or objects. Or — and this genuinely happens in real models — maybe it's just a fairly useless head that training didn't manage to specialize well. Researchers really have found attention heads like this, contributing very little.

Each head runs the *entire* attention pipeline from Chapter 4 independently — its own Q, K, V, its own scores, its own softmax, its own weighted sum of V — and produces its own output vector. Suppose:

```text
Head 1 output → [7, 2]
Head 2 output → [1, 8]
Head 3 output → [5, 6]
```

Three heads, three completely independent opinions about how to update the word `"it"`.

---

## Part 5 — Combining the heads: concatenation

What do we do with three separate output vectors? The answer surprises a lot of people the first time they see it: **we don't average them, and we don't add them — we glue them end-to-end.** This operation is called **concatenation**:

```text
Head 1  [7, 2]
Head 2  [1, 8]     →  Concatenate  →  [7, 2, 1, 8, 5, 6]
Head 3  [5, 6]
```

### Why concatenate instead of average?

Imagine three specialists each write you a report: a doctor says *"heart is healthy,"* a mechanic says *"engine needs oil,"* a lawyer says *"paperwork expires next month."* Would you average those reports into one blended opinion? Of course not — each one contains genuinely unique, non-overlapping information. Averaging would destroy exactly the specialization we worked so hard to get. Concatenation keeps every head's finding fully intact, side by side, for later layers to make sense of.

### The dimension math works out on purpose

Concatenating grows the vector's size: 3 heads of 2 numbers each becomes 6 numbers. In a real model, this scales up the exact same way — say, 32 heads each producing 128 numbers:

```text
128 (per head) × 32 (heads) = 4096
```

And 4096 happens to be exactly the model's original embedding size. **That is not a coincidence.** Model designers deliberately choose the number of heads and the per-head dimension so that:

```text
Embedding size = Number of heads × Dimension per head
```

This isn't an accident of nature — it's a design constraint, chosen precisely so the concatenated output lines back up with the size the rest of the model expects.

---

## Part 6 — The last learned matrix: W_O

There's one problem left. After concatenation we have `[7, 2, 1, 8, 5, 6]` — but what does that number actually *mean*? It's not one coherent representation; it's three completely independent opinions, stapled together with no interaction between them.

Picture three engineers independently reviewing your code: one scores performance `9/10`, one scores security `8/10`, one scores readability `6/10`. Someone hands you `[9, 8, 6]`. What now? Eventually, *someone* has to combine these separate opinions into one coherent judgment. That's exactly the job of the final learned matrix in attention, called `W_O` — the **output projection**.

```text
Concatenated heads
        │
        ▼
    Multiply by W_O
        │
        ▼
Final Multi-Head Attention Output
```

Think of `W_O` as an **opinion merger**. Before it runs, each head's findings sit in complete isolation — without `W_O`, the heads would never actually interact; they'd remain forever-separate streams of information bolted together. `W_O` is what lets the model mix information *across* heads: "when Head 17 says X and Head 4 says Y, that combination actually means Z."

### A tiny worked example

Suppose our concatenated vector is `X = [7, 2, 1, 8, 5, 6]`, and (for illustration) `W_O` is a very simple matrix:

```text
        col→0  col→1
W_O  =  [ 1     0 ]
        [ 0     1 ]
        [ 1     0 ]
        [ 0     1 ]
        [ 1     0 ]
        [ 0     1 ]
```

Multiplying `X` by `W_O` (dot product of `X` with each column):

```text
Output₀ = 7×1 + 2×0 + 1×1 + 8×0 + 5×1 + 6×0 = 7+1+5 = 13
Output₁ = 7×0 + 2×1 + 1×0 + 8×1 + 5×0 + 6×1 = 2+8+6 = 16

Final output = [13, 16]
```

In this toy example, `W_O` just happened to add certain positions together — but only because we deliberately picked an extremely simple matrix to make the arithmetic obvious. A real `W_O` is a dense grid of thousands of trained numbers, mixing every head's output into every output position in far more intricate ways — but the operation is identical: dot-product-with-each-column, exactly like `W_Q`, `W_K`, and `W_V`.

Notice also the shape: the input to `W_O` has 6 numbers (3 heads × 2 each), and the output has 2 — back to the original embedding size. `W_O` isn't just mixing information, it's also **compressing the concatenated output back down** to the size the rest of the Transformer block expects to work with.

---

## The complete picture

```text
Embeddings (same input to every head)
        │
   ┌────┼────┬─────────┐
   ▼    ▼    ▼         ▼
 Head1 Head2 Head3 ... HeadN
   │    │    │         │     (each: its own W_Q, W_K, W_V,
   │    │    │         │      its own full attention pipeline
   ▼    ▼    ▼         ▼      from Chapter 4)
[out1][out2][out3]...[outN]
        │
        ▼
   Concatenate
        │
        ▼
   Multiply by W_O
        │
        ▼
Final Multi-Head Attention Output
```

Take a moment to notice what this entire chapter — and Chapter 4 before it — has actually accomplished:

```text
Sentence → Tokenization → Embeddings → Q,K,V → Attention →
Softmax → Weighted V → Multiple Heads → Concatenate → W_O
```

We're still only doing **one thing**: making each word's representation richer by letting it gather relevant information from every other word. We haven't predicted anything, reasoned about anything, or generated any text — we've simply improved *what each token knows*. That distinction matters, because it sets up exactly where this handbook goes next.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "Each head is literally assigned a job like 'grammar' or 'pronouns.'" | That's a useful first intuition, not a guarantee. Real heads range from clearly specialized, to generic, to nearly redundant — some get pruned from trained models with little effect. |
| "More heads always means a better model." | Every extra head costs compute and memory, with diminishing returns past a certain point — it's a deliberate design tradeoff, not "more is always better." |
| "Multi-head attention is a fundamentally different, more complex mechanism than single-head attention." | It's the exact same mechanism from Chapter 4, run several times in parallel with separate learned matrices, then merged. Nothing new is invented. |
| "The head outputs get averaged together." | They're concatenated (joined end-to-end), specifically to preserve each head's unique findings rather than blending them into mush. |
| "W_O is just for resizing the vector back down." | Resizing is a side effect. Its real job is mixing information *across* heads — without it, heads would never interact with each other at all. |
| "Attention alone lets the model 'think' about the sentence." | Attention (single or multi-head) only lets words *gather* information from each other. Genuine per-token processing of that gathered information is a separate step — coming up next. |

---

## Quick reference

```text
Multi-Head Attention:

1. Every head gets the SAME embeddings as input.
2. Every head has its OWN learned W_Q, W_K, W_V.
3. Each head runs the full Chapter-4 attention pipeline independently.
4. Head outputs are CONCATENATED (joined), never averaged.
5. The concatenated vector is multiplied by W_O (learned) to:
     - mix information across heads
     - project back down to the original embedding size

Design choice (fixed by architect): number of heads, head dimension.
Learned by training: the content of every W_Q, W_K, W_V, and W_O.

Embedding size = Number of heads × Dimension per head   (by design)
```

---

## The architecture so far

The diagram so far. The double-lined box is this chapter's addition: the single attention head is now many heads in parallel, plus `W_O`.

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
                          ▼
╔══════════════════════════════════════════════════╗
║ MULTI-HEAD ATTENTION                       ◄ NEW ║
║  Head 1: own W_Q, W_K, W_V → attention           ║
║  Head 2: own W_Q, W_K, W_V → attention           ║
║     ...                                          ║
║  Head H: own W_Q, W_K, W_V → attention           ║
║  each head: softmax(Q·Kᵀ ÷ √dₖ + mask) · V       ║
║  concatenate the heads                           ║
║  × W_O  (mix the heads, back to size d)          ║
╚══════════════════════════════════════════════════╝
                          │
                          ▼
            Richer context-aware vectors
```

Learned so far: embedding and position tables; W_Q, W_K, W_V for every head; W_O.

---

## What's next

We've now built the full attention mechanism — the part of a Transformer responsible for letting every word gather relevant information from every other word. But gathering information isn't the same as *thinking* about it. Chapter 6 covers the **Feed-Forward Network** — the part of the architecture that finally lets each token actually process what it just gathered. Chapter 7 then assembles Attention and the FFN together, using residual connections and Layer Normalization, into one complete **Transformer Block**.
