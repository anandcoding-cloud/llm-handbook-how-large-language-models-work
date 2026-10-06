# Chapter 4 — Attention: Queries, Keys, Values and the Causal Mask

## Introduction

If this chapter feels slower and more deliberate than the last three, that's on purpose. Attention is the single most important idea in this handbook, and it's also the idea most tutorials rush through or explain backwards — introducing the letters Q, K, and V before ever explaining why they need to exist. We're not going to do that. We're going to build up to them one problem at a time, and then spend real time on the part most explanations skate past entirely: the learned weight matrices that make the whole thing trainable in the first place. Those matrices are, quite literally, where the model's intelligence about *relationships between words* actually lives — so they deserve more than a passing mention.

---

## Part 1 — Why do we need attention at all?

### Step 1: What your brain already does

Before any math, notice you already perform attention constantly, without thinking about it. Read this sentence:

> **The cat drank the milk because it was thirsty.**

When you hit the word **"it,"** your brain automatically asks: *who is "it"?* It quietly scans backward over the earlier words:

```text
The       → No
cat       → Maybe
drank     → No
the       → No
milk      → Maybe
because   → No
```

and settles on:

```text
cat   → very likely
milk  → not very likely
```

You just performed attention. No math, no vectors, no Q/K/V — just deciding *which previous words deserve my attention right now.* That's the entire idea. Everything below is teaching a computer, which only understands numbers, how to do the same thing.

### Step 2: Words start out isolated

After tokenization and embedding (Chapters 2 and 3), suppose our sentence is:

```text
"The cat drank milk"
        ↓
"The"   → [0.2, 0.8]
"cat"   → [0.7, 0.1]
"drank" → [0.4, 0.5]
"milk"  → [0.8, 0.2]
```

Here's the problem: each of these vectors only knows about *itself*.

```text
The    → knows only "The"
cat    → knows only "cat"
drank  → knows only "drank"
milk   → knows only "milk"
```

The vector for **"cat"** has no idea **"milk"** even exists in the same sentence. They're isolated islands. But the meaning of `"drank"` clearly depends on knowing that a cat is doing the drinking, and milk is what's being drunk. Without some way for words to exchange information, the model can never combine them.

```text
Before attention:            After attention:

cat    (isolated)            cat  ────┐
drank  (isolated)      →     drank ←──┼── exchange information
milk   (isolated)            milk  ───┘
```

**Attention is simply a mechanism that lets words communicate with each other.** That's the whole motivation, before a single formula appears.

### Step 3: Should every word listen equally?

Now a second, sharper problem appears. Take a longer sentence:

```text
"The cat drank milk because it was thirsty."
```

When updating the word `"it"`, should it listen equally to *every* earlier word — `"The"`, `"cat"`, `"drank"`, `"milk"`, `"because"`? Obviously not. Some words are far more relevant to resolving `"it"` than others. So the real problem attention has to solve isn't just *"let words talk to each other,"* it's:

```text
Listen MORE to important words,
and listen LESS to irrelevant ones.
```

And how do we decide what counts as "important"? This is where we finally need a precise, computable definition of *relevance* — and the simplest tool for measuring how related two vectors are is a **dot product**, a similarity score between two vectors that's larger when they point in similar directions.

```text
Attention = finding which word-vectors are most relevant
            to the current word, where relevance is
            measured by similarity.
```

That's the concept. Suppose the model produces these raw relevance scores for `"it"`:

| Previous word | Score |
|---|---:|
| The | 0.1 |
| cat | 9.8 |
| drank | 1.2 |
| milk | 0.6 |

These aren't probabilities yet — just relative signals of relevance. Notice something important, though: **we don't replace `"it"` with `"cat"`.** Instead, we build a brand-new representation of `"it"` that *incorporates* information from the words it found relevant:

```text
Old representation of "it"
        ↓
     Attention
        ↓
New representation = "it" + information gathered from "cat", etc.
```

> **This is probably the single most important sentence in this chapter: attention enriches a word's representation. It doesn't replace it.**

---

## Part 2 — Why can't we just compare the raw embeddings?

Here's a completely reasonable next question: if attention is just "compare vectors for similarity," why not compare the embeddings we already have, directly?

The problem: if I tell you *"compare every word with every other word,"* you immediately have to ask — **compare what, exactly?** The whole embedding? Part of it? Something else? And once you decide a word is relevant, what should get copied over — the whole embedding again, unchanged?

This is precisely the problem that **Q, K, and V** exist to solve: they give the model a way to define *what* to compare, separately from *what information to pass along* once a match is found.

---

## Part 3 — Introducing Q, K, and V

Here's the intuition, using a library. Imagine searching for a book. You don't compare the entire contents of every book against every other book. Each book instead has three separate things:

```text
Book
 ↓
Title            (how you search for it)
 ↓
Shelf number     (where it's matched/found)
 ↓
Contents         (the actual information you wanted)
```

You search using the **title**, not the full contents. Once you find the right book, *then* you read the contents. Attention does exactly this. Every word creates three different versions of its own embedding:

```text
Embedding
   ↓
Q = "What am I looking for?"       (Query)
K = "What can I be matched on?"    (Key)
V = "What information do I carry?" (Value)
```

Think of a teacher: sometimes they're asking a question, sometimes they're the one being asked, sometimes they're handing over the actual material. Same person, three different roles depending on the moment. Q, K, and V are the same idea, applied to a word — and when processing `"it"`, the model creates `Q(it)` and compares it against the **Keys** of every earlier word:

```text
                Q(it)
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
   K(cat)      K(milk)     K(drank)
```

Whichever Key scores highest against `Q(it)` "wins" the most attention — and then the model reads that word's **Value**, not its Key. **You match using K. You copy from V.**

---

## Part 4 — Where do Q, K, and V actually come from?

We now know *what job* Q, K, and V do. The natural next question is: how does a word actually get one of each? Are they just three more random numbers assigned per token, separately trained, sitting alongside the embedding?

### They can't just be random — they have to come from the embedding

Imagine, for a moment, that Q, K, and V were completely independent of the embedding — say, three more lookup tables, keyed by token ID, filled with their own numbers that have nothing to do with the embedding table from Chapter 3.

That can't be right, and it's worth being precise about *why*. The entire point of attention is to compare words based on **meaning** — the exact meaning that Chapter 3 spent a whole chapter carefully building into the embedding through training. If Q, K, and V were disconnected from that embedding, attention would end up comparing something arbitrary, with no guaranteed relationship to what the word actually means. Whatever `"cat"` ends up looking like as a Query, a Key, and a Value, all three **must be built out of `"cat"`'s embedding** — otherwise there would be nothing tying the comparison back to meaning at all.

So: Q, K, and V are not separately stored, and they are not independently random. They are **three different representations, derived from the one embedding we already have** — three different lenses applied to the same vector.

```text
Embedding("cat")
        │
   ┌────┼────┐
   ▼    ▼    ▼
   Q    K    V
```

### But "derived from the embedding" raises a harder question

Fine — Q, K, and V come *from* the embedding. But derived **how**, exactly? What decides how to turn "the meaning of cat" into "what cat is looking for" (Q), versus "how cat can be found" (K), versus "what cat has to offer" (V)?

This is the real crux of the idea, and it's worth sitting with, because there's no hand-writable answer to it. Nobody sat down and wrote a formula that says *"to compute a Query, take the embedding and do X to it."* We don't actually know, in advance, what mathematical operation correctly extracts "what this word is looking for" out of a meaning vector — it isn't obvious, and it isn't fixed, because the right answer depends on the language, the sentence, and the specific relationship being detected.

This is exactly the moment where **training** enters the picture, and it's really the whole idea behind attention being a *learned* mechanism instead of a hand-engineered rule:

> Instead of hand-designing the rule that turns an embedding into a Query, we let the model **learn** that rule — by making the rule itself a grid of adjustable numbers (a matrix), and letting training discover which numbers make the whole system predict better.

That adjustable rule is `W_Q`. The same logic applies separately to `W_K` (the rule for "how should I be found") and `W_V` (the rule for "what do I hand over"). Three different jobs, so three different learned rules — three different matrices.

### An important correction: it's the *rule*, not the vectors, that starts random and gets trained

It's tempting — and very natural — to picture Q, K, and V themselves as "random values that get trained, then fine-tuned." That framing is close, but mixes up two different things, so it's worth separating them precisely:

```text
W_Q, W_K, W_V   →  the RULES.
                   These start random. Training reshapes THEM.

Q, K, V         →  the OUTPUTS of applying those rules to an embedding.
                   Freshly computed, every single time, for every
                   single token. Never stored. Never trained directly.
```

Once training is finished, computing `Q` for a brand-new sentence the model has never seen isn't a fuzzy, learned guess — it's a fixed, fast, completely deterministic multiplication: `embedding × (the now-trained) W_Q`. Nothing random happens at that point. What *was* random, and what training actually reshaped, is the content of `W_Q` itself — the numbers inside the rule, not the numbers that come out of applying it.

### The mechanics: how W_Q actually turns an embedding into Q

Let's make "apply a learned rule" completely concrete — this is the moment the embedding's information genuinely gets **projected into three useful parts**. Suppose a word's embedding is a 2-number vector:

```text
x = [2, 1]
```

And suppose `W_Q` is a small matrix — a grid of learned numbers, one column per output number we want:

```text
        col→Q₀   col→Q₁
W_Q  =  [ 1        4 ]
        [ 3        0 ]
```

To compute `Q`, we take the dot product of `x` with *each column* of `W_Q`:

```text
Q₀ = (2 × 1) + (1 × 3) = 2 + 3 = 5
Q₁ = (2 × 4) + (1 × 0) = 8 + 0 = 8

Q = [5, 8]
```

Now apply two *different* rules — `W_K` and `W_V` — to that exact same embedding:

```text
W_K = [ 0   1 ]      W_V = [ 3   2 ]
      [ 2   1 ]            [ 1   2 ]

K₀ = (2×0)+(1×2) = 2      V₀ = (2×3)+(1×1) = 7
K₁ = (2×1)+(1×1) = 3      V₁ = (2×2)+(1×2) = 6

K = [2, 3]                V = [7, 6]
```

Look at what just happened: **one embedding, `[2, 1]`, produced three genuinely different vectors** — `Q = [5, 8]`, `K = [2, 3]`, `V = [7, 6]` — purely because three different rules were applied to it. Nothing about the word's underlying meaning changed; that one vector simply got projected through three different lenses, splitting and re-emphasizing its information three different ways. In a real model, embeddings have hundreds or thousands of numbers, and `W_Q`, `W_K`, `W_V` are correspondingly large grids — but this exact same operation, dot-product-with-each-column, is all that's happening, at any scale.

*(Quick note if matrix multiplication is rusty: multiplying a vector by a matrix just means "compute one dot product per output column, using the same input vector every time." That's the entire operation, repeated as many times as there are output numbers.)*

### Why three separate rules, instead of one, or none?

Imagine someone asks you to build a recommendation engine from scratch, starting only from a customer ID.

First question: *"What should a customer ID become?"* You'd probably invent something like a **profile vector** — a list of numbers summarizing that customer. That's exactly an embedding.

Next question: *"Which customers are similar to each other?"* Comparing raw profile vectors directly is crude — you'd probably invent a small transformation first, one that emphasizes the specific traits relevant to *similarity*, and apply it to every customer before comparing them. That transformation is exactly what `W_Q` and `W_K` are — two separate rules, because "what I'm looking for" and "how I present myself to be found" are genuinely different jobs, even though both are ultimately in service of matching.

Final question: *"Once you know two customers are similar, what information should actually flow between them?"* That's a different question again — so you'd invent yet another transformation, specialized for *that* job. That's `W_V`.

The Transformer wasn't designed all at once as a monolithic idea — it's the accumulation of exactly these kinds of small, individually-motivated design decisions, each solving one specific sub-problem, each therefore earning its own learned rule.

### There's nothing inherently "question-like" about Q

One more correction worth making explicit: it's tempting to think of Q, K, V as manually assigned roles — as if the architecture *tells* a vector "you are now a question." That's not quite right. There is nothing intrinsically question-shaped about the numbers sitting inside a Q vector. The model *learns to use it that way*, purely because doing so improves its predictions during training. The labels Q, K, V describe the *role a vector plays in the computation*, not some inherent property baked into the numbers — that role emerges entirely from how `W_Q` gets shaped by training, not from anything the architecture assigns up front.

### So how does training actually shape W_Q into something useful?

This is the last piece of the puzzle: what does it actually mean for the *rule itself* to "figure out relevance," rather than some vector just happening to land on the right answer?

Every matrix starts filled with small random numbers — `W_Q` included. It has no idea, yet, what makes a good Query. After the model sees one training example and makes a prediction, a process called backpropagation (the full mechanics are in Chapters 10 and 11) nudges every number inside every matrix that contributed to the mistake, very slightly, in whichever direction would have made the prediction a little better:

```text
W_Q[0][0]:  0.83  →  0.84  →  0.85  →  0.86  → ...
            (after training example 1, 2, 3, 4...)
```

A single nudge might change a value by something like `0.0000003` — utterly negligible on its own. But after billions of tokens and trillions of such nudges, those microscopic adjustments accumulate into a matrix that reliably does its job: `W_Q` and `W_K` end up shaped so that, whenever they're applied, *genuinely related* words score highly against each other — and `W_V` ends up shaped so that the information it hands over is actually useful once found.

So, to directly answer the question this whole section opened with — *how does the model figure out relevance and relationships between vectors?* — the honest answer is: it isn't computed by any single clever formula. It's **discovered**, gradually, by reshaping `W_Q`, `W_K`, and `W_V` across an enormous number of examples, until the same simple operation — "multiply the embedding by this matrix" — reliably produces vectors that behave the way attention needs them to. In a very real sense, *this is what training a Transformer means*: not memorizing facts, but shaping these matrices until relevance and relationship between words falls naturally out of ordinary multiplication.

A more useful mental picture than "the model is learning facts" is: **the model is learning a geometric space.** Recall from Chapter 3 that similar words end up as nearby vectors after training. The exact same thing happens to Q-space and K-space — training organizes them so that words which *should* attend to each other end up geometrically close once projected through `W_Q` and `W_K`, even if they weren't especially close as raw embeddings.

### Which parts of attention are actually learnable?

It's worth being explicit about this, because it's easy to lose track once formulas start appearing. Here is the attention computation, with every step labeled as either a learned parameter or pure, fixed mathematics:

```text
Embedding                        (learned — Chapter 3)
     │
     ▼
Matrix Multiply (W_Q)            ← LEARNED
     │
     ▼
Matrix Multiply (W_K)            ← LEARNED
     │
     ▼
Matrix Multiply (W_V)            ← LEARNED
     │
     ▼
Dot Product (Q · K)              — pure math, nothing to learn
     │
     ▼
Scale (÷ √dₖ)                    — pure math, nothing to learn
     │
     ▼
Causal Mask                      — pure math, nothing to learn
     │
     ▼
Softmax                          — pure math, nothing to learn
     │
     ▼
Weighted Sum (with V)            — pure math, nothing to learn
```

Notice how few boxes are actually learnable. `W_Q`, `W_K`, and `W_V` are the *entire* set of trainable knobs inside one attention head — every other step is deterministic arithmetic that has no numbers of its own to adjust. This is exactly why these three matrices carry so much weight (pun intended): when people say "attention is learned," this — three matrices, and nothing else in the mechanism — is literally the whole truth of what that means.

---

## Part 5 — A full worked example, start to finish

Let's now see one complete attention computation, using real (if small) numbers throughout. We're processing the sentence:

```text
"The animal didn't cross the road because it was tired."
                                          ↑
                                    current word
```

We're standing at the word `"it"`, with everything before it already processed. Suppose that, after multiplying each word's embedding through `W_Q`, `W_K`, and `W_V` (exactly the operation we just walked through above), we get:

| Word | Q | K | V |
|------|------|------|------|
| animal | [3,1] | [2,1] | [9,2] |
| road | [2,3] | [0,2] | [4,8] |
| because | [1,2] | [2,2] | [2,5] |
| it | **[2,1]** | [1,1] | [3,3] |

We only need `Q(it) = [2,1]` — that's the question being asked right now. We compare it against every earlier Key using a dot product:

```text
Q(it)·K(animal)   = 2×2 + 1×1 = 5
Q(it)·K(road)     = 2×0 + 1×2 = 2
Q(it)·K(because)  = 2×2 + 1×2 = 6
```

| Word | Score |
|------|------:|
| animal | 5 |
| road | 2 |
| because | 6 |

**Wait — "because" scored highest. Does the model think "it" means "because"?**

No — and this is an important nuance. An attention score is never saying *"this is the correct referent."* It's saying *"this word has useful information for updating the current word."* "Because" might score highly because it signals a causal relationship that's useful context, even though "animal" is the actual thing being referred to. Attention is richer, and messier, than simple coreference resolution — multiple words can legitimately be useful for different reasons at once.

### Turning scores into weights: Softmax

Raw scores like `5`, `2`, `6` aren't directly usable — we want them to behave like percentages that add up to 100%. **Softmax** does exactly this conversion, turning any list of numbers into positive weights that sum to 1, while keeping larger scores proportionally larger. Suppose it gives us:

| Word | Weight |
|------|-------:|
| animal | 0.26 |
| road | 0.01 |
| because | 0.73 |

In plain terms: *"When updating the word 'it', listen 73% to 'because', 26% to 'animal', and basically ignore 'road'."*

### Building the new representation: a weighted blend of Values

Now — and only now — do the **Value** vectors get used. We blend them together using the softmax weights as a recipe:

```text
New representation of "it"
   =  0.26 × V(animal)  +  0.01 × V(road)  +  0.73 × V(because)
```

Computing each position:

```text
First number:  0.26×9 + 0.01×4 + 0.73×2 = 2.34 + 0.04 + 1.46 = 3.84
Second number: 0.26×2 + 0.01×8 + 0.73×5 = 0.52 + 0.08 + 3.65 = 4.25
```

```text
New representation of "it" = [3.84, 4.25]
```

We started with the word `"it"` on its own, meaning nothing beyond itself. We end with a new vector that has genuinely absorbed information from `"animal"`, `"road"`, and `"because"`, weighted by how relevant each one was judged to be. **This is the output of one attention head** — the core operation of the entire Transformer, running once.

*(A quick honest caveat: these numbers were invented purely to show the mechanics clearly. In a real trained model, `W_Q`, `W_K`, and `W_V` would very likely make "animal" score higher than "because" for this sentence — but the arithmetic works identically at any scale, and with real, trained weights instead of made-up ones.)*

---

## Part 6 — Keeping the scores well-behaved

There's a subtle numerical problem hiding in the process above. A dot product sums one multiplication per dimension — and real models don't use 2-dimensional vectors, they use hundreds. Summing more numbers tends to produce bigger totals:

```text
dk = 2    →  dot products around single digits     (fine)
dk = 128  →  dot products can reach 100, 150, 180+   (a problem)
```

Why is a big number a problem? Because Softmax uses the exponential function, `e^x`, which explodes extremely fast:

| Input | e^x |
|-------:|------------:|
| 1 | 2.7 |
| 5 | 148 |
| 10 | 22,026 |
| 20 | ~485 million |

If Softmax receives scores like `[180, 175, 172]`, the gap between them gets wildly exaggerated, and the result collapses into something like `[0.9999999, 0.0000001, 0.0]` — the model ends up ignoring every word except one, almost at random, simply because of arithmetic scale rather than genuine relevance. That's not what we want, especially early in training.

The fix: divide every score by `√dₖ` (the square root of the vector's dimension) before applying softmax. This keeps the scores in a sane range no matter how large the vectors get.

Putting it all together, this whole chapter compresses into one formula:

```text
Attention(Q, K, V) = softmax( (Q · Kᵀ) / √dₖ ) · V
```

Every symbol in that formula now has a story behind it: `Q · Kᵀ` is "compare my question against every key," `/ √dₖ` is "keep the scores numerically sane," `softmax(...)` is "turn scores into a percentage recipe," and `· V` is "blend the actual information using that recipe."

---

## Part 7 — The one rule GPT adds: no peeking at the future

There's one more constraint, specific to how models like GPT generate text one word at a time: **a word is only allowed to attend to itself and the words before it — never words that come later.**

Think of a school exam: if the answer key were accidentally printed on the question paper, every student would score perfectly without learning anything. Letting a model "attend to" a future word it's supposed to be predicting is exactly that mistake.

So GPT builds a **causal mask** — a simple allow/deny grid:

```text
          The    cat    sat
The        ✓      ✗      ✗
cat        ✓      ✓      ✗
sat        ✓      ✓      ✓
```

Notice the triangle shape — this lower-triangular pattern shows up in nearly every GPT-style implementation, and it's one of the easiest ways to recognize decoder-only code at a glance.

**How it's actually implemented:** before softmax runs, every forbidden (future) position in the score grid gets replaced with negative infinity:

```text
          The    cat    sat
The        3     -∞     -∞
cat        1      4     -∞
sat        6      3      2
```

Why negative infinity specifically? Because softmax involves `e^x`, and `e^(-∞) = 0`. So after softmax:

```text
          The    cat    sat
The        1      0      0
cat      0.05   0.95     0
sat      0.94   0.05   0.02
```

Forbidden positions receive *exactly* zero attention — they don't fade out, they vanish completely. Nothing about the softmax formula itself changed; we just quietly rigged the input so the impossible positions could never win any share of attention.

---

## The complete picture

```text
Embedding (meaning + position)
        │
        ▼
   x · W_Q  →  Q        ┐
   x · W_K  →  K        ├── the ONLY learnable steps in this whole diagram
   x · W_V  →  V        ┘
        │
        ▼
   Q · Kᵀ  for every pair of positions
        │
        ▼
   Apply causal mask   (future positions → -∞)
        │
        ▼
   Divide by √dₖ
        │
        ▼
   Softmax              (scores → percentages)
        │
        ▼
   Weighted sum of V     (blend information)
        │
        ▼
   New, context-aware representation of the word
```

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "A high attention score means 'this is the correct referent.'" | It means "this word carries useful information for updating the current word" — which is broader and messier than pronoun resolution. |
| "Q, K, and V are three unrelated things, manually assigned by the architecture." | They're three different learned *projections* of the exact same embedding — the "roles" are only meaningful because training makes them useful, not because they're inherently different in kind. |
| "We compare Q against V." | We compare Q against K to decide *how much* to listen, then read from V to decide *what* we actually receive. |
| "The model just creates as many random representations as possible and magically tunes them." | It creates a *specific, fixed set* of representations, each assigned a distinct job by the architecture (embedding = meaning, Q = what I seek, K = how I'm found, V = what I offer) — training fills in the numbers, it doesn't invent the jobs. |
| "Q, K, and V start random and get trained, then fine-tuned." | Close, but it's `W_Q`, `W_K`, and `W_V` — the *rules* — that start random and get trained. Q, K, and V themselves are freshly *computed* from the embedding every time, using whatever the current (trained) rule is. They are never stored, and never trained directly. |
| "W_Q, W_K, and W_V are some special, unique kind of object." | They're plain matrices — the same *kind* of learnable object as the embedding table (Chapter 3) or the feed-forward weights (Chapter 6), just used for a different purpose. |
| "Scaling by √dₖ changes what the model can learn." | It doesn't change *what* the model can represent — it only keeps the numbers well-behaved so training doesn't collapse into near-random guessing early on. |
| "Masking is a special separate mechanism." | It's just a small edit to the scores (forbidden positions → −∞) before the exact same softmax step runs. |

---

## Quick reference

```text
Attention(Q, K, V) = softmax( (Q·Kᵀ) / √dₖ  [+ causal mask] ) · V

Q = x · W_Q   "what am I looking for?"
K = x · W_K   "what can I be matched on?"
V = x · W_V   "what information do I carry?"

W_Q, W_K, W_V  =  the RULES.        Start random. TRAINED.
Q, K, V        =  the OUTPUTS.      Computed fresh every time. NOT trained directly.

W_Q, W_K, W_V are LEARNED MATRICES — the only trainable parts
of the whole attention computation. Everything else (dot product,
scaling, softmax, weighted sum) is fixed math with nothing to learn.

Match using K.  Copy from V.
Causal mask: future positions get score = -∞, so softmax gives them 0.
Divide by √dₖ before softmax, so scores don't blow up as dimension grows.
```

---

## The architecture so far

The diagram so far. The double-lined box is this chapter's addition.

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
║ ATTENTION (one head)                       ◄ NEW ║
║  x · W_Q → Q     x · W_K → K     x · W_V → V     ║
║  scores = Q · Kᵀ ÷ √dₖ                           ║
║  causal mask: future positions → −∞              ║
║  softmax: scores → weights                       ║
║  output = weights · V                            ║
╚══════════════════════════════════════════════════╝
                          │
                          ▼
               Context-aware vectors
```

Learned so far: embedding table, position table, and W_Q, W_K, W_V.

---

## What's next

We just walked through **one** attention computation — one "search" happening once, using one set of `W_Q`, `W_K`, `W_V` matrices. Real models run many of these searches in parallel, each with its *own* separate set of Q/K/V matrices, free to specialize in a different kind of relationship (grammar, meaning, reference...). Chapter 5 covers exactly that: **Multi-Head Attention** — and because we've now built single-head attention this thoroughly, multi-head will turn out to be a surprisingly small step.
