# Chapter 12 — Inference: The KV Cache, Sampling, Temperature and Top-p

## Introduction

Chapters 2 to 11 built a GPT-style model and trained it. Training happens once, on a huge cluster, and produces a set of fixed weights. **Inference** is what happens every time someone actually *uses* those weights: you type a prompt, and the model writes a reply, one token at a time.

Inference sounds simple, because it's just the forward pass from Chapter 8 in a loop. But two practical questions turn out to shape almost everything about how real systems like llama.cpp, vLLM and ChatGPT are built:

1. **How do we avoid redoing the same work for every new token?** The answer is the **KV cache**.
2. **How do we turn the model's probabilities into one actual word?** The answer is **sampling**.

This chapter covers both. Neither changes the model itself. They are about *how we run it*.

---

## Part 1 — The generation loop, and its hidden cost

Recall from Chapter 1 that a model writes by looping: predict one token, add it to the text, run again.

```text
"Hello"             →  model predicts  "there"
"Hello there"       →  model predicts  "!"
"Hello there!"      →  model predicts  next token...
```

Now think like an engineer who has to build this. At the second step, the input is `Hello there`. Does the model really have to process `Hello` **again**, all the way through every block, only to get to the new word `there`?

If it does, the cost adds up fast. Suppose we generate 500 tokens, and each step re-processes everything so far:

```text
Step 1:     process 1 token
Step 2:     process 2 tokens
Step 3:     process 3 tokens
   ...
Step 500:   process 500 tokens

Total:  1 + 2 + 3 + ... + 500  =  125,250 token-passes   (to produce only 500 tokens)
```

Almost all of that is repeated work: the model keeps re-reading words it has already read. Is there a way to avoid it?

---

## Part 2 — Rediscovering the KV cache

There are two options for the earlier tokens when a new token arrives:

```text
Option A:  recompute their Keys and Values from scratch every step
Option B:  keep the old Keys and Values somewhere, and reuse them
```

Option B looks obviously better, but it raises an objection that almost everyone has the first time. It's worth stating, because answering it *is* the whole idea.

> *"I'm feeding `Hello there` into the model as the new context. Doesn't the entire sentence still have to pass through all the blocks? Otherwise `there` would be an isolated token with no context."*

It's a fair worry. To answer it, look at what the new token actually needs from the old ones.

### What the new token needs

Recall the attention computation from Chapter 4. When the newest token, `there`, is processed, it builds its own query `Q(there)`. It then compares that query against the **Keys** of every token it can see (`K(Hello)` and `K(there)`), and takes a weighted blend of their **Values** (`V(Hello)` and `V(there)`).

Look at what appears in that description: `Q(there)`, `K(Hello)`, `K(there)`, `V(Hello)`, `V(there)`. What does *not* appear is `Q(Hello)`. **The old tokens' queries are never needed again.** Only their Keys and Values are. And `Hello` still participates in the new token's attention, just through its **stored** Key and Value rather than by being pushed through the whole network again.

### Why reusing them is safe

This works only if the stored Keys and Values are exactly the same as they would be if recomputed. They are, because of the causal mask from Chapter 4. A token can only look at itself and **earlier** tokens, never later ones. So when `there` and then `friend` are added after `Hello`, **nothing about `Hello` changes**: it gained no new information, since it was never allowed to look ahead. Its Key and Value are frozen the moment they are first computed. That is the only reason the cache is possible.

### A tiny worked example

Take two tokens, with small made-up vectors (and, to keep the arithmetic simple, leaving out the `√dₖ` scaling). The cache already holds the Key and Value of `Hello`:

```text
Cache (from the first step):   K(Hello) = [1, 0]      V(Hello) = [2, 4]

New token "there" arrives. Compute ONLY its own:
   Q(there) = [1, 1]     K(there) = [0, 1]     V(there) = [6, 2]

Add K(there), V(there) to the cache. Then attend over the cache:
   scores:   Q(there)·K(Hello) = 1     Q(there)·K(there) = 1
   softmax:  weights = [0.5, 0.5]
   output:   0.5 × V(Hello) + 0.5 × V(there) = 0.5×[2,4] + 0.5×[6,2] = [4, 3]
```

`Hello` clearly contributes to the answer, yet we never re-ran it through the network.

### Every block has its own cache

One more detail answers the next natural question: *"What about Block 2? Doesn't it need the updated outputs of Block 1 for all the old tokens?"* The answer is that **each block keeps its own K and V cache**, storing the Keys and Values it computed for every earlier token, from that token's own input to that block.

```text
New token arrives
      │
      ▼
Block 1:  compute this token's Q, K, V   →  append K, V to Block 1's cache  →  attend over cache 1
      │
      ▼
Block 2:  compute this token's Q, K, V   →  append K, V to Block 2's cache  →  attend over cache 2
      │
      ▼
   ...
Block 32: same
      │
      ▼
LM head → next token
```

Only the **new** token's computation runs through all the blocks. Every earlier token's contribution comes from the caches, one per block. For a 32-block model, that is 32 caches, each holding a Key and a Value for every token so far.

### The saving

With the cache, producing token 501 means computing the Q, K and V of one token, per block, and attending over what is already stored. Without it, you would recompute all 500 earlier tokens as well. Over a whole reply, the repeated quadratic work of Part 1 shrinks to one token's worth of work per step.

---

## Part 3 — Two very different phases: prefill and decode

The cache creates a split in how every request runs, and it explains something you've likely noticed. When you paste in a long document, nothing appears for a few seconds. Then the answer suddenly streams out quickly. Those are two different phases.

**Phase 1: prefill (prompt processing).** At the start, nothing is cached, so the model must compute the Keys and Values for **every token of the prompt**. It can do this for the whole prompt in one parallel pass, using the same causal-mask trick as in training (Chapter 9). This phase is heavy on computation, and it's the pause you feel before the first word. People call it the *time to first token*.

**Phase 2: decode (generation).** Now the cache exists. For each new token the model computes only that token's own Q, K and V, attends over the cache, and produces one word. Each step is light on arithmetic, but it must read the model's weights and the whole cache from memory on every single step, so decoding tends to be limited by how fast memory can be read, rather than by raw computing power.

| | Prefill | Decode |
|---|---|---|
| What happens | Process the whole prompt, build the cache | Add one token, reuse the cache |
| Parallelism | Many tokens at once | One token per step |
| Main cost | Computation | Reading memory (weights + cache) |
| What you feel | The pause before the answer | The speed the answer streams at |

Many people picture LLM inference as one continuous process. It is really these two distinct phases, and much of the engineering in later chapters (batching, paged memory, speculative decoding) is aimed at one phase or the other.

---

## Part 4 — The catch: the cache keeps growing

The cache makes generation fast, but it has a cost. **Memory.** Every new token adds one more Key and Value *per block*, and nothing is ever deleted:

```text
After 1 token:     K1 V1
After 2 tokens:    K1 V1   K2 V2
After 3 tokens:    K1 V1   K2 V2   K3 V3
   ...  (in every one of the model's blocks)
```

A model therefore has two very different memory demands:

```text
Model weights:   fixed size (for example about 16 GB for an 8B-parameter model in 16-bit)
KV cache:        grows with every token of context, for every user
```

How big does it get? Per token, the cache holds a Key and a Value for each block and each key/value head:

```text
KV cache per token  =  2 (K and V)  ×  blocks  ×  KV heads  ×  head dimension  ×  bytes per number
```

For a model shaped like Llama 3 8B (32 blocks, 8 key/value heads, head dimension 128, stored in 16-bit), that is `2 × 32 × 8 × 128 × 2 = 131,072` bytes, or **128 KiB per token**:

```text
    8,000 tokens of context   →  about  1 GiB
  128,000 tokens of context   →  about 16 GiB
  250,000 tokens (a 500-page book, at ~500 tokens per page)  →  about 30 GiB
```

(That model keeps only 8 key/value heads, a space-saving design covered in Chapter 17. A plain multi-head design would need several times more.)

This is why the maximum context length of a model, such as 128K tokens, isn't only a question of attention computation. The cache has to *fit in GPU memory*, and a server handling many conversations at once needs one cache per conversation. For long contexts the cache can become the main bottleneck.

A useful consequence: feeding a long document in pieces is no worse than all at once. Each new piece only needs *its own* new Keys and Values computed, because everything before it is already cached. The expensive part is the initial prefill of whatever has not been cached yet.

### What do systems do about it?

The pressure to shrink or manage the cache has produced a family of techniques, covered in later chapters:

- Reorganizing how the cache is laid out in memory (**PagedAttention**, Chapter 22)
- Storing the cache in fewer bits (**KV-cache quantization**, Chapter 14)
- Designing attention that needs a smaller cache to begin with (**GQA, MLA, sliding-window and sparse attention**, Chapter 17)
- Dropping or compressing old entries (**eviction and compression**)

### How long does the cache live?

An obvious question follows from "the cache keeps growing": it can't be kept for ever, so how long does it actually last? The answer is that the **KV cache is temporary working memory**. It exists for the lifetime of a running context, and then it is freed.

```text
A conversation starts
        ↓
Prompt tokens are processed   →  KV cache is built (prefill)
        ↓
Tokens are generated          →  KV cache grows, one entry per token
        ↓
The context ends or is reset  →  KV cache is freed
```

There are three levels to think about:

**1. During one generation.** The cache lives so the model never recomputes earlier tokens. This is exactly what we built in Part 2.

**2. Across the turns of a chat.** This depends on the serving system, and it is an implementation choice rather than a permanent property of the conversation. Most chat APIs are *stateless*: on every turn, the application sends the **entire conversation so far** as the prompt. The server then either:

- **reuses** a stored cache, if the start of the new prompt is identical to one it has already processed (this is called **prefix caching**, and many servers and API providers offer it, sometimes under the name "prompt caching"), or
- **rebuilds** the cache from scratch by running the prompt through prefill again (Part 3).

Reuse is purely a speed and cost optimization. The answer is the same either way.

**3. When the context is reset.** A new conversation starts with a new, empty cache, and the old one is deleted.

### But isn't a long conversation's cache huge, and stored somewhere?

Suppose you have been chatting with a model for hours, and the conversation is tens of thousands of tokens long. Is there one giant cache that has been sitting there since your first message? **No.** Keep three different things apart:

```text
Persistent memory     the conversation history, stored by the application
(database,            (as text, or as summaries). Permanent, until deleted.
 summaries)
      │
      ▼   for each request, the application chooses what to send
Model context         the tokens the model sees right now,
(the prompt)          which can never exceed the context window
      │
      ▼   the model processes that context
KV cache              the Keys and Values for those tokens.
(temporary)           Working memory for this run. Not your chat history.
```

The KV cache is not something you would normally save to disk as the record of a conversation. (Some servers can spill caches to CPU memory or disk so that they can be reused later, but that is a performance trick, not your conversation history.) The history lives in the application, and the cache is rebuilt or reused from it when needed.

### What if the text is longer than the context window?

Imagine feeding a 500-page book into one context, a line at a time. The cache grows with every line until it hits the model's context limit, or the server's memory limit. It doesn't magically become permanent storage. At that point something has to manage the context. The usual options are:

```text
Context too large
      │
      ├── discard the oldest parts
      ├── summarize the older parts
      ├── retrieve only the relevant parts when needed   (Chapter 24)
      └── start a new context
```

This is why real applications keep a **persistent memory outside the model**, separate from the model's context and its temporary cache. The model's window is working memory, not a library.

### Could we just compress the cache into a summary?

A natural idea: instead of keeping `K1 V1 ... K100000 V100000`, periodically compress everything so far into one compact representation and throw the originals away. It is a good idea, and it's an active research direction. But there is a real trade-off. A KV cache holds **exact, token-level** information, while a compressed one holds **higher-level** information. Replace the sentence *"The cat sat on the mat"* with a summary, and if someone later asks *"what color was the mat?"*, that detail may be gone. It's like compressing a photo to a smaller JPEG: it still looks fine until you zoom in.

It also helps to separate two ideas. A GPT has **context**, like RAM: everything in the window is still present and accessible. It does not have **memory** in the human sense, which constantly compresses and forgets detail while keeping the gist. Researchers are exploring ways to give models something closer to the second, including special memory tokens that summarize earlier text, recurrent memory states, and fixed-size state models (the same idea as the linear-attention designs in Chapter 17). Each one trades exact recall for a bounded memory.

---

## Part 5 — From probabilities to a word: sampling

Now to the second question. After the LM head and softmax (Chapter 8), the model hands us a **probability for every word in the vocabulary**. Something still has to choose one. That's the job of the **sampler**, and it is entirely separate from the model:

```text
Transformer → LM head → logits → [ sampling methods ] → one token
```

Temperature, top-k, top-p, greedy decoding and beam search are all **post-processing**. They don't change the model at all. They only change how we interpret its output.

### Greedy decoding, and why it isn't enough

The simplest rule is **greedy decoding**: always pick the single most probable word. It's deterministic, but it tends to be repetitive and dull. At the other extreme, picking purely at random from the full distribution produces nonsense, because some words have tiny but non-zero probabilities. What we want is somewhere between.

Think of a spinning wheel with slices proportional to each word's probability. The likely word wins most spins, an unlikely word occasionally wins, and variety comes naturally. Different use cases want different behavior. A medical or legal assistant should be consistent and reproducible, with minimal randomness. A creative-writing assistant should have more variety. The sampling settings below are how you choose.

### Temperature: changing the shape of the distribution

People often say temperature "controls creativity." That's the *effect*; the mechanism is mathematical. Temperature divides the logits by a number `T` **before** softmax:

```text
P(word i) = e^(zᵢ / T)  /  Σ e^(zⱼ / T)        z = the logits,  T = temperature
```

Take three words with logits `8, 6, 3`:

| Temperature | cat | dog | rabbit | Effect |
|---|---:|---:|---:|---|
| T = 0.5 | 98.2% | 1.8% | 0.004% | Much more peaked: nearly certain |
| T = 1 | 87.6% | 11.9% | 0.6% | The model's original distribution |
| T = 2 | 69.0% | 25.4% | 5.7% | Flatter: weaker words get a real chance |

Dividing by a number above 1 shrinks the gaps between logits, so softmax spreads probability out. Dividing by a number below 1 stretches the gaps, so the leader dominates. Temperature isn't telling the model "be creative." It controls **how much we trust the model's own confidence**.

At `T = 0` the formula would divide by zero, so implementations treat it as a special case: "temperature 0" simply means *use greedy decoding*. Some common starting points (not hard rules): code and medical or legal use around 0 to 0.3, general chat around 0.6 to 0.8, brainstorming around 0.8 to 1.0, and creative writing at 1.0 or above.

### Top-k: keep only the k best

Even at a sensible temperature, a vocabulary of 128,000 words has a long tail of absurd candidates (a "spaceship" to follow *"The cat chased the..."*). **Top-k** removes the tail: keep only the `k` most probable words, throw away the rest, and **renormalize** so the survivors sum to 100%.

```text
Original:   cat 35%   dog 25%   rabbit 15%   tiger 10%   ...   spaceship 0.3%

k = 3:   keep cat, dog, rabbit  (35 + 25 + 15 = 75%)
         renormalize:  cat 35/75 = 46.7%   dog 25/75 = 33.3%   rabbit 15/75 = 20.0%
```

Two edge cases clarify what `k` does: `k = 1` is just greedy decoding, and `k = vocabulary size` removes nothing, which is plain sampling.

**Top-k's weakness** is that it is blind to confidence. Compare two situations, both with `k = 5`:

```text
Situation A:  cat 90%, dog 5%, rabbit 2%, tiger 1%, others 2%    (the model is very sure)
Situation B:  20%, 19%, 18%, 17%, 16%, then 10% more             (the model is genuinely unsure)
```

The same `k = 5` keeps essentially everything useful in A (where one word dominates anyway), but cuts off 10% of the real probability mass in B. Top-k asks "how many tokens?" but never "how confident is the model?"

### Top-p (nucleus sampling): keep enough to cover the confidence

**Top-p** fixes this by asking a different question: *how many of the most likely words do I need before I've covered most of what the model believes?* The `p` is a **cumulative probability**, not a per-word threshold. (A common first guess is that `p = 0.90` means "keep words above 90%". It doesn't.) You keep adding words, from most to least likely, until their running total reaches `p`.

With `p = 0.90`:

| Word | Probability | Cumulative |
|---|---:|---:|
| A | 55% | 55% |
| B | 30% | 85% |
| C | 10% | **95%**, so stop here |
| D | 3% | 98% |
| E | 2% | 100% |

Keep A, B and C, renormalize them, and sample. The strength of top-p is that it **adapts**:

```text
A 92% sure model:      A alone already reaches 90%   →  keep 1 token
A nearly flat model:   needs many words to reach 90% →  keep many tokens
```

Top-k keeps a fixed count; top-p keeps however many the model's confidence calls for.

**One implementation detail matters a lot: the probabilities must be sorted from highest to lowest first.** If you accumulated in vocabulary order, you could end up keeping `rabbit (10%)` and skipping `dog (30%)` purely because of where they happened to sit in the list. The order of the vocabulary should never affect the result. So the procedure is:

```text
1. Compute probabilities (softmax).
2. Sort them from highest to lowest, remembering which token each belongs to.
3. Compute the running (cumulative) total.
4. Stop once it reaches p (the word that crosses the line is kept).
5. Renormalize the kept words.
6. Sample one.
```

Sorting 128,000 probabilities for every token sounds expensive, and it can be. Real implementations use partial-sort and selection tricks to find only the top candidates, but conceptually it is "sort, accumulate, cut."

### How they combine

Production systems rarely use just one. A common pipeline looks like:

```text
Logits  →  Temperature  →  Softmax  →  Top-p  (and sometimes Top-k)  →  Sample
```

Each stage removes a different kind of undesirable behavior: temperature reshapes the confidence, top-k limits the pool, and top-p adapts that pool to how sure the model is.

### Beam search, and why chat models mostly skip it

One more method deserves a mention because you'll meet it in older systems. **Beam search** doesn't commit to one word at a time. It keeps several candidate *sentences* alive. With a beam width of 2, say, it keeps both `I` and `We`, expands each with its best continuations, scores the full candidate sentences, keeps the best two, and repeats:

```text
Greedy:       choose one road at every step and never look back
Beam search:  explore a few roads at once, keep the best few at each step
```

Beam search became standard in translation, speech recognition and OCR, where there is usually **one correct answer** and the goal is to find the highest-probability sequence. But for open-ended chat it works poorly. The most probable sentence is often the most generic one, so beam search favors safe, boring text (a failure sometimes called mode collapse), and it multiplies the compute because several futures must be tracked at once. Conversation has thousands of good continuations, not one right one, which is why chat models use sampling instead.

---

## The complete picture: one request, end to end

```text
Your prompt
    │
    ▼
Tokenizer → token IDs                                 (Chapter 2)
    │
    ▼
PREFILL: run the whole prompt through all blocks in parallel,
         building each block's KV cache                (Part 3)
    │
    ▼
Logits for the last position → sampler → first token   (Part 5)
    │
    ▼
┌─────────────────────────────────────────────┐
│ DECODE LOOP (repeat for every new token):   │
│                                             │
│   new token → blocks (reading the caches,   │
│               adding its own K and V)       │
│            → LM head → logits               │
│            → temperature, top-k / top-p     │
│            → sample the next token          │
└─────────────────────────────────────────────┘
    │
    ▼
Stop when the model produces the end-of-sequence token (Chapter 2)
or a maximum length is reached. The tokenizer then turns the tokens back into text.
```

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "With the cache, the new token never sees the earlier text." | It sees it fully. It reads the earlier tokens' *stored* Keys and Values instead of recomputing them. |
| "The cache stores the old tokens' queries too." | Only Keys and Values are cached. Old queries are never needed again. |
| "There is one KV cache for the whole model." | Every block keeps its own cache, so a 32-block model has 32. |
| "Reusing the cache is an approximation." | It's exact, and it's valid because the causal mask means an earlier token's Key and Value never change once computed. |
| "LLM inference is one uniform process." | It has two phases: a compute-heavy prefill (the pause) and a memory-bound decode (the streaming). |
| "The model's memory use stays fixed during a conversation." | The weights are fixed, but the KV cache grows with every token, and it is a major reason for context limits. |
| "Temperature makes the model more creative." | It rescales the logits before softmax, flattening or sharpening the distribution. Creativity is the side effect. |
| "Top-p = 0.9 keeps words with probability above 90%." | It keeps the smallest set of top words whose *cumulative* probability reaches 90%. |
| "Top-p can scan the words in any order." | It must sort by probability first, or the vocabulary order would change the result. |
| "Beam search is just a better version of sampling." | It optimizes for the single most probable sequence, which suits translation but produces bland, generic chat. |

---

## Quick reference

```text
KV cache:    store K and V of every earlier token, per block; compute only the new token's
             Q, K, V.  Valid because the causal mask freezes earlier tokens.
Prefill:     whole prompt in parallel, builds the cache (compute-heavy; the pause)
Decode:      one token per step, reuses the cache (memory-bound; the streaming)
Cache size:  2 × blocks × KV heads × head dim × bytes, per token   (grows with context)

Sampling (after the model, before the token is chosen):
  Greedy       pick the most probable word
  Temperature  softmax(logits / T):  T<1 sharper, T>1 flatter, T=0 means greedy
  Top-k        keep the k most probable, renormalize
  Top-p        sort descending, keep until cumulative ≥ p, renormalize
  Beam search  keep several candidate sequences; good for translation, poor for chat

Typical pipeline:  logits → temperature → softmax → top-p (+ top-k) → sample
```

---

## What's next

So far, every model we've discussed learned only from next-word prediction on raw text. That produces a capable text predictor, but not yet an assistant that follows instructions or has a particular style. Chapter 13 covers how a trained model is adapted: supervised fine-tuning, parameter-efficient methods like LoRA, and learning from human feedback with RLHF and DPO.
