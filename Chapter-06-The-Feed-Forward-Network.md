# Chapter 6 — The Feed-Forward Network: Expansion, Activation and GELU

## Introduction

Chapters 4 and 5 built the entire attention mechanism, and ended on a pointed observation: attention only lets words **gather** information from each other. It doesn't let a token actually *think* about what it gathered. That "thinking" step is a separate component — one that comes right after attention inside every Transformer block — called the **Feed-Forward Network (FFN)**, and it's exactly what this chapter goes deep on.

Compared to attention, it's refreshingly simple — no queries, no keys, no values, no softmax, no comparing tokens against each other at all. But don't mistake "simple" for "unimportant": the FFN typically holds a large share of a model's total parameters, and it's doing a genuinely different job from attention — one attention can't do on its own, no matter how many heads you give it.

---

## Part 1 — What problem is the FFN actually solving?

Take our running sentence, after it's been through attention:

```text
"The animal didn't cross the road because it was tired."
```

Suppose the word `"it"` now has the vector `[3.84, 4.25]` — the result from Chapter 4's worked example. This vector already contains information gathered from `"animal"`, `"road"`, and `"because"`. So: is the model done? Can it go straight from here to predicting the next word?

Here's the catch. Look at what attention actually *did* to produce that vector — it took a weighted blend of other tokens' Value vectors:

```text
0.26 × V(animal) + 0.01 × V(road) + 0.73 × V(because)
```

That's a **mix** of existing information. Attention can combine and re-weight information that's already present in the sentence, but it can't invent a genuinely new feature or draw a deeper conclusion from what it gathered — that's simply not the kind of operation a weighted average is capable of.

Here's an analogy that makes this concrete: imagine attention as gathering ingredients — tomato, cheese, bread. After attention, you have tomato + cheese + bread sitting on the counter. Have you made pizza? No — you've only *gathered* ingredients. Something still has to actually cook them. That "something" is the FFN.

Or think of it as Google Search: you ask *"what causes rain?"* and Google hands you a hundred relevant web pages. Has Google *answered* your question? No — it only **retrieved** information. Understanding those pages and drawing a conclusion is a separate step, done by *you*, afterward. Attention is the retrieval. The FFN is the understanding:

```text
Attention  →  retrieval / gathering
FFN        →  understanding / processing
```

Or, as a human analogy: imagine conducting an interview. Phase 1 is asking questions and collecting information — that's attention. Phase 2 is sitting alone afterward, thinking it over, and drawing your own conclusions — that's the FFN.

---

## Part 2 — What does the FFN actually look like?

Surprisingly, for something so important, it's just this:

```text
Input Vector
      │
      ▼
Linear Layer (W1)
      │
      ▼
Activation Function
      │
      ▼
Linear Layer (W2)
      │
      ▼
Output Vector
```

That's it — no attention, no other tokens involved, no communication between words at all. Just three operations, applied to one token's vector, completely independent of every other token in the sentence.

### Two new learned matrices — and nothing else

The FFN introduces exactly **two** new learned matrices, `W1` and `W2`. Compare that to everything attention needed — `W_Q`, `W_K`, `W_V` per head, plus `W_O` — and the FFN suddenly looks refreshingly light:

```text
Attention's learned parameters:      The FFN's learned parameters:
  W_Q, W_K, W_V (per head)             W_1
  W_O                                  W_2
```

Both `W1` and `W2` follow the exact same pattern you already know from Chapter 4: randomly initialized, then reshaped by backpropagation over training. Nothing new to learn about *how* they get trained — only about *what job* they do.

### Why isn't it just called a "Linear Layer," then?

A fair question, since it's built from two of them. A **linear layer** is one of the simplest operations in this entire handbook:

```text
Output = Input × W
```

You've actually already seen this exact operation — computing `Q = x · W_Q` in Chapter 4 *was* a linear layer. The FFN uses the same basic operation, twice, with something in between. The name "Feed-Forward Network" predates Transformers entirely — it's older neural-network terminology for any network where information only ever flows forward, layer by layer, with no loops and no memory of previous inputs. The Transformer's FFN is a small feed-forward network in exactly that older sense; it just happens to sit inside a much bigger architecture.

### Why not skip the activation function and use one big linear layer?

This is worth pausing on, because the reason is a bit surprising: **stacking two linear layers back-to-back, with nothing in between, is mathematically equivalent to just one bigger linear layer.** Two machines that only multiply and add numbers can always be combined into one machine that multiplies and adds numbers — nothing new becomes *possible* just by chaining them. So without something non-linear sitting between `W1` and `W2`, the "two-layer" FFN would secretly collapse into a single, ordinary linear transformation, and the extra layer would be pure wasted computation.

That "something in between" — the **activation function** — is what actually gives the FFN (and neural networks generally) their expressive power. We'll get to exactly what it does in Part 3.

---

## Part 3 — The activation function

This is worth slowing down for. Once you genuinely understand why activation functions exist, you understand not just the FFN, but a huge part of *why neural networks work at all* — this single idea shows up in essentially every neural network ever built, not just Transformers.

### Why do we need this, really?

Here's a question worth answering carefully, because the answer is more surprising than it first appears. Suppose you have this pipeline:

```text
Input → Linear Layer → Linear Layer → Output
```

Is that more powerful than just this?

```text
Input → One Bigger Linear Layer → Output
```

Surprisingly — **no.** This is one of the most important facts in all of deep learning, and it's worth seeing exactly why, not just accepting it.

A linear layer is `Output = Input × W`. Chain two of them together — first `× W1`, then `× W2` — and mathematically, that's *identical* to a single multiplication by the combined matrix `(W1 × W2)`:

```text
Input × W1 × W2   ≡   Input × (W1 × W2)
```

Two linear transformations always collapse into one. So if nothing else happens between them, the second layer contributes *nothing new* — it's mathematically redundant.

Here's a human-scale version of exactly the same idea. Imagine a company where every employee follows one single rule: *"multiply everything by 2."* Employee 1 doubles the number, hands it to Employee 2, who doubles it again, hands it to Employee 3, and so on. After four employees, the whole company has accomplished exactly one thing: multiplying by 16. It doesn't matter that four separate people were involved — nobody actually made a *decision*. It's one operation wearing a disguise of four steps.

Now think about what this means at real scale: **without an activation function, a 100-layer neural network is mathematically equivalent to a single linear transformation.** A hundred layers, collapsed into one. All of that depth, all of that computation — wasted.

### So what do we actually need?

We need something that **breaks linearity** — something that can say *"sometimes keep this information, sometimes suppress it, sometimes reshape it non-linearly,"* rather than just scaling everything by a fixed amount no matter what. That's exactly what an **activation function** is: a fixed rule, applied to each number in a vector individually, that introduces a genuine **decision point** into the computation.

Back to the company analogy: suppose one employee, instead of blindly doubling everything, follows a different rule: *"if the value is small, ignore it — if it's important, keep it."* Suddenly the company's behavior changes completely. It's no longer just scaling numbers uniformly; it's making context-dependent decisions about what to keep and what to discard. That's the qualitative shift an activation function introduces — not "another linear step," but the network's first opportunity to behave *non-uniformly*.

Here's a language-flavored version of the same idea. Suppose, deep inside a model processing the sentence `"The animal was tired,"` one internal number roughly corresponds to *"is this about a living thing?"*, another to *"is this about movement?"*, and another to *"is this about emotion?"* An activation function is what lets the network decide, case by case: *"emotion is very relevant here — keep it strongly,"* while *"movement isn't very relevant to this particular prediction — suppress it."* Without an activation function, all of these signals would just get blended together in a fixed, unconditional way, with no ability for the network to say "this one matters more, right now, in this context."

### What does an activation function actually *do*, mechanically?

It's worth being precise here, because it's easy to conflate this with what a linear layer does. A linear layer (like `W1` or `W2`) **mixes numbers together** — every output number is a weighted combination of *every* input number. An activation function does the opposite kind of thing: it looks at each number **one at a time, completely independently**, and applies the exact same fixed rule to it, with no mixing across positions at all.

```text
Linear layer (W1, W2):   mixes ALL input numbers together to produce each output number
Activation function:      transforms EACH number on its own, with no mixing at all
```

So the FFN's rhythm is: *mix* (via `W1`), then *individually filter/reshape* (via the activation function), then *mix again* (via `W2`). Mixing alone, no matter how many times you repeat it, stays linear. Alternating mixing with this kind of individual, non-linear filtering is what actually gives the network its expressive power.

### What the activation function is *not* for: keeping numbers in range

A common assumption is that the activation function is there to stop the vector flowing through the FFN from "blowing up" or shrinking toward zero, keeping the numbers in a manageable range. That's a different job, and the activation function doesn't do it:

- **Its job is non-linearity, not range control.** ReLU and GELU don't cap how large a value can get. A large positive input comes out just as large. Sigmoid does squash values into the 0–1 range, but that's a side effect, and it is exactly what caused the vanishing-gradient problem described below.
- **Range control is done elsewhere.** The `√dₖ` scaling inside attention (Chapter 4), Layer Normalization (Chapter 7), and careful choices for the starting values of the weights are what keep numbers in a workable range.
- **It acts on values, not parameters.** The weights `W1` and `W2` are never touched by the activation function. Only training changes them. The activation works on the numbers *flowing through* the network, the vector for one token as it passes through the FFN.

There is one true part in the intuition. Because the activation treats each number differently depending on its value, it does determine, for each input, which numbers pass through strongly and which are suppressed. ReLU, for example, zeroes out roughly half of them. So which parts of that vector are "active" really does depend on the token. But that comes from the non-linearity, not from keeping anything in range.

```text
Activation function  →  makes the computation non-linear, input by input
√dₖ scaling, LayerNorm, careful initialization  →  keep numbers in a workable range
```

### A brief, honest history — and *why* each one got replaced

Activation functions have a real history, and each transition happened for a concrete, well-understood reason — not just fashion.

**Sigmoid** was the early, natural choice. It smoothly squashes any input into a range between 0 and 1 — a shape that made intuitive sense, since it loosely resembles a biological neuron either "firing" or not. For a long time, it was the default.

The problem showed up once networks got deep. Picture the Sigmoid curve: for very negative inputs it hugs close to 0, for very positive inputs it hugs close to 1, and almost all of the actual *change* happens in a narrow band near the middle. Out on either tail, the curve is nearly flat — which means its **slope** (how much the output moves when the input moves slightly) is close to zero. That slope is exactly what backpropagation (Chapter 10 covers this fully) relies on to send a training signal backward through the network. Chain together many nearly-flat slopes — multiplying them together, layer after layer, which is literally what backpropagation does — and the signal reaching the earliest layers shrinks toward zero shockingly fast. This is the well-known **vanishing gradient problem**, and it made very deep Sigmoid-based networks notoriously difficult to train: the early layers would barely update at all, no matter how long training continued.

**ReLU** (Rectified Linear Unit) was the fix that took over for years afterward, and it's almost embarrassingly simple:

```text
If input < 0   →  output 0   (discard it completely)
If input ≥ 0   →  output unchanged
```

For positive inputs, ReLU's slope is a constant `1` — not shrinking, no matter how many layers you stack. The vanishing gradient problem, for the positive side at least, simply disappears. It's also extremely cheap to compute — just a comparison, no exponentials involved — which mattered a great deal as networks grew larger.

But ReLU trades one problem for a different one. For *negative* inputs, its output is flat, constant `0` — and the slope of a perfectly flat line is exactly `0`. If a particular neuron happens to receive negative inputs consistently during training (which genuinely does happen), its gradient becomes permanently zero — meaning it stops receiving any training signal at all, forever. This is known, somewhat dramatically but accurately, as a **"dead" or "dying" ReLU neuron**, and in large networks it's possible for a meaningful fraction of neurons to end up permanently stuck this way, silently wasting capacity. Separately, ReLU also throws away **every single negative value**, no matter how large or small — `-0.2` becomes `0`, `-4` becomes `0`, `-100` becomes `0` — which is sometimes too aggressive, since a small negative value can still carry genuinely useful information that a hard cutoff destroys outright.

**GELU** (Gaussian Error Linear Unit) — the activation most modern LLMs actually use — was designed to sidestep both problems at once. Don't be intimidated by the name; the idea is simple even though the name isn't. Like ReLU, large positive inputs pass through almost unchanged, with a gradient close to `1` — so the vanishing-gradient fix is preserved. But unlike ReLU, GELU is **smooth**, with no perfectly flat, zero-gradient region anywhere — even negative inputs retain at least a small trickle of gradient. That means a neuron using GELU can never become permanently, completely "dead" the way a ReLU neuron can.

Instead of ReLU's blunt rule, GELU behaves more like a graded confidence judgment:

```text
Large negative input   →  reduce it almost to zero, but not abruptly
Small negative input   →  reduce it gently, keep a little
Small positive input   →  keep some of it, not all
Large positive input   →  keep almost all of it
```

Compare all three side by side on the same inputs:

```text
Input:      [-4,   -2,   -1,   0,   1,     2,     4]
Sigmoid:    [≈0,  0.12, 0.27, 0.5, 0.73, 0.88,   ≈1]   (squashed into 0–1, flat at both tails)
ReLU:       [ 0,    0,    0,   0,   1,     2,     4]   (hard cutoff at 0)
GELU:       [≈0, -0.05, -0.16,  0, 0.84, 1.95,     4]   (smooth everywhere)
```

Notice ReLU makes a hard, binary decision exactly at zero, while GELU transitions smoothly through it. Think of it in terms of confidence: if a computation produces a small value like `0.05`, ReLU says *"positive — keep it fully,"* with no nuance. GELU instead says, roughly, *"I'm not fully convinced this matters — I'll keep some of it, not all of it."* That smoother, more graded behavior — combined with never fully "killing" a neuron the way ReLU can — is why GELU (and close relatives like SwiGLU, which several of the most recent models use instead) became the standard choice for large, very deep Transformer-based LLMs. We won't go deeper into SwiGLU and its relatives in this handbook, but it's worth knowing the family name if you go looking at real model code.

### An important correction: the activation function itself is not learned

It's tempting to describe GELU as if it's "deciding" or "judging" — useful shorthand, but not literally accurate. **GELU is a fixed mathematical function. It never changes, and it has no parameters of its own to train.** The exact same GELU formula runs at every position, in every layer, for every model that uses it — what actually gets shaped by training are the numbers inside `W1` and `W2`, which determine *what gets handed to* GELU in the first place. GELU just reliably applies the same fixed rule to whatever it receives.

```text
Learned (Chapters 9–11 train these):      Not learned (fixed forever):
  W1, W2                                 GELU itself (and Sigmoid, ReLU, etc.)
                                         Every matrix multiplication mechanic
```

---

## Part 4 — Expanding into a larger workspace

Now that we know why a non-linear step has to sit between the two linear layers, let's walk through the FFN's three steps in order: expand with `W1`, apply the activation function, compress with `W2`. This part covers the first one, with a genuine worked example. Suppose our token's vector, after attention, is:

```text
x = [3, 4]
```

The first linear layer multiplies this by `W1`. Suppose:

```text
        col0  col1  col2  col3
W1  =  [ 2     1     0     3 ]
       [ 1     2     2     0 ]
```

Computing each output number (dot product of `x` with each column):

```text
Output₀ = 3×2 + 4×1 = 6+4  = 10
Output₁ = 3×1 + 4×2 = 3+8  = 11
Output₂ = 3×0 + 4×2 = 0+8  =  8
Output₃ = 3×3 + 4×0 = 9+0  =  9

Result: [10, 11, 8, 9]
```

Notice what happened to the shape: a 2-number input turned into a 4-number output, purely because `W1` has 4 columns. **The shape of the weight matrix decides the output size** — nothing more mysterious than that, and it's the exact same mechanic as computing Q, K, or V in Chapter 4.

### Why deliberately make the vector bigger?

This is the strange-looking part worth sitting with. Why go from 2 dimensions to 4 (or in real models, from a few thousand to tens of thousands) only to shrink back down afterward?

Imagine your boss asks you to explain your entire company in exactly two words. Nearly impossible. Now imagine they instead say: *first, brainstorm freely using 100 words — then* summarize that down to two. Much easier, and the final two-word summary is likely far better, because it was distilled from a much richer draft rather than forced to be terse from the start. The FFN does exactly this:

```text
Small representation → Expand → Room to think → Compress → Better representation
```

Or think of it with LEGO: with only 2 bricks, there aren't many things you can build. With 200 bricks, there are vastly more possible structures. Expanding the vector temporarily gives the model a much larger "workspace" to form richer combinations of features, before compressing everything useful back down.

In real GPT-style models, this expansion is typically **4×**: an embedding dimension of `4096` commonly expands to `16384` inside the FFN, before compressing back to `4096`. This "expand, then shrink" pattern is one of the most recognizable signatures of a Transformer FFN in real code.

---

## Part 5 — Compressing back down

After the activation function, we compress the vector back to its original size using the second linear layer, `W2`. Continuing our toy example:

```text
Before FFN:        [3, 4]
After W1 (expand):  [10, 11, 8, 9]
After GELU:         [10, 10.8, 7.8, 8.9]     (illustrative — values shift slightly)
After W2 (compress): [9, 2]
```

The exact numbers here aren't the point — the *process* is: expand, apply a fixed non-linear function, compress back down. In a real model, this is the same `4096 → 16384 → 4096` shape mentioned earlier. And critically: **why shrink back down at all, rather than letting the vector keep growing?** Because every subsequent layer — every following attention computation, every following FFN — would have to work with an ever-larger vector at every single token, in every single layer. Left unchecked, that would make the model computationally impossible to run. So the FFN insists on returning to the same size it started with: think in a large, temporary workspace, then summarize your conclusions back down before handing them to the next block.

---

## Part 6 — What just happened, exactly?

This is the single most important realization in this chapter: **the FFN never communicates with any other token.** Attention was all about cross-token communication — comparing this word against every other word. The FFN does the complete opposite: it takes one token's vector, and transforms *only* that vector, with zero knowledge of what any other token's vector currently looks like.

```text
Attention  →  moves information BETWEEN tokens
FFN        →  creates richer features WITHIN one token
```

### An important clarification: "private," but not separately learned

It's tempting to say each token gets "its own private neural network" while passing through the FFN — and in one sense, that's true: each token's computation happens completely independently, with no cross-talk between tokens at this step, exactly the opposite of attention. But be precise about what's actually private here, because this is an easy point to get backwards:

```text
NOT private:  W1 and W2 themselves.
              There is exactly ONE W1 and ONE W2 per Transformer
              block — the same two matrices are reused, unchanged,
              for every single token, at every position.

PRIVATE:      The input and output VECTOR for each token, and the
              fact that no token's computation depends on any
              other token's computation during this step.
```

A better mental picture: the FFN is like a single rubber stamp — the exact same stamp, `W1` → activation → `W2` — pressed independently onto every token's vector, one at a time (conceptually; in practice, all at once, in parallel, for speed). It's one shared function applied many times, not many different functions.

---

## Part 7 — Couldn't `W_O` just do this job?

A very natural question at this point: `W_O` (Chapter 5) is also a learned matrix, and it already decides how much each attention head contributes. If we trained it well enough, couldn't it learn which heads matter, give the useful ones more weight, and make the FFN and its activation function unnecessary?

Half of that intuition is correct, and it's worth saying which half. **`W_O` really does learn head preferences.** Each head's output is multiplied by its own slice of `W_O` before the results are combined, so a head that turns out to be useful ends up with large weights there, and a head that isn't ends up with small ones. That is exactly why some heads can be removed from trained models with little effect.

But no amount of training can turn `W_O` into a replacement for the FFN. There are four reasons.

### Reason 1 — `W_O` is linear, and training can't change that

Training changes the *numbers inside* `W_O`. It can't change the *kind of operation* `W_O` performs, which is multiplication by a matrix. And there's a further wrinkle: each head's `W_V` and `W_O` are both linear, so together they behave like one single linear transformation. Making `W_O` better trained makes that linear transformation better, but it never becomes anything other than linear. Everything from Part 3 applies: a linear step can scale and mix what's already there, but it can't create genuinely new features.

### Reason 2 — its preferences are the same for every token

Once training is finished, `W_O` is a fixed grid of numbers. If it says "trust head 3 at 80%", that holds for every token, in every sentence. What you'd actually want is something like *"for a pronoun, listen to head 3; for a verb, listen to head 5."* That is a decision that **depends on the input**, and a fixed linear mix cannot make one. Decisions that depend on the input need a non-linear step, and that is exactly what the activation function provides. (Attention's softmax does make input-dependent choices, but between *tokens* inside a head, not between heads.)

### Reason 3 — attention only mixes; it can't compute new features

Whatever attention produces is, at heart, a weighted average of existing Value vectors. Here is a small example of what a mixing-only step cannot do. Suppose a token has two signals, `a` and `b`, each either 0 or 1, and the model wants a new feature that fires **only when both are present**:

```text
a  b     wanted
1  0  →    0
0  1  →    0
1  1  →    1
```

Any linear mix has the form `w1·a + w2·b`, so the "both present" case is always just the sum of the two single cases, and it can't be made special. Now add one fixed non-linear step, `ReLU(a + b − 1)`:

```text
a=1, b=0  →  ReLU(0)  =  0
a=0, b=1  →  ReLU(0)  =  0
a=1, b=1  →  ReLU(1)  =  1
```

The feature now exists. This is exactly the kind of thing the FFN does at scale, inside every token: it builds new combined features ("A *and* B", "A *but not* B") that mixing alone can never produce. A stack of attention-only layers does the opposite. It keeps re-averaging the same information, and tokens tend to become more alike rather than richer.

### Reason 4 — the FFN is more than its activation function

Even if you added a non-linearity right after `W_O`, you'd only have a very small FFN. It would lack the expansion from Part 4, the large temporary workspace in which rich feature combinations can form. The expanded FFN layers also hold a large share of a model's parameters, and evidence suggests that much of what a model has learned about language and the world lives there.

```text
Attention  →  decides WHERE to look (routes information between tokens)
W_O        →  mixes the heads together (linear, same for every token)
FFN        →  computes NEW features inside each token (non-linear, with room to work)
```

Each part does a job the others cannot, which is why every Transformer block contains both attention and an FFN.

---

## The complete picture

```text
Token's vector (after attention)
        │
        ▼
   x · W1              ← LEARNED (expands the dimension, e.g. 4096 → 16384)
        │
        ▼
   GELU (or similar)    — fixed math, nothing to learn
        │
        ▼
   (·) · W2             ← LEARNED (compresses back down, e.g. 16384 → 4096)
        │
        ▼
Token's new, "thought-about" vector
```

Applied completely independently, in parallel, to every token in the sequence — the same two matrices, reused everywhere.

### One analogy to hold it all together: a sack of mangoes

Imagine you're handed a sack of mixed mangoes, and you need to know what's really in it.

```text
The sack of mangoes        →  the token's vector, as it arrives from attention
Asking many questions      →  W1 (expanding into a larger workspace)
Grading each mango alone   →  the activation function
Packing a summary          →  W2 (compressing back to sack size)
```

1. **The sack.** This is the token's vector after attention: everything gathered so far, all mixed together.
2. **Spreading it out and asking questions (`W1`).** You tip the sack onto a large floor so there's room to look closely. Strictly, you aren't just laying the same fruit out in rows. Each spot on the floor is a *different question* about the whole sack: "how much ripe mango is in here?", "is it mostly raw?", "is anything spoiled?" `W1` is your trained eyes: through training it learns *which* questions are worth asking. Nobody writes those questions in advance.
3. **Grading each one (the activation function).** Now you judge each reading on its own, without comparing it to the others: strong signals pass through, weak ones are damped, and the very weakest may be dropped altogether. This is a fixed rule. It knows nothing about mangoes. It only decides how strongly each reading goes forward. (With GELU, "dropping" is usually gentle damping rather than throwing away. ReLU is the blunt version that zeroes weak readings completely.)
4. **Packing a summary (`W2`).** You now have thousands of readings but only a sack-sized space to put them in. `W2` learns which combinations of readings are worth keeping and writes a compact summary that fits back into the sack. It can't keep every detail, so it keeps the ones that proved most useful.

What comes back is the same size as what went in, but it now contains conclusions drawn from the contents, not just the contents themselves. In Chapter 7 you'll see one more step: the original sack isn't thrown away. The summary is added on top of it, which is the residual connection.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "The FFN lets tokens exchange information, just like attention does." | The opposite — the FFN's defining feature is that it does *not* look at any other token. All cross-token communication happens in attention, not here. |
| "Each token gets its own separately-learned FFN." | Every token is processed by the exact same shared `W1` and `W2` — one FFN per block, reused identically for every position. What differs per token is only the *input vector*, not the weights. |
| "The activation function (GELU) is learned, like the weight matrices." | GELU is a fixed formula with no trainable parameters. Only `W1` and `W2` are shaped by training — GELU always behaves the same way, everywhere. |
| "Expanding the dimension and then shrinking it back is wasted, pointless work." | The temporary expansion gives the model a larger "workspace" to form richer feature combinations, which is only possible because of the non-linear activation sitting in between the two linear layers. |
| "Two linear layers in a row would work just as well without an activation function." | Two linear layers with nothing non-linear between them mathematically collapse into a single linear layer — the activation function is what makes the second layer add anything new at all. |
| "ReLU was replaced because it's 'less accurate' in some vague sense." | ReLU has a specific, well-understood failure mode: for negative inputs its gradient is exactly zero, so a neuron stuck receiving negative inputs can stop training permanently (a "dead" neuron). GELU keeps a small gradient everywhere, so this can't happen. |
| "The activation function keeps the numbers from blowing up or shrinking away." | That's the job of `√dₖ` scaling, Layer Normalization, and careful weight initialization. ReLU and GELU don't cap large values at all; the activation function's job is to make the computation non-linear. |
| "The activation function mixes information the same way a linear layer does." | It's the opposite: linear layers (`W1`, `W2`) mix all numbers together; the activation function transforms each number individually, with no mixing across positions at all. |

---

## Quick reference

```text
FFN (applied independently, per token, same weights for every token):

  x  →  x·W1  →  Activation (GELU)  →  (·)·W2  →  output

W1, W2  = LEARNED matrices (only 2 — much simpler than attention).
GELU    = fixed function, NOT learned.

Typical shape in real models:  d_model → 4×d_model → d_model
  (e.g. 4096 → 16384 → 4096)

Attention: moves information BETWEEN tokens.
FFN:       creates richer features WITHIN one token, independently.

Without the activation function, W1 then W2 would collapse into
one ordinary linear layer — no extra expressive power gained.

History, and why each step happened:
  Sigmoid → vanishing gradients (flat tails kill backprop signal)
  ReLU    → fixes positive-side gradient, but negative side can
            "die" permanently (gradient exactly 0 forever)
  GELU    → smooth everywhere, never fully dies, standard in modern LLMs
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
┌──────────────────────────────────────────────────┐
│ MULTI-HEAD ATTENTION                             │
│  each head: softmax(Q·Kᵀ ÷ √dₖ + mask) · V       │
│  concatenate heads, then × W_O                   │
└──────────────────────────────────────────────────┘
                          │
                          ▼
╔══════════════════════════════════════════════════╗
║ FEED-FORWARD NETWORK (FFN)                 ◄ NEW ║
║  x · W1   expand, e.g. d → 4d                    ║
║  GELU     fixed non-linear step, not learned     ║
║  · W2     compress back to d                     ║
╚══════════════════════════════════════════════════╝
                          │
                          ▼
       Vectors the model has 'thought about'
```

Learned so far: embedding and position tables; W_Q, W_K, W_V per head; W_O; W1 and W2.

---

## What's next

We now have both halves of a Transformer block fully explained: Attention (Chapters 4–5) for gathering information across tokens, and the FFN (this chapter) for processing that information within each token. Chapter 7 covers how these two pieces actually get wired together — using residual connections and Layer Normalization — into one complete, stackable **Transformer Block**.
