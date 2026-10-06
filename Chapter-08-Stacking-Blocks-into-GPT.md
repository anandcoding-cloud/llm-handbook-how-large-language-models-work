# Chapter 8 — Stacking Blocks into GPT: From Text to a Next-Word Probability

## Introduction

Chapter 7 gave us one complete Transformer block: attention, residual, LayerNorm, FFN, residual, LayerNorm. That single block is the repeating unit of a Transformer, but a model with one block would be shallow and weak. This chapter does two things that turn that one block into an actual GPT-style model:

1. **Stack many copies of the block** on top of one another, so each token's vector gets refined again and again.
2. **Add the final step we've been missing**, turning the last vector into an actual word, so that the model can finally *predict* something.

By the end, you'll be able to trace a sentence through the entire forward pass of a GPT model, from raw text all the way to a predicted next token.

---

## Part 1 — From one block to many

Let's follow one token through the stack. Take the word `"it"` in our running sentence, and suppose its starting embedding is `[0.3, 0.9]`. It enters Block 1 and comes out as, say, `[1.8, 2.1]`:

```text
Embedding  →  Block 1  →  [1.8, 2.1]
```

Now notice something important. **We do not go back to the embedding table.** The output of Block 1 simply becomes the *input* of Block 2:

```text
Embedding → Block 1 → Block 2 → Block 3 → ... → Block N
 [0.3,0.9]  [1.8,2.1]  [3.4,4.8]  [5.1,6.7]
```

This only works because of a design choice we've seen in passing: **every block takes in a vector of a certain size and gives back a vector of exactly the same size.** If the embedding is 4096 numbers, then every block receives 4096 numbers and returns 4096 numbers. The *meaning* of the vector changes dramatically from layer to layer, but its *shape* never does. That is precisely why you can stack dozens of identical blocks, one after another, with no adapters in between.

```text
Embedding (4096)
   ↓
Better embedding (4096)
   ↓
Even better embedding (4096)
   ↓
Final contextual embedding (4096)
```

When people say "a 32-layer Transformer", this is all they mean: **32 Transformer blocks stacked one after another.**

### Why repeat the same kind of block?

Each block isn't learning *different words*. It's making the **same token's representation richer**, one step at a time. Think of reading a dense legal contract:

```text
First reading:   general understanding
Second reading:  notice the important clauses
Third reading:   understand the implications
Fourth reading:  spot contradictions
```

You're reading the *same document* each time, and your understanding gets deeper with each pass. In the same way, our token might go from "I am the word *it*" (Block 1), to "I probably refer to the animal" (Block 2), to "I understand that *because* expresses causality" (Block 3), and onward toward very rich, abstract information in later blocks. Chapter 7's residual connections are what make this safe: each block *adds* its refinement on top of what the previous blocks already built, rather than overwriting it.

---

## Part 2 — Same design, separate weights

Here is one of the most common misconceptions about stacked Transformers:

> *"All the blocks share the same weights, so the model just reuses one block over and over."*

**They do not.** Every block has the **same architecture** but its **own completely separate learned weights**:

```text
Block 1                Block 2                Block 3
───────                ───────                ───────
W_Q₁ W_K₁ W_V₁         W_Q₂ W_K₂ W_V₂         W_Q₃ W_K₃ W_V₃
W_O₁                   W_O₂                   W_O₃
W1₁, W2₁               W1₂, W2₂               W1₃, W2₃
LayerNorm γ₁, β₁       LayerNorm γ₂, β₂       LayerNorm γ₃, β₃
```

Think of a hospital. A patient moves from Doctor 1 (diagnosis) to Doctor 2 (treatment) to Doctor 3 (rehabilitation). Every doctor follows the same general routine, which is to examine, think, and update the notes. But each has different expertise. Transformer blocks work the same way: identical *structure*, different *learned expertise*.

This is also a big part of why modern models have billions of parameters. The architecture repeats, but the learned numbers do not. A 32-block model has 32 separate sets of `W_Q`, `W_K`, `W_V`, `W_O`, `W1`, `W2`, `γ` and `β`.

---

## Part 3 — What does each block actually contribute?

Researchers who study trained models have found that different depths tend to contribute different kinds of refinement. As a rough, *simplified* pattern:

| Where in the stack | What tends to happen there |
|---|---|
| Early blocks | Grammar, punctuation, local word-to-word structure |
| Middle blocks | Phrase structure, relationships between parts of the sentence |
| Later blocks | Higher-level meaning, abstract concepts, more complex reasoning patterns |

Treat this as a tendency, not a rule. Different models spread the work out differently, and the first block isn't "worse" than the last. They simply contribute different refinements.

There's also an important correction to the way this is sometimes told. It's tempting to picture "Block 1 stores grammar, Block 2 stores syntax, Block 3 stores facts", as if each block were a filing cabinet. That isn't quite right. A block is better thought of as a **learned procedure for transforming a representation**, not a database of sentences. A calculator doesn't contain the answer to `789 × 432`. It contains the *procedure* that produces the answer for whatever you type. Likewise, the model doesn't contain the sentence "Paris is the capital of France" stored somewhere. What it contains are billions of trained weights that, together, transform a representation of "the capital of France is…" into one that makes "Paris" the likely next word. Knowledge is real, and it's consolidated into the weights, but it's spread across many weights and layers rather than stored as retrievable text.

---

## Part 4 — Why not just one giant block?

A tempting question: after training, all the knowledge is already learned. So at prediction time, why run 32 or 80 blocks in sequence? Couldn't we merge all of them into a single block?

The answer comes straight from Chapter 6. **If every operation in the model were linear, you could merge everything.** Chaining `× W1 × W2 × W3` is the same as one multiplication by `W1·W2·W3`. A hundred layers would collapse into one.

But the moment a non-linear step sits between them (GELU, softmax, LayerNorm), that collapse stops working:

```text
Linear → GELU → Linear      cannot be combined into one matrix
```

Each block's output depends on the output of the one before it *through* a non-linearity, so the blocks can't simply be merged without losing something.

There's a second reason, which is about *how* complex things get learned. Imagine each layer builds on the previous abstraction:

```text
Layer 1:  dog → animal
Layer 2:  animal → living thing
Layer 3:  living thing → can breathe
```

Squeezing all three into one layer would require that single layer to learn every level of abstraction at once, which is much harder than learning them in stages. It's like teaching a child addition, then multiplication, then algebra, then calculus, rather than handing them one giant book of all mathematics. Mathematically, the full model is a **composition of functions**: `f₃(f₂(f₁(x)))`. A single function that reproduces a deep composition exactly, or even approximately, is usually far harder to find than learning the steps one by one. This is what the "deep" in deep learning really means: building increasingly sophisticated representations through a sequence of manageable transformations, rather than solving everything in one leap.

Researchers have tried to compress deep models into shallower ones (a field called **model compression**). It sometimes works surprisingly well, but the smaller model almost always loses some capability. There's no free lunch.

### Three ways to make a Transformer bigger

Since the stack can't be collapsed, designers have three knobs for increasing a model's capacity, all of which are architecture choices fixed before training:

```text
Wider   →  larger embedding size        (each vector carries more information)
More heads  →  more attention heads     (more relationships learned in parallel)
Deeper  →  more blocks                  (more sequential refinement steps)
```

Going deeper gives every token more refinement steps. The tradeoff, as always, is more computation, more memory, and slower responses.

### Heads versus blocks: width versus depth

Chapter 5 said different heads can pick up different relationships, and this chapter says more blocks give deeper refinement. That raises a natural question: when the model needs to handle some kind of relationship, does the work happen in a head or in a block? The answer is that heads and blocks are two different dimensions of the model, and they do different jobs:

```text
Heads  →  WIDTH.  They run in parallel, inside ONE block. They all read the
          SAME input at the same moment, and each looks at it in its own way.

Blocks →  DEPTH.  They run one after another. Each block reads the OUTPUT of
          the previous one, so it can build on what earlier blocks already did.
```

An assembly line is a good picture for both. Each **block** is one *stage* on the line. The **heads** are the specialists working *side by side at that stage*. They don't wait for each other, but the next stage can use everything they produced.

Here is an example to make it concrete. Take `"The trophy didn't fit in the suitcase because it was too big."` To work out what `"it"` refers to, one plausible division of labor looks like this:

```text
Block 1 (heads work in PARALLEL on the same input)
   Head A: links "it" to "trophy"
   Head B: links "it" to "suitcase"
   Head C: notices "too big" is a property being described
        ↓   (all three results are mixed together by W_O, then refined by the FFN)
Block 2 (works on the COMBINED result of Block 1)
   Uses all of it together: the thing that was "too big" to fit is the trophy,
   so "it" most likely means "trophy"
```

This is only an illustration. Real heads are rarely this tidy, as Chapter 5 explained. But it shows the pattern:

- **Heads in the same block** can't use each other's findings, because they all start from the same input. That makes them good for gathering several independent pieces of information at once.
- **A later block** can use the combined findings of the earlier ones. That makes depth necessary whenever one conclusion depends on another conclusion being reached first.

So why can't we pick which one handles a given kind of relationship? Because nobody assigns that job. The architect chooses only the *counts*: how many heads per block and how many blocks. Those counts set how much room the model has for parallel information-gathering and for step-by-step refinement. Which head or which block ends up doing what is settled by training.

```text
More heads  →  more things can be looked at in parallel, at the same stage
More blocks →  more steps where one conclusion can build on the last
```

---

## Part 5 — Counting parameters

It's useful to have a rough sense of where a model's parameters actually live. Let `d` be the embedding size. In a standard GPT-style block:

```text
Attention:  W_Q, W_K, W_V, W_O  →  4 matrices of size d × d   ≈  4d²
FFN:        W1 (d × 4d) and W2 (4d × d)                       ≈  8d²

One block  ≈  12d² parameters   (LayerNorm adds a negligible amount)
```

So a model's blocks hold roughly `12 × d² × (number of blocks)` parameters, plus the embedding table (`vocabulary size × d`). Try it on GPT-2 Small, which has `d = 768`, 12 blocks, and a vocabulary of about 50,000 tokens:

```text
Blocks:      12 × 12 × 768²      ≈  85 million
Embeddings:  50,257 × 768         ≈  39 million
Total                              ≈  124 million parameters
```

That matches the published size of GPT-2 Small. Real models differ in details (biases, different FFN shapes), so treat `12d²` as a back-of-the-envelope tool, but it shows an interesting fact: the FFN holds about two-thirds of every block's parameters, and doubling the embedding size roughly *quadruples* the parameters in each block.

---

## Part 6 — The missing last step: from vectors to words

After the final block, every token has a rich vector. Researchers call the final one the **hidden state**, written `H`. But `H` is still just numbers. Suppose, for the last token of our sentence, `"The animal didn't cross the road because it was"`, it is:

```text
H = [2, 3]
```

It conceptually contains everything the model has understood (the animal, the movement, the reason, the tiredness), but it isn't a word, and it isn't a probability. How do we turn it into the next word?

### The output matrix (the "LM head")

We add one more learned matrix, usually called the **LM head**, the **output projection**, or the **vocabulary projection**. Its shape is set by the vocabulary: it has one output column for **every word the model could predict.** For a tiny four-word vocabulary and a 2-number hidden state, it's a `2 × 4` matrix:

```text
              tired  hungry  asleep  running
W_vocab  =  [  -2      3       0        1  ]
            [   1      1       1        1  ]
```

Multiplying `H` by it, the same dot-product-with-each-column operation as every other matrix in this handbook:

```text
tired:    2×(-2) + 3×1 = -1
hungry:   2×3    + 3×1 =  9
asleep:   2×0    + 3×1 =  3
running:  2×1    + 3×1 =  5
```

We get one score per word. These are called **logits**, meaning raw, unnormalized scores. They aren't probabilities. They can be negative, huge, or anything else.

### Softmax, again

We now need probabilities, and we already know the tool for that. It's the same **softmax** from Chapter 4:

| Word | Logit | Probability |
|---|---:|---:|
| tired | -1 | 0.00004 |
| hungry | 9 | 0.980 |
| asleep | 3 | 0.002 |
| running | 5 | 0.018 |

These add up to 1. Earlier, softmax answered *"which token should I pay attention to?"*. Here the mathematics is identical, but it answers a different question: *"which word should I predict?"* The model then picks the most likely word, or samples from the distribution (Chapter 12 covers how), and the sentence grows by one word.

### But the toy model said "hungry"...

The real sentence continues *"…because it was tired."* Our toy model confidently predicted "hungry". That isn't a flaw in the *process*. It simply tells us our numbers are made up and untrained. A real, trained model would have learned weights that make "tired" much more likely here. What matters is the pipeline, not the toy prediction.

### Weight tying: the same space, used in both directions

There's an elegant trick many GPT-style models use. Recall the embedding table, which maps **word → meaning**. The LM head does the reverse, mapping **meaning → word**. Each column of the LM head is, in effect, a direction associated with one word, and its score for that word measures how well `H` lines up with it. That's very close to what each word's embedding is already doing. So many models save parameters by using the **same matrix for both jobs**, the embedding table in one direction and its transpose as the LM head in the other. This is called **weight tying**. Not every architecture does it, but many do.

```text
Embedding table:  word  → meaning
LM head:          meaning → word      (often the same matrix, transposed)
```

---

## Part 7 — The complete forward pass

We can finally draw the whole picture, from raw text to predicted word:

```text
Sentence
    │
Tokenizer                      (Chapter 2)
    │
Token IDs
    │
Embedding lookup + position    (Chapter 3)
    │
┌───────────────────────────┐
│  Transformer Block 1      │  ← Attention (4–5), FFN (6),
│  Transformer Block 2      │    residual + LayerNorm (7)
│  ...                      │    repeated N times,
│  Transformer Block N      │    each with its own weights
└───────────────────────────┘
    │
Final hidden state H
    │
LM head (vocabulary projection)
    │
Logits
    │
Softmax
    │
Probability distribution
    │
Next token
```

And a summary of what's fixed, what's learned, and what's computed along the way:

```text
Fixed by the architect (before training)
────────────────────────────────────────
Vocabulary size · Embedding dimension · Number of heads · Number of blocks

Learned by training
────────────────────────────────────────
Token embedding table · Position embedding table
Per block:  W_Q, W_K, W_V (per head), W_O, W1, W2, LayerNorm γ and β
LM head (unless tied to the embedding table)

Computed fresh on every forward pass (never stored)
────────────────────────────────────────
Token IDs · Embeddings · Q, K, V · Attention scores and weights
Head outputs · Residual outputs · LayerNorm outputs · FFN outputs
Final hidden state H · Logits · Probabilities
```

This is the moment worth pausing on. If someone asked you *"how does GPT predict the next word?"*, you could now explain every major step. One question remains: **everything above assumes all those matrices already contain meaningful numbers.** When a model is first created, every one of them is essentially random. How do billions of random numbers turn into something intelligent? That is the subject of the next chapter.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "All the blocks share the same weights." | Every block has the same *architecture* but its own separate learned weights. That's a big reason models have billions of parameters. |
| "Each block feeds back to the embedding table." | Only the very first block reads embeddings. Every later block takes the *previous block's output* as its input. |
| "Block 1 stores grammar, Block 2 stores facts, and so on." | Roughly useful as a tendency, but blocks are learned *procedures*, not filing cabinets. Knowledge is spread across the weights of many layers. |
| "We could merge all the blocks into one after training." | Non-linear steps between blocks (GELU, softmax, LayerNorm) stop them collapsing into a single matrix, and a single function that reproduces the whole stack is much harder to find. |
| "Logits are probabilities." | Logits are raw scores that can be negative or very large. Softmax is what turns them into probabilities that sum to 1. |
| "A wrong prediction from a toy example means the process is wrong." | The process is the same for toy and real models. A wrong prediction usually just means the weights haven't been trained. |

---

## Quick reference

```text
GPT = embeddings + N stacked Transformer blocks + LM head + softmax

Every block:  vector in (d numbers)  →  vector out (d numbers)
              same architecture, separate learned weights

Final hidden state H  →  × LM head  →  logits (one per vocabulary word)
                      →  softmax    →  probabilities  →  next token

Parameters per block ≈ 12·d²   (4d² attention + 8d² FFN)
Make a model bigger by going wider, adding heads, or going deeper.

Weight tying: the embedding table is often reused (transposed)
as the LM head.
```

---

## The architecture so far

The diagram so far, now the whole model. The double-lined boxes are this chapter's additions.

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
║ TRANSFORMER BLOCK  ×  N                    ◄ NEW ║
║  attention → add → norm → FFN → add → norm       ║
║  N blocks stacked, each with its own weights     ║
║  (the full block is drawn in Chapter 7)          ║
╚══════════════════════════════════════════════════╝
                          │
                          ▼
                Final hidden state H
                          │
                          ▼
╔══════════════════════════════════════════════════╗
║ LM HEAD                                    ◄ NEW ║
║  H · W_vocab  →  one score (logit) per word      ║
║  often tied to the embedding table               ║
╚══════════════════════════════════════════════════╝
                          │
                          ▼
╔══════════════════════════════════════════════════╗
║ SOFTMAX                                    ◄ NEW ║
║  logits → probabilities that sum to 1            ║
╚══════════════════════════════════════════════════╝
                          │
                          ▼
           Next token (picked or sampled)
```

Learned: embedding and position tables; in every one of the N blocks: W_Q, W_K, W_V, W_O, W1, W2, γ, β; and the LM head (unless tied to the embeddings).

This is the complete forward pass of a GPT-style model. Running it again, with the new token added, produces the following word.

---

## What's next

The forward pass is complete, but every matrix in it starts out random. Chapters 9 to 11 explain how they get trained: how the model learns from raw text, how a loss function measures a wrong prediction, and how backpropagation and an optimizer called AdamW nudge billions of parameters, step by step, until the random numbers become intelligent.
