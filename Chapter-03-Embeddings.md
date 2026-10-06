# Chapter 3 — Embeddings: Turning Token IDs into Meaning and Position

## Introduction

Chapter 2 left us with a small cliffhanger. Our sentence:

```text
"The cat slept"
        ↓
    [10, 25, 41]
```

turned into a list of integers — and we were very insistent that those integers **mean nothing**. `25` is not "more cat-like" than `24`. It's just a row number in a table.

So where does actual *meaning* come from? That's this entire chapter.

---

## The big idea: use the ID as an address, not a value

Here's the concept that unlocks everything: the model never does math directly on the token ID. Instead, it uses the ID purely as an **address** to look something else up.

```text
Token ID = 25
        ↓
   "Go to row 25"
        ↓
  Read whatever is stored there
```

Think of a paper dictionary:

```text
Word
  ↓
"CAT"
  ↓
Page 145
```

You don't *calculate* page 145 from the letters C-A-T. You just look it up. Embedding lookup is exactly this — no multiplication, no neural network involved in this particular step, just a table lookup.

Or think of an employee ID at a company: the ID `25` doesn't tell you anything about the person. It's an address you use to pull up their actual record — name, role, department — from a separate database. The **token embedding table** is that database, and the "record" it stores for each word is a list of numbers called a **vector**.

```text
Embedding Table

Token ID     Embedding Vector
5      →     [0.2, 0.8]
10     →     [0.7, 0.1]
11     →     [0.3, 0.6]
12     →     [0.9, 0.5]
```

So:

```text
Token ID 10
     ↓
"Go to row 10"
     ↓
Read the vector
     ↓
[0.7, 0.1]
```

This is exactly the same idea as a dictionary in most programming languages:

```python
embedding_table = {
    5:  [0.2, 0.8],
    10: [0.7, 0.1],
    11: [0.3, 0.6],
    12: [0.9, 0.5],
}

token = 10
embedding = embedding_table[token]
```

A real model doesn't literally use a Python dictionary — it stores this table as one big matrix, with one row per vocabulary entry — but the idea is identical: **ID in, vector out, via lookup, not calculation.**

---

## Wait — where did those numbers come from?

This is the part that feels like magic the first time you see it, and it isn't.

When training a model **starts**, this table is filled with completely random numbers:

```text
Token ID     Vector
5   →   [0.12, -0.43]
10  →   [-0.87, 0.91]
11  →   [0.44, -0.18]
12  →   [-0.22, 0.73]
```

Meaningless. Random noise.

Then training happens (we'll cover *how* this actually works, in detail, in Chapters 9–11 — for now, just the shape of the idea): the model makes a prediction, the prediction is wrong by some amount, and a process called **backpropagation** nudges every number that contributed to the mistake, ever so slightly, in the direction that would have made the prediction better. That includes the embedding vectors.

```text
[-0.87, 0.91]
      ↓  (tiny nudge)
[-0.84, 0.95]
      ↓  (tiny nudge, next example)
[-0.80, 0.99]
      ↓
      ⋮
   (millions of nudges later)
      ↓
[1.92, -0.63, ..., 0.17]
```

One crucial detail: **the token ID itself never changes.** `"cat"` is always `10`. What changes, slowly, over millions of training examples, is *only* the content of row 10 in the table. By the end of training, that row has settled into a vector that genuinely captures something useful about how the word "cat" behaves in language.

```text
Sentence: "The cat slept"
        │
        ▼
   Tokenizer
        │
        ▼
  [5, 10, 12]
        │
        ▼
Embedding Lookup
        │
   ┌────┼────┐
   ▼    ▼    ▼
 Row5  Row10 Row12
  │      │     │
[0.2,  [0.7, [0.9,
 0.8]   0.1]  0.5]
```

Nothing mysterious happened. The token ID was simply used as an address into a table that *itself* got trained.

---

## What does a "dimension" actually mean?

In our tiny examples the vectors only have 2 numbers, e.g. `[0.7, 0.1]`. A real model uses vectors with hundreds or thousands of numbers — this size is called the **embedding dimension**, and it's a number the model's *architect* chooses before training even begins (not something the model learns on its own).

```text
Architect decides (fixed before training):
  Vocabulary size    — how many possible tokens exist
  Embedding dimension — how many numbers per vector (e.g. 768, 4096, ...)

Training learns (changes during training):
  The actual numbers inside the embedding table
```

It's tempting to hope that each individual number in the vector has a clean label — like "number 3 measures fuzziness" or "number 7 measures animal-ness." It doesn't work that cleanly in practice. No single dimension has a tidy, human-readable meaning on its own. What *does* end up meaningful is the vector **as a whole**, and especially its position relative to other vectors — which brings us to the next point.

---

## Why does this actually capture "meaning"?

Here's the payoff. Because training pushes these vectors around based on how words are *used*, words that get used in similar ways end up with similar vectors — which means they land **close together** in this abstract vector space.

```text
                    "dog" •
                            • "cat"
                     "kitten" •

                                        • "car"
                                    • "truck"
```

`"cat"` and `"dog"` end up near each other, because they tend to appear in similar contexts ("the ___ ran", "feed the ___", "my ___ is sleeping"). `"car"` and `"truck"` cluster somewhere else entirely. Nobody programmed this clustering by hand — it's a side effect of training on the simple goal of predicting the next word well. Similar usage patterns naturally pull similar words toward similar vectors.

This is also why the classic embedding party trick works: if you take the vector for `"king"`, subtract the vector for `"man"`, and add the vector for `"woman"`, you land very close to the vector for `"queen"` — the vector space ends up organized enough that relationships like "male → female" or "capital city → country" show up as consistent directions you can walk in.

```text
king − man + woman  ≈  queen
```

You won't need to compute this yourself anywhere in this handbook — it's included here purely because it's the clearest possible proof that these vectors capture real structure, not just arbitrary numbers.

---

## One more thing embeddings forget: order

Here's a problem hiding in plain sight. Suppose we rearrange a sentence:

```text
"The cat sat"        →   "sat cat The"
```

The embedding for `"cat"` is looked up from the exact same row of the exact same table, no matter where `"cat"` appears in the sentence. **The embedding for a word never changes based on its position.** That means, so far, our model would see these two sentences as containing exactly the same information:

```text
"Dog bites man."
"Man bites dog."
```

Same words, same embeddings, wildly different meaning. Word order clearly matters, but nothing we've built so far encodes it. That's a real gap — not a detail we can skip.

### The fix: a second table, just for positions

The solution is refreshingly simple: alongside the token embedding table, the model learns a completely separate **position embedding table** — one vector per position number (position 0, position 1, position 2, ...), learned the exact same way (random at first, nudged during training).

```text
Position   Position Embedding
   0    →  [0.1, 0.0, 0.2, 0.1]
   1    →  [0.0, 0.3, 0.1, 0.0]
   2    →  [0.2, 0.1, 0.0, 0.4]
```

Then, for every token, we simply **add** its word embedding and its position embedding together:

```text
"The" word embedding        [0.2, 0.6, 0.4, 0.1]
"The" is at position 0  +   [0.1, 0.0, 0.2, 0.1]
                         ─────────────────────────
Combined vector              [0.3, 0.6, 0.6, 0.2]
```

*This* combined vector — meaning **plus** position — is what actually enters the Transformer blocks (Chapter 4 onward), never the raw word embedding alone. Now the same word in two different positions produces two different final vectors, and order is no longer invisible to the model.

> **Common question:** doesn't attention (next chapter) already figure out relationships between words, including order? Attention can learn relationships like *"this word refers to that word,"* but without positional information it has no built-in sense of *before* or *after* at all — it would treat a sentence more like an unordered bag of words than a sequence. Position embeddings are what give it that missing sense of order to work with in the first place.

---

## The full picture, updated

```text
Sentence
   │
   ▼
Tokenizer  →  Token IDs
   │
   ▼
Token Embedding Lookup  +  Position Embedding Lookup
   │
   ▼
Combined Vectors  (meaning + position)
   │
   ▼
   (Chapter 4 picks up here: Attention)
```

And our running list of what's fixed-by-design versus learned-through-training now looks like this:

```text
Fixed by the architect (chosen once, never trained)
────────────────────────────────────────────────────
Vocabulary size
Embedding dimension

Learned during training
────────────────────────────────────────────────────
Token Embedding Table     — meaning of each word
Position Embedding Table  — meaning of each position
```

Keep this "architect-defined vs. learned vs. computed-on-the-fly" split in the back of your mind — it's a useful lens for almost every new component in the rest of this handbook, including attention, the feed-forward network, and even much more advanced things like Mixture of Experts.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "Embeddings are hand-designed to represent grammar or meaning." | They start as random numbers and are shaped entirely by training — nobody designs their content directly. |
| "One number in the vector represents one specific concept." | No single dimension has a clean, human-readable meaning. Meaning lives in the whole vector, and in its relationship to other vectors. |
| "Embeddings already encode word order." | They don't. Order is added separately, via a second, position-based table, and the two are combined by simple addition. |
| "A bigger embedding dimension always means a smarter model." | It gives the model more room to represent nuance, but it's an architectural choice with real memory and compute cost — not a free win. |

---

## Quick reference

```text
Token ID  →  Embedding Table lookup  →  Word meaning vector
Position  →  Position Table lookup   →  Position vector

Final input to the Transformer = Word vector + Position vector

Embedding tables start random, are shaped entirely by training.
Similar-meaning words end up with similar (nearby) vectors.
Embedding dimension is chosen by the architect; the vectors inside
the table are what training actually learns.
```

---

## The architecture so far

Here is the diagram so far. The double-lined box is this chapter's addition.

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
╔══════════════════════════════════════════════════╗
║ EMBEDDINGS                                 ◄ NEW ║
║  token ID → vector (learned embedding table)     ║
║  + position vector (learned position table)      ║
║  meaning + order, added together                 ║
╚══════════════════════════════════════════════════╝
                          │
                          ▼
     Vectors: one per token, meaning + position
```

Learned so far: the token embedding table, and the position embedding table.

---

## What's next

We now have, for every token, a single vector that encodes both *what it means* and *where it sits* in the sentence. Chapter 4 finally asks the question this whole handbook has been building toward: how does a word figure out which *other* words in the sentence it should pay attention to?
