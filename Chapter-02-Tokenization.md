# Chapter 2 — Tokenization: Turning Text into Tokens with BPE

## Introduction

In Chapter 1 we said the very first step of the pipeline is:

```text
Your text  →  Tokenizer  →  Token IDs
```

That looks like a small, almost boring step compared to attention or training. It isn't. Tokenization decides *what a model is even capable of seeing*. Get an intuition for it now, and a dozen confusing things later in this handbook — why the model sometimes misspells rare words, why some languages are more expensive to run than others, why the vocabulary size shows up in formulas — will make immediate sense.

---

## The core question

A neural network only understands numbers. It has no idea what the letters `"c"`, `"a"`, `"t"` are. So before any "understanding" can happen, we need a strict, mechanical answer to one question:

> **How do we turn text into numbers, and back again?**

That's the tokenizer's entire job. Nothing more.

---

## Step 1: Chop the text into pieces

Suppose the sentence is:

```text
"The cat slept"
```

A tokenizer first breaks this into smaller pieces called **tokens**. For a simple example, let's pretend each whole word is one token:

```text
"The cat slept"
        ↓
   ["The", "cat", "slept"]
```

(We'll come back to why real tokenizers don't actually split on whole words — see "The reality: subword tokenization" below. For now, whole words make the idea easiest to see.)

---

## Step 2: Turn each piece into a number

Every tokenizer comes with a fixed **vocabulary** — a big lookup table that was decided once, before training ever started, and never changes afterward. Suppose ours looks like this:

| Word | Token ID |
|------|---------:|
| The | 10 |
| cat | 25 |
| slept | 41 |

So our sentence becomes:

```text
"The cat slept"
        ↓
    [10, 25, 41]
```

That's it. That's tokenization. A sentence in, a list of integers out.

---

## The most important thing to understand: the ID means nothing

This is the part almost every beginner tutorial glosses over, and it's worth sitting with.

> **The number 25 does not mean "cat-like" in any way.** It's not a measurement of anything. It's just an arbitrary index — a row number in a table.

Think of it exactly like an **employee ID**:

```text
Employee ID 25 → "works in Cat Department"?
```

No. Employee 25 is just whoever got assigned that number when they joined. The ID carries no information about the person — it's purely an address you use to go look up their actual record somewhere else.

```text
"cat" → 25 → (just an address, not a meaning)
```

The *meaning* of "cat" doesn't live in the number 25 at all. It lives one step later, in the **embedding table** — which is exactly what Chapter 3 is about. For now, the only thing to lock in is:

```text
Tokenization = text → arbitrary ID numbers.
Nothing about "meaning" has happened yet.
```

---

## Why bother chopping text up at all?

You might reasonably ask: why not just feed the model raw letters, or even raw computer bytes?

You *can* — some models do exactly that (character-level or byte-level models). But it comes with a tradeoff, and it's worth seeing why most large models avoid it:

```text
Character-level:
"tokenization" → t-o-k-e-n-i-z-a-t-i-o-n   (12 tokens for one word!)

Word-piece level:
"tokenization" → token + ization             (2 tokens for the same word)
```

Every extra token means:
- more steps the model has to process,
- more opportunities to lose track of long-range structure,
- more compute and more memory (remember, cost scales with the number of tokens).

So there's a real engineering incentive to make each token carry as much "chunk of meaning" as possible, while keeping the total vocabulary (the size of the lookup table) manageable. That balance is exactly what real-world tokenizers are designed to hit.

---

## The reality: subword tokenization

Real tokenizers (the ones GPT, Claude, and similar models actually use) don't split on whole words like our toy example. They split on **subword pieces**, using an algorithm typically called **Byte-Pair Encoding (BPE)** or a close relative of it.

Here's the intuition, without the algorithm details:

```text
Common words → stay as ONE token
"the", "is", "cat", "dog"

Rare or long words → get split into pieces that ARE common
"unhappiness"  →  "un" + "happi" + "ness"
"tokenization" →  "token" + "ization"
```

The tokenizer was built by scanning a huge amount of text beforehand and asking: *"What chunks of characters show up together so often that they deserve their own single token?"* Frequent chunks (whole common words, common prefixes like `"un"`, common suffixes like `"ing"`) get merged into single tokens. Rare or unfamiliar words get broken down into smaller, more common pieces — down to individual letters if truly necessary. This is also why a tokenizer never truly gets "stuck" on a word it has never seen: worst case, it falls back to spelling it out piece by piece.

```text
Word never seen before?
        ↓
Break into smaller known pieces
        ↓
Worst case: fall back to individual characters
        ↓
There is always SOME valid tokenization
```

This is also a nice explanation for a real, observable quirk: LLMs are sometimes noticeably worse at character-level tasks (like counting letters in a word, or reversing a string) — because the model often never sees individual letters as separate tokens. It sees `"strawberry"` as one or two chunks, not as ten individual letters, so counting the letter `"r"` in it is genuinely harder for the model than it sounds.

---

## Special tokens

Alongside ordinary word-piece tokens, the vocabulary also reserves a handful of **special tokens** that don't correspond to any word at all — they're control signals:

```text
<bos>   → "beginning of sequence"
<eos>   → "end of sequence, stop generating"
<pad>   → "empty filler, ignore this position"
```

For example, generation keeps producing tokens in a loop (Chapter 1) until the model itself produces `<eos>`, or a maximum length is hit. We'll see more special tokens later (Chapter 13) when we look at how chat models separate "system," "user," and "assistant" turns — that separation is also implemented with special tokens, not magic.

---

## And decoding — the reverse direction

The tokenizer's job isn't only text → IDs. It also has to go the other way, turning the model's output back into readable text:

```text
Model predicts token ID 41
        ↓
Tokenizer looks up ID 41 in the vocabulary
        ↓
"slept"
        ↓
Appears on your screen
```

Same table, same lookup, just used in reverse. Nothing new to learn here — it's worth naming explicitly only so the word "tokenizer" doesn't feel like it does two unrelated jobs. It does one job (a fixed mapping between text pieces and numbers) used in both directions.

---

## The complete picture so far

```text
"The cat slept"
        │
        ▼
    Tokenizer  (chop into pieces using the vocabulary)
        │
        ▼
    [10, 25, 41]   ← just addresses, no meaning yet
        │
        ▼
   (Chapter 3 picks up here: Embeddings)
```

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "Token IDs encode meaning — bigger numbers mean something." | IDs are arbitrary row numbers in a lookup table. `25` and `26` could be totally unrelated words. |
| "Tokens are always whole words." | Real tokenizers mostly use subword pieces. Common words are one token; rare or long words get split into smaller, more frequent chunks. |
| "The tokenizer changes during training." | No — the vocabulary and the tokenizer's splitting rules are fixed *before* training starts, and never change afterward. Only the embedding table (Chapter 3) changes during training. |
| "A word the tokenizer has never seen breaks it." | It can't — worst case, an unfamiliar word gets spelled out as individual characters, which are always in the vocabulary. |

---

## Quick reference

```text
Tokenization = a fixed, two-way mapping between text pieces and integer IDs.

Text → Tokenizer → Token IDs   (encoding)
Token IDs → Tokenizer → Text   (decoding)

Real tokenizers split into SUBWORD pieces (via BPE or similar),
not whole words — common chunks get one token, rare words get split.

Token IDs carry NO meaning by themselves.
They are addresses. Meaning is added in the next step: Embeddings.
```

---

## The architecture so far

From this chapter on, each chapter adds one more piece to a single diagram of the whole model. By Chapter 7 you will have a complete Transformer block, and Chapter 8 stacks it into a full GPT. Boxes drawn with double lines are the new addition in the chapter you are reading. Boxes with single lines were added in earlier chapters, shown in compact form.

```text
             Raw text: "The cat slept"
                          │
                          ▼
╔══════════════════════════════════════════════════╗
║ TOKENIZER                                  ◄ NEW ║
║  text → token IDs (a fixed vocabulary)           ║
║  the same mapping is used to decode back to text ║
╚══════════════════════════════════════════════════╝
                          │
                          ▼
              Token IDs: [10, 25, 41]
```

Learned so far: nothing yet. The vocabulary is fixed before training starts.

---

## What's next

We now have a list of meaningless integers: `[10, 25, 41]`. Chapter 3 answers the question this naturally raises — how does an arbitrary ID like `25` turn into something that actually captures what "cat" *means*, and how does the model ever learn that in the first place?
