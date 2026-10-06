# Chapter 9 — Training I: Learning from Text and Measuring Error with Cross-Entropy Loss

## Introduction

Chapter 8 completed the forward pass: raw text goes in, a predicted next word comes out. But we ended on an uncomfortable fact. **Every matrix in that pipeline starts out random**: the embedding table, every `W_Q`, `W_K`, `W_V`, `W_O`, `W1`, `W2`, every LayerNorm `γ` and `β`, and the LM head. A model made of random numbers predicts random words.

Over this chapter and the next two, we explain how billions of random numbers gradually become useful. The reassuring part is that **nothing new gets added to the architecture.** The architecture is finished. Training is simply a process that improves every learned matrix, a little at a time. And the answer turns out to be a loop built from a handful of ideas:

```text
Training data  →  Loss  →  Gradients  →  Backpropagation  →  Gradient descent  →  AdamW
                                                                      (+ mini-batches and epochs)
```

We'll take them one at a time, following the same pattern as the rest of this handbook: intuition first, then a tiny numerical example, then the formula, then how real LLMs use it. This chapter covers the first two steps (the training data, and the loss). Chapter 10 covers gradients and backpropagation. Chapter 11 covers gradient descent, AdamW, and the full training loop.

---

## Part 1 — What does GPT actually learn from?

Take one ordinary sentence:

```text
The cat sat on the mat.
```

How many training examples does it produce? Most people say one. The real answer is **five**:

```text
Input                          Target (the word to predict)
────────────────────────────   ───────────
The                       →    cat
The cat                   →    sat
The cat sat               →    on
The cat sat on            →    the
The cat sat on the        →    mat
```

GPT doesn't read books the way a human does. It plays a fill-in-the-blank game, constantly:

```text
"The animal didn't cross the road because it was ______"   →   tired
"Paris is the capital of ______"                            →   France
```

...billions or trillions of times, over a huge amount of text. Nobody has to label anything by hand: the correct answer for every blank is simply the next word in the text. That is why raw internet text is usable training data.

### One forward pass, many examples at once

The five examples above look like they need five separate runs of the model: one run on `The`, another on `The cat`, another on `The cat sat`, and so on. In practice the model is run **only once** on the whole sentence, and all five examples are learned from that single run. Here is how.

**Step 1: feed in the whole sentence, and line it up against itself shifted by one word.**

```text
Words fed in:     The    cat    sat    on     the    mat
                   │      │      │     │      │      │
Each word's        ▼      ▼      ▼     ▼      ▼      ▼
output predicts:  next   next   next  next   next   (nothing
                  word   word   word  word   word    to predict:
                                                      no word after "mat")
Correct answers:  cat    sat    on    the    mat
```

Every word, after passing through all the Transformer blocks, ends up with its own final vector. That vector is used to predict **the word that comes right after it**. So the correct answers are just the same sentence slid over by one position. This is where the five examples come from: `The → cat`, `cat → sat`, `sat → on`, `on → the`, `the → mat`.

**Step 2: stop each word from seeing its own answer.** There is an obvious danger. Look at the word `cat`: its job is to predict `sat`, but `sat` is sitting right there in the input. If `cat` were allowed to look at `sat`, it could simply copy it, and the model would score perfectly without learning anything, like a student who can see the answer key.

This is exactly what the **causal mask** from Chapter 4 prevents. A word can only look at itself and the words *before* it, so:

```text
Word       What it can see          What it must predict
The        The                      cat
cat        The cat                  sat
sat        The cat sat              on
on         The cat sat on           the
the        The cat sat on the       mat
```

Read the middle column as exactly what you'd see at that point if you were writing the sentence from left to right. The word `sat` is in the input, but the mask hides it from `cat` (and from `The`), so it can't be copied.

**The payoff.** The five rows in that table are the same five examples from before, but they all happen inside **one** forward pass. The mask makes each row see only its own past, so the rows don't interfere with each other, and the model learns from all five at once.

The saving is large. Doing it the naive way (a separate run for each example, each with a longer input) would process 1 + 2 + 3 + 4 + 5 = 15 word-positions for this tiny sentence, and the gap grows quickly: for a 1,000-word passage it's about 500,000 word-positions versus 1,000. One pass, many lessons, is a big part of why training on enormous amounts of text is practical.

---

## Part 2 — Measuring how wrong the model is: the loss

### Step 1: the forward pass makes a prediction

Take the first example, with the correct answer `tired`. The forward pass gives:

| Word | Probability |
|---|---:|
| tired | 0.30 |
| hungry | 0.55 |
| asleep | 0.10 |
| running | 0.05 |

The model thinks `hungry` is more likely. It's wrong. But "wrong" isn't useful on its own: to improve, the model needs a **number** that says *how* wrong. Not a yes/no, but something like `2.31` or `0.17`. That number is called the **loss**.

Think of throwing darts. Hitting the bullseye is perfect. Missing by a centimeter is a small mistake. Missing by two meters is a huge one. The loss is the distance from the bullseye, and training is the process of shrinking it.

### Step 2: why not just use "1 minus the probability"?

The obvious idea is `loss = 1 − p`, where `p` is the probability given to the correct word. It sounds reasonable:

| Probability of the correct word | `1 − p` |
|---:|---:|
| 0.90 | 0.10 |
| 0.60 | 0.40 |
| 0.10 | 0.90 |
| 0.01 | 0.99 |

The problem shows up when you compare the last two rows. Going from `p = 0.10` to `p = 0.01` is a catastrophic drop (the model went from "unlikely" to "essentially certain it's wrong"), yet the loss only creeps from `0.90` to `0.99`. Think of three students: one who scores 99% instead of 100%, one who scores 60%, and one who scores 1%. The last student isn't just "a bit worse". They are **completely** wrong, and the punishment should reflect that. Simple subtraction doesn't punish catastrophic mistakes nearly strongly enough.

### Step 3: the logarithm

The fix is the loss GPT actually uses, called **cross-entropy**. For one prediction it is:

```text
Loss = −log(p)
```

where `p` is the probability the model gave to the **correct** word (using natural logarithms). Compare the two:

| Probability of the correct word | `1 − p` | `−log(p)` |
|---:|---:|---:|
| 0.99 | 0.01 | 0.010 |
| 0.90 | 0.10 | 0.105 |
| 0.60 | 0.40 | 0.511 |
| 0.30 | 0.70 | 1.204 |
| 0.10 | 0.90 | 2.303 |
| 0.01 | 0.99 | 4.605 |

The logarithm barely punishes a model that was nearly right, but punishes a confidently wrong one very heavily, and the penalty keeps growing as the probability approaches zero. When the model gave the correct word `p = 0.30`, the loss is `1.204`. If it had given `0.99`, the loss would be `0.010`, almost nothing.

### Why specifically the logarithm?

This isn't an arbitrary choice. It comes from information theory, and it's worth a short detour, because it explains what the loss really *means*. Claude Shannon asked: how do we measure **surprise**? Suppose I tell you "the sun will rise tomorrow." You learn almost nothing, because you expected it. If I tell you "a new element was discovered," you learn more, and "the speed of light changed yesterday" would be enormous news. So the amount of information in an event should be:

1. **Zero for a certain event** (probability 1).
2. **Larger the rarer the event is.**
3. **Additive for independent events.** The surprise of two independent things happening should be the surprise of the first *plus* the surprise of the second. But probabilities of independent events *multiply* (`½ × ½ = ¼`).

Property 3 is the clever one. We need a function `f` where `f(a × b) = f(a) + f(b)`. The logarithm is the function with exactly that property: `log(ab) = log(a) + log(b)`. So the natural measure of surprise is `−log(p)` (the minus sign just makes it positive, since `log` of a probability is negative).

So the loss is, literally, **how surprised the model was by the correct answer**. If it said `tired` was 99% likely and `tired` was right, it wasn't surprised, and the loss is tiny. If it said 1% and `tired` was right, it should be very surprised, and the loss is large. That's all cross-entropy measures.

### The full formula (and why it collapses)

For completeness, the general cross-entropy formula sums over the **whole vocabulary**:

```text
Loss = − Σ  yᵢ · log(pᵢ)          (sum over every word i in the vocabulary)
```

Here `pᵢ` is the model's probability for word `i`, and `yᵢ` is the truth, written as a **one-hot vector**: `1` for the correct word and `0` for every other word.

| Word | `yᵢ` (truth) | `pᵢ` (prediction) |
|---|---:|---:|
| tired | 1 | 0.70 |
| hungry | 0 | 0.20 |
| asleep | 0 | 0.07 |
| running | 0 | 0.03 |

Substituting: `−(1·log 0.70 + 0·log 0.20 + 0·log 0.07 + 0·log 0.03)`. Everything multiplied by zero disappears, leaving `−log(0.70)`. Even with a vocabulary of 200,000 words, 199,999 of the terms vanish and only one survives. So the simple `−log(p)` we've been using is not a different formula. It is the *same* formula, simplified for next-token prediction.

One thing worth being precise about: the loss depends only on the probability of the correct word, but the model still learns about the wrong words indirectly. Probabilities must add up to 1, so raising the correct word's probability necessarily lowers the others.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "The loss is a yes/no, right/wrong signal." | It's a number measuring surprise: `−log(p)` of the correct word. Tiny when the model was nearly right, large when it was confidently wrong. |

---

## Quick reference

```text
Training data:   every position predicts the next word (one forward pass, many examples,
                 thanks to the causal mask)
Loss:            L = −log(p_correct)      (full form: −Σ yᵢ log pᵢ, collapses via one-hot)
```

---

## What's next

The model now has a loss: a single number measuring how wrong its prediction was. Chapter 10 explains how that one number is turned into a separate instruction for every one of billions of weights, using gradients and backpropagation.
