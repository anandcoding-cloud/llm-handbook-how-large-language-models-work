# Chapter 1 — The Big Picture: How a Large Language Model Predicts the Next Token

## Introduction

Large Language Models can feel like a black box — text goes in, coherent (sometimes brilliant) text comes out, and the mechanism in between is easy to wave away as "AI magic." It isn't magic. It's a long chain of genuinely simple ideas — counting, looking things up in tables, comparing vectors, weighted averages — stacked hundreds of layers deep and trained on an enormous amount of text.

This handbook exists to take that chain apart, one link at a time, and put it back together in your head so that "how does ChatGPT actually work?" stops being a mystery and starts being something you can explain, sketch, and reason about.

The approach throughout is **intuition first, then math, then engineering**:

1. **Intuition** — a real-world analogy for the concept, before any notation.
2. **Concept** — what the piece actually does, and why it exists at all.
3. **Mathematics** — the formulas, introduced only once the intuition is solid enough that the notation feels like a natural way to write down something you already understand.
4. **Engineering perspective** — how the idea actually shows up in real model code, tensor shapes, and production systems.

Underneath that sequence is one habit, and it is what makes this handbook different from a list of components: **every new piece enters the story because the previous one left a problem unsolved.** For each one we ask what problem we have, why what we have so far isn't enough, how the new piece solves it, and what it produces. In short: *why, then how, then what.* (The [Introduction](README.md) explains this approach in more detail.)

Every chapter builds directly on the ones before it, so it's worth reading in order the first time through. After that, treat it as a reference and jump straight to whichever chapter you need.

By the end, you should be able to explain, from memory, how a sentence you type turns into a reply — and, further in, how that same model gets trained, shrunk down to run on a laptop, and served to thousands of users at once.

Let's start with the single idea everything else in this handbook is built on top of.

---

## Where this handbook is going

Before we zoom into any single piece, it helps to see the whole shape of the journey:

```text
Chapter 1  →  The Big Picture            (you are here)

Build a GPT
Chapter 2  →  Tokenization
Chapter 3  →  Embeddings
Chapter 4  →  Attention
Chapter 5  →  Multi-Head Attention
Chapter 6  →  The Feed-Forward Network
Chapter 7  →  The Transformer Block
Chapter 8  →  Stacking Blocks into GPT

Train it
Chapter 9  →  Training I: Learning from Text and Measuring Error
Chapter 10 →  Training II: Gradients and Backpropagation
Chapter 11 →  Training III: Gradient Descent, AdamW and the Training Loop

Run it, adapt it, shrink it
Chapter 12 →  Inference (KV Cache, Sampling)
Chapter 13 →  Fine-tuning (SFT, LoRA, RLHF, DPO)
Chapter 14 →  Quantization

How modern models are built
Chapter 15 →  The Transformer Family (BERT, GPT, T5)
Chapter 16 →  Modern Building Blocks (RoPE, RMSNorm, SwiGLU)
Chapter 17 →  Attention Variants (MQA, GQA, MLA, sliding window, sparse, hybrid)
Chapter 18 →  Mixture of Experts (MoE)
Chapter 19 →  Multimodal Models

How models reason, and real models
Chapter 20 →  Reasoning Models
Chapter 21 →  Inside Modern Models (Qwen and DeepSeek)

Interfacing the model with its users
Chapter 22 →  Serving a Model (GGUF, llama.cpp, vLLM)
Chapter 23 →  Speed and Scale (Speculative Decoding, Parallelism, Hardware)

In production
Chapter 24 →  Production AI (agents, tool routing, memory, RAG)

Appendix
Appendix A →  TinyGPT by Hand (one full forward and backward pass with real numbers)
References →  Papers, documentation and the study material behind the handbook
```

Every chapter builds on the one before it. Nothing here is magic — it's a long chain of fairly simple ideas, stacked on top of each other until the result *looks* like magic.

---

## The one-sentence answer

> **A Large Language Model is, underneath everything, a very good "guess the next word" machine.**

That's it. That's the whole trick. Everything else in this handbook — attention, transformers, training, fine-tuning, quantization — exists purely to make that one guess as good as possible, as efficiently as possible.

---

## An analogy you already know

You've used this technology before, just a much weaker version of it: **predictive text on your phone keyboard.**

You type:

```text
"I'll see you at the"
```

and your phone suggests:

```text
"store"   "park"   "gym"
```

Your phone isn't *thinking*. It's just asking:

> "Given everything typed so far, what word is statistically most likely to come next?"

An LLM does exactly this — except instead of looking at the last 2-3 words like your keyboard, it looks at **thousands of words of context**, and instead of a crude frequency table, it uses a very deep, very trained neural network to make the guess. Scale that trick up enough, and the guesses start looking like reasoning, writing, and conversation.

---

## The complete pipeline (bird's-eye view)

Here is the entire journey a piece of text takes through an LLM, from the moment you hit "send" to the moment a reply starts appearing:

```text
Your text
   ↓
Tokenizer            (chop text into small pieces called tokens)
   ↓
Token IDs             (turn each piece into a number)
   ↓
Embedding Lookup       (turn each number into a vector of meaning)
   ↓
Positional Information (tell the model the order of the words)
   ↓
Transformer Block × N   (the "thinking" layers — this is most of the model)
   ↓
Contextual Embedding    (a final, context-aware vector for each token)
   ↓
LM Head                 (turn that vector into a score for every possible next word)
   ↓
Logits                  (raw scores)
   ↓
Softmax                 (turn scores into probabilities)
   ↓
Sampler                 (pick one word, according to those probabilities)
   ↓
Next token
```

Then — and this is the part people often miss —

```text
Next token
   ↓
Add it to the text
   ↓
Run the ENTIRE pipeline again
```

**An LLM doesn't write a sentence. It writes one token, then re-reads everything (including its own new token) and writes the next one. Over and over.** A whole paragraph is really just this loop running hundreds of times in a row, fast enough that it feels instant.

Every chapter after this one is really just a deep dive into **one box** in that diagram above. Tokenization is Chapter 2. Embeddings is Chapter 3. Attention (the biggest, most important box, hidden inside "Transformer Block") gets Chapters 4 and 5 all to itself. And so on.

---

## Walking through it once, with a real sentence

Suppose the model has seen:

```text
"The cat sat on the ___"
```

Roughly, here's what happens:

1. **Tokenizer** breaks this into pieces: `["The", "cat", "sat", "on", "the"]`
2. Each piece becomes a **token ID** — just an integer, like an index in a giant dictionary.
3. Each ID is looked up in an **embedding table** and turned into a vector — a long list of numbers that captures *meaning*.
4. The model adds **positional information** so it knows "cat" came second, not fifth.
5. These vectors pass through many **Transformer Blocks**, where each word gets to "look at" every other word and update its understanding based on context (this is **attention**, and it's the real star of the show).
6. After the last block, the model has one final vector representing "everything relevant about what comes next, given this whole sentence so far."
7. The **LM Head** turns that vector into a score for *every word in the vocabulary* — tens of thousands of candidates.
8. **Softmax** turns those scores into probabilities:

```text
"mat"      →  61%
"floor"    →  14%
"chair"    →   9%
"roof"     →   2%
...
```

9. A **sampler** picks one — usually the highest-probability option, sometimes a slightly less likely one on purpose (more on why in Chapter 12).
10. The chosen word — `"mat"` — is appended, and the *entire process runs again* to predict the word after that.

That's the whole engine. Every "intelligent-sounding" thing an LLM does is this loop, running over and over, on top of a model that has seen an enormous amount of text during training.

---

## Why does something this simple look intelligent?

Two reasons, and both matter:

**1. Scale.** A modern LLM isn't guessing based on a few rules — it has been trained on a huge fraction of the internet, and it has billions of internal parameters to encode subtle patterns of grammar, facts, reasoning style, and even code. "Guess the next word" done at this scale ends up requiring the model to implicitly learn grammar, facts, logic, and style — because all of those *help* it guess better.

**2. Context.** The model doesn't just look at the last word — it looks at everything in the conversation so far (up to its context limit), and it can relate *any* word to *any* other word through attention (Chapter 4). That's what lets it stay on topic, remember what you said three paragraphs ago, and follow instructions.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "The LLM looks up answers in a database." | No database lookup happens at inference time. Everything is generated fresh, one token at a time, from learned parameters. |
| "It plans the whole sentence before writing it." | Mostly no — it predicts one token at a time. (Reasoning models, Chapter 20, add a mechanism that gets closer to planning, but the base mechanism is still one-token-at-a-time.) |
| "Bigger context window means it remembers everything perfectly." | It only "sees" what's inside the context window for that pass — nothing more, nothing magically remembered outside it. |
| "It understands language the way humans do." | It has learned extremely rich *statistical* and *structural* patterns of language — which produces understanding-*like* behavior, but the underlying mechanism is prediction, not comprehension in the human sense. |

---

## Quick reference

```text
LLM = a next-token predictor, run in a loop.

Pipeline:
Text → Tokens → Token IDs → Embeddings → +Position →
Transformer Blocks → Contextual Embedding → LM Head →
Logits → Softmax → Sample → Next Token → repeat

Two things make it feel intelligent: SCALE + CONTEXT.
```

---

## Keep this loop in mind

keep this loop in the back of your mind every time you design something that sits around an LLM call — a tool router, a memory system, a planner. All of it is ultimately feeding tokens into, and reading tokens out of, exactly this loop. Nothing your framework does changes the loop itself; it changes what gets put into the context window before the loop runs, and what gets done with the tokens that come out. That framing will make a lot of later engineering chapters (especially Chapter 24) click faster.

---

## What's next

Step one of the pipeline above was "Tokenizer" — text gets chopped into small pieces before anything else happens. Chapter 2 goes deep on **why** we do this, how a tokenizer actually decides where to cut, and why token IDs are just arbitrary index numbers with no built-in meaning of their own.
