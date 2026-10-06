# Chapter 13 — Fine-tuning: SFT, LoRA, RLHF and DPO, from Base Model to Assistant

## Introduction

Chapters 2 to 11 built a GPT-style model and trained it on raw text. Chapter 12 showed how to run it. But if you actually sat down with a freshly pre-trained model and typed a question, you would be surprised: it would not answer you. It would *continue* your text, as if your question were the first line of a document.

This chapter answers the question that follows naturally from everything so far:

> **How does one GPT become ChatGPT?**
> And: **how does one base model become a coding model, a legal model, or a medical model, without retraining from scratch every time?**

The answer is **fine-tuning**: continuing to train an already-trained model, on carefully chosen data, with a carefully chosen definition of "good". The chapter follows the usual pipeline in order:

1. **Base model vs assistant**: what is missing after pre-training.
2. **Supervised fine-tuning (SFT)**: teaching the model to follow instructions, including the one trick that makes it work, the **loss mask**.
3. **LoRA and QLoRA**: how to fine-tune a huge model without updating all of its billions of weights.
4. **RLHF and DPO**: how a model learns what humans *prefer*, not just what is "correct".

A reassuring thing to know before we start: **almost nothing from the earlier chapters changes.** Same Transformer, same backpropagation, same AdamW. What changes is the *data* and the *definition of a good answer*.

---

## Part 1 — The base model: all knowledge, no manners

### What a base model is

Suppose a lab trains a large model on Wikipedia, books, Stack Overflow, GitHub, news and research papers. After months of training on thousands of GPUs, it becomes a **base model**, something like `Llama-Base`.

Can you now ask it:

```text
Write me Python code.
Summarize this article.
Explain quantum physics simply.
```

You might think: of course. In practice: **not very well.**

Remember what pre-training taught it (Chapter 9). The objective was only:

```text
Predict the next token.
```

The model never learned to answer questions, follow instructions, or be helpful. It only learned *language*, and along the way, a great deal of knowledge about the world.

### What it does with your prompt

Give a base model this prompt:

```text
Translate to French:

Good morning
```

It may well continue like this:

```text
Translate to French:

Good morning

Examples:

Good night...

Good evening...
```

Why? Because it is **just continuing text**. It does not realize you are *asking it to do something*. It doesn't understand "the user is giving me an instruction". It only understands "this kind of text usually continues like this".

An analogy: a child can spend years reading, listening and watching the world. That doesn't automatically mean they know how to sit an exam, teach someone, or write software. Those are *additional skills* built on top of what they know.

```text
Base model  =  enormous knowledge
               + no personality
               + no conversational behaviour
               + no instruction-following
```

### The pipeline that fixes this

ChatGPT did not throw GPT away, and it didn't retrain everything from scratch. It added further training stages on top of the same model:

```text
Stage 1                       Stage 2                      Stage 3
-------                       -------                      -------
Internet text                 Instruction examples         Human preferences
     ↓                             ↓                            ↓
Predict the next token        Fine-tuning (SFT)            Alignment (RLHF or DPO)
     ↓                             ↓                            ↓
BASE MODEL            →       ASSISTANT            →       CHAT MODEL
```

Think of university. Pre-training is earning the degree. Instruction tuning is learning how to teach. Preference alignment is learning to communicate helpfully and politely. **You are still the same person**, but you have acquired new behaviours.

This explains model names. `Llama-3-8B-Base` and `Llama-3-8B-Instruct` have the same architecture and the same number of parameters. They differ only in which training stages they have been through.

```text
Base Model
    │
    ├── continue training on instruction data
    │
    ▼
Instruct Model
```

### One of the biggest misconceptions

Many people think:

> "ChatGPT must be a different architecture."

It isn't. It is still a Transformer. Throughout all these stages:

- the tokenizer stays the same,
- the Transformer architecture stays the same,
- the attention mechanism stays the same,
- the LM head stays the same.

We are not redesigning the engine. We are **changing its behaviour by continuing training on carefully chosen data.**

### Why not mix instructions into pre-training?

A fair question. Why not throw books, code, Wikipedia, conversations and instruction datasets into one giant pile and train once?

The model would still learn some instruction-following, but much more weakly. Researchers found that **separating the stages works much better**: first teach the model *language*, then teach it *how to use that language helpfully*.

### Why specialize one model instead of training many?

Imagine you run the lab. You have already spent millions training the base model, and now you want a medical assistant, a legal assistant and a coding assistant. Two options:

```text
Option A: train three models from scratch
          each needs ~15 trillion tokens, thousands of GPUs,
          months of time, tens or hundreds of millions of dollars.

Option B: train the base model once, then specialize copies of it

          Base Model
          ├── Medical
          ├── Legal
          ├── Coding
          ├── Finance
          └── Math
```

Option B is not just nicer. It is the only economically viable choice. All the specialists already know English, grammar, programming syntax, mathematics and general world knowledge. The only thing left to teach is **behaviour and specialization**.

Humans work the same way. After a computer science degree you become a software engineer, a data scientist, or a researcher. Nobody sends you back to kindergarten.

---

## Part 2 — What "fine-tuning" really means

### It is not "train only the last layer"

When beginners hear *fine-tuning*, they often imagine training only the last layer, or only the LM head. Traditional fine-tuning actually means:

> **Continue training the entire model.**

Every learned parameter from the tables in the earlier chapters keeps learning:

```text
Embeddings            ✅
W_Q, W_K, W_V         ✅
W_O                   ✅
FFN (W1, W2)          ✅
LayerNorm γ and β     ✅
LM Head               ✅
```

For a 70-billion-parameter model, that means updating **70 billion numbers** on every step. It also creates a storage problem: a medical version is another ~70 GB (more, depending on precision), then a legal version is another 70 GB, then finance, and so on. Storage explodes.

That is exactly the problem LoRA (Part 5) will solve. But first we need to know what data we are fine-tuning on, and how the loss is calculated.

### The data: instruction → answer

For an assistant, the fine-tuning data is not books or Wikipedia. It looks like this:

```text
User:
What is Python?

Assistant:
Python is a programming language...
```

```text
User:
Translate to French:
Good morning

Assistant:
Bonjour.
```

```text
User:
Write a bubble sort in C++

Assistant:
...
```

The task is no longer "predict the next token from the internet". It is **"given an instruction, produce a helpful answer."**

### But the training mechanics are unchanged

Here is the elegant part. If the objective changed, are we no longer doing next-token prediction?

**We still are.** The loss is still cross-entropy on the next token. The only difference is *what text we feed in*. We simply glue the conversation into one sequence:

```text
<User>
What is Python?

<Assistant>
Python is...
```

and the model keeps predicting the next token as before:

```text
P → y → t → h → o → n ...
```

(In real systems the `<User>` and `<Assistant>` markers are special tokens defined by the model's **chat template**, and different model families use different ones. The idea is identical.)

> **Instead of changing the algorithm, we changed the examples.**

It's like teaching a pianist jazz. You don't redesign their hands; you expose them to jazz and let them practice. The learning mechanism stays exactly the same. That is why the entire training pipeline from Chapters 9 to 11 can be reused.

---

## Part 3 — The loss mask: reading everything, being graded only on the answer

Here is the one genuinely new trick in supervised fine-tuning. Take this training example:

```text
User:
What is the capital of France?

Assistant:
Paris.
```

During training, on which tokens should the model compute loss?

- **Option A:** the whole sequence, including the user's question.
- **Option B:** only the assistant's answer.

Option A feels natural: it's what pre-training did, where every token contributed. (It is also the answer most people give the first time. It's a very reasonable guess.) The correct answer is:

> **Option B: only the assistant's response contributes to the loss.**

### Why

In pre-training, every token has a label and every token counts:

```text
The → cat
cat → sat
sat → on
on  → the
...
```

That makes sense, because the model is learning language itself.

In instruction tuning, should we teach the model to *generate* `What is the capital of France?` No. The user already wrote that. We don't want the model to learn to *ask the question*. We want it to learn to **answer it**.

An exam is a good comparison. The teacher writes `2 + 2 =`, you write `4`. Your grade should not depend on whether the teacher wrote the question correctly. You are graded **only on your answer**.

### How it's implemented

The entire conversation is still fed through the Transformer, because the model needs the question as **context**. But the loss is only computed on the assistant's tokens:

```text
User question  →  Assistant: Paris
                              ^^^^^
                              loss
```

Internally every token has a label. Here is the same example, token by token:

| Token | Used as context? | Contributes to loss? |
|---|:---:|:---:|
| `<User>` | ✅ | ❌ |
| What | ✅ | ❌ |
| is | ✅ | ❌ |
| the | ✅ | ❌ |
| capital | ✅ | ❌ |
| of | ✅ | ❌ |
| France | ✅ | ❌ |
| ? | ✅ | ❌ |
| `<Assistant>` | ✅ | ❌ |
| **Paris** | ✅ | **✅** |
| **.** | ✅ | **✅** |

In PyTorch, the labels for the prompt positions are commonly set to a special value, usually `-100`, which tells the loss function: **"ignore this position."** The model still runs the forward pass over the whole sequence, so it understands the prompt, but the gradients come **only from the assistant's tokens**.

Still next-token prediction. Suppose the assistant says `Paris is the capital of France.` The model is trained on:

```text
Paris   → is
is      → the
the     → capital
capital → of
of      → France
```

exactly as in pre-training. Nothing changed except **which tokens receive loss.**

### A worked example with a matrix view

Take the instruction `What is AI?` with the answer `Artificial Intelligence.` The model receives one 9-token sequence:

| Position | Token |
|---:|---|
| 1 | `<User>` |
| 2 | What |
| 3 | is |
| 4 | AI |
| 5 | ? |
| 6 | `<Assistant>` |
| 7 | Artificial |
| 8 | Intelligence |
| 9 | . |

**Step 1: the forward pass is unchanged.** Every token gets an embedding, attention, FFN, a contextual embedding, the LM head and logits. So the model predicts the next token at *every* position:

| Input token | Predicts |
|---|---|
| `<User>` | What |
| What | is |
| is | AI |
| AI | ? |
| ? | `<Assistant>` |
| `<Assistant>` | Artificial |
| Artificial | Intelligence |
| Intelligence | . |

**Step 2: the loss mask decides which predictions count.**

```text
Position:     1  2  3  4  5  6  7  8  9

Logits:      L1 L2 L3 L4 L5 L6 L7 L8 L9     (every position produces them)

Loss mask:    X  X  X  X  X  X  ✓  ✓  ✓

              X = ignore        ✓ = compute cross-entropy
```

So the total loss is:

```text
0 + 0 + 0 + 0 + 0 + 0 + Loss7 + Loss8 + Loss9
```

Only the assistant's tokens contribute.

### A quick reminder: what cross-entropy measures

Since the loss mask decides *where* cross-entropy is applied, it's worth refreshing *what* cross-entropy is (introduced in Chapter 9).

> **Cross-entropy measures how wrong the model's probability was for the correct next token.** Equivalently: it punishes the model for not being confident about the right answer.

```text
Loss = −log( P_correct )
```

The model only cares about one number: *what probability did I give the correct token?*

| Probability given to the correct token | Loss |
|---:|---:|
| 99% | very small (≈ 0.01) |
| 90% | small (≈ 0.105) |
| 50% | medium (≈ 0.693) |
| 10% | large (≈ 2.30) |
| 1% | very large (≈ 4.60) |

Two examples with the correct answer `Paris`:

| Token | Model A | Model B |
|---|---:|---:|
| Paris | **90%** | **10%** |
| London | 5% | 60% |
| Berlin | 3% | 20% |
| Rome | 2% | 10% |
| Loss | low | high |

Why a logarithm? A simpler loss like `1 − probability` would barely distinguish 10% from 1%. But being 99% sure of the *wrong* answer is far worse than being slightly unsure. The logarithm amplifies confident mistakes, which pushes the model to be both accurate and well-calibrated. Think of it as the model's **report card** after every prediction.

In SFT, this loss is computed on the assistant's tokens only, averaged (or summed) over them.

### Two masks, two jobs

You now have two different "masks" in your mental model. They are easy to confuse, but they do completely different jobs.

| | **Attention mask (causal mask)** | **Loss mask** |
|---|---|---|
| Where | **Inside attention**, between Q·Kᵀ and softmax | **After the LM head**, at the cross-entropy step |
| Purpose | Stop a token from seeing the *future* | Stop the prompt from being a training *target* |
| Exists in | Pre-training, SFT, and inference | Only where we choose to ignore some tokens (SFT) |
| Concerns | What the model can *read* | What the model is *graded on* |

```text
Attention mask:    Q·Kᵀ  →  Mask  →  Softmax              (Chapter 4)

Loss mask:         Logits  →  Cross-entropy  →  ignore prompt positions  →  Backprop
```

A quick refresher on the causal mask (Chapter 4). For the three tokens `I love AI`, before masking, each token could attend to every other token, including future ones, which would be cheating. After masking:

| Predicting ↓ / Can attend → | I | love | AI |
|---|:---:|:---:|:---:|
| I | ✓ | ✗ | ✗ |
| love | ✓ | ✓ | ✗ |
| AI | ✓ | ✓ | ✓ |

Put together for instruction tuning:

```text
Prompt tokens
    ├── used for attention   ✅   (the answer depends on them)
    └── used for loss        ❌   (we don't want to teach the model to write the question)

Assistant response tokens
    ├── used for attention   ✅
    └── used for loss        ✅
```

The model **reads everything** but is **graded only on the answer**, like a student who reads the whole exam paper but is marked only on what they write.

> **The practical takeaway:** instruction tuning didn't invent a new training algorithm. Researchers realized they could reuse the entire pre-training pipeline and just *mask the labels for the prompt*.

---

## Part 4 — Supervised fine-tuning (SFT), end to end

**Supervised fine-tuning** sounds complicated. It simply means:

> **We have examples of good question–answer pairs, and we train the model to imitate them.**

It's called *supervised* because we already know the correct answer for each question, written by humans. (In pre-training the "supervision" is automatic: the next token in the text is the label.)

Follow one example through the whole pipeline, just as we did with the training loop in Chapters 9 to 11:

```text
 1. Start with the base model      (already knows English, code, science, history, maths)
 2. Build an instruction dataset   (thousands to millions of  User → Assistant  pairs)
 3. Tokenize                       (<User> What is AI ? <Assistant> Artificial Intelligence .)
 4. Embeddings                     (exactly as before)
 5. Transformer blocks             (exactly as before)
 6. LM head → logits               (exactly as before)
 7. Softmax → probabilities        (exactly as before)
 8. Cross-entropy, WITH the loss mask   ← the one new thing
 9. Backpropagation                (gradients for every parameter)
10. AdamW update, repeat           (over many examples)
```

Steps 3 to 7 are identical to pre-training. Step 8 is where SFT differs. Step 9 is **traditional fine-tuning**: every parameter, from the embedding matrix to the LM head, receives a gradient and gets updated.

After enough examples, the model notices a pattern:

```text
Translate...   →  Bonjour
Summarize...   →  a summary
Write Python.. →  def ...
```

and it learns something nobody explicitly programmed:

```text
When I see  <User>  I should generate  <Assistant>.
```

We never told the model to "become an assistant". We showed it many examples of *instruction → helpful answer*, and gradient descent did the rest.

### Pre-training vs SFT

| Pre-training | Supervised fine-tuning |
|---|---|
| Internet text | Instruction–response pairs |
| Learn language | Learn to follow instructions |
| Predict **every** next token | Predict **only the assistant's response** |
| Massive datasets | Much smaller, curated datasets |
| Builds general knowledge | Builds helpful behaviour |

So the only two things that changed are:

1. **The dataset** (instruction–response pairs instead of raw text).
2. **The loss mask** (only the assistant's response counts).

A surprisingly small change that produces a dramatically more useful model.

### How long do we fine-tune?

In Chapter 11 we saw three ways to decide when training stops: a fixed number of epochs, early stopping on validation loss, or a fixed token budget. Fine-tuning uses the same three tools, but which one matters most changes, because the situation is different:

| | Pre-training | Fine-tuning |
|---|---|---|
| Data size | trillions of tokens | thousands to a few hundred thousand examples |
| Main risk | not enough training | **overfitting**: memorising the examples |
| Typical budget | a token count (e.g. "10 trillion tokens") | **1 to 3 epochs** (full passes) |
| Main safeguard | scale, a fixed token budget | validation loss, checkpoints, a small learning rate |

Pre-training data is so large that the model sees almost every text only once, so memorisation barely happens, and the budget is simply "how many tokens can we afford?" Fine-tuning data is tiny by comparison. A big model can memorise 10,000 examples in a few passes, so the question flips from "how much more can it learn?" to "**when does it start to memorise?**"

**1. A small, fixed number of epochs.** Because fine-tuning sets are small, an epoch is cheap, and 1 to 3 epochs is the usual range. For example:

```text
10,000 examples, batch size 16   →   625 steps per epoch
3 epochs                         →   about 1,900 steps total
```

The whole run takes minutes to hours, not months.

**2. Validation loss, the key signal.** Hold back a slice of the data (say 5 to 10%) that the model never trains on. Every epoch, or every few hundred steps, compute the loss on it too (the same masked cross-entropy as in training):

```text
loss
 │
 │ ╲
 │  ╲___            training loss keeps falling
 │      ╲____
 │           ╲______
 │                  ╲________
 │
 │ ╲
 │  ╲___                         validation loss bottoms out…
 │      ╲___
 │          ╲___╱‾‾‾‾‾‾‾‾‾  ← …then rises: the model is now memorising
 └───────────────┬────────────── steps
              best checkpoint
```

When validation loss stops improving or starts to rise, the model is memorising your examples rather than learning the behaviour. You stop there, or you save checkpoints along the way and **pick the best one afterwards**. This is the early stopping of Chapter 11, and in fine-tuning it is the main control.

**3. A token budget** is rarely used for fine-tuning, because the dataset is small enough that "epochs" is the natural unit.

Three further things make fine-tuning stop earlier than you might expect:

- **A smaller learning rate than pre-training.** You are nudging a model that already works, so the steps are small. A large learning rate can damage what the model already knows.
- **Catastrophic forgetting.** The longer and harder you fine-tune on a narrow dataset, the more the model can lose its general abilities: it gets great at your task and worse at everything else. This is another reason to stop early, and it is also a reason LoRA is attractive, since the base weights are frozen and can't be overwritten.
- **Loss is not quality.** A lower loss means a better next-token fit on your examples, not necessarily better answers. So people also try a handful of real prompts, or a task benchmark, at each checkpoint, and sometimes choose a checkpoint with slightly *worse* loss because it behaves better.

**LoRA and QLoRA** (Parts 5 and 6) follow exactly the same rules, since they are SFT with fewer trainable weights. A small adapter is somewhat harder to overfit with, but you still watch validation loss and compare checkpoints. The preference stages work a little differently, and are covered at the end of Parts 7 and 8.

> **In one line:** pre-training stops when the token budget is spent. Fine-tuning stops when validation loss stops improving (usually within 1 to 3 epochs), because the danger is memorising and forgetting, not under-training.

### The cost problem

But look at step 9 again. For a 70-billion-parameter model, *every step* updates 70 billion parameters. That means huge GPU memory, huge storage, huge compute cost. Researchers looked at this and asked:

> **Do we really need to update all 70 billion parameters?**

That question gave birth to LoRA.

---

## Part 5 — LoRA: fine-tuning by learning a small correction

### The idea in one line

Suppose a layer computes `Y = XW`, where `W` is `4096 × 4096`, about **16.7 million weights**. Traditional fine-tuning changes every one of them.

LoRA says: **leave `W` completely frozen, and learn a small correction `ΔW` on the side.**

```text
Before:   Y = X W
LoRA:     Y = X (W + ΔW)
```

That's literally the core idea. The rest of LoRA is about making `ΔW` cheap.

### A tiny numeric example

Take a very small layer with input `X = [2 5]` and original weights:

```text
W = [1  2]
    [3  4]

Y = X·W = [2 5] × [1 2; 3 4] = [17  24]
```

**Traditional fine-tuning** changes `W` itself, say to `[1.1 2.2; 3.1 4.2]`, so the output becomes `[17.7 25.4]`. The behaviour changed.

**LoRA** never touches `W`. It learns a correction instead. Suppose the correction is:

```text
ΔW = [0.2   0.1]        W + ΔW = [1.2  2.1]
     [0.0  −0.1]                 [3.0  3.9]

Y = X·(W + ΔW) = [17.4  23.7]
```

The output changed, and `W` was never modified. The model doesn't care *why* a matrix changed. It just multiplies `X` by whatever matrix it is given.

### Why not learn ΔW directly?

If `ΔW` is also `4096 × 4096`, we have 16.7 million trainable values again. We've gained nothing. So the real LoRA trick is to describe `ΔW` with something much smaller.

Think of a photograph. A 4000 × 4000 image has 16 million pixels, but does it contain 16 million *independent* pieces of information? Usually not: sky, grass, road and trees are large similar regions. JPEG exploits that structure to store a far smaller file of an almost identical image. LoRA does something similar to the update matrix.

### Matrix factorization: ΔW = B·A

Instead of storing `ΔW`, LoRA writes it as the product of two thin matrices:

```text
ΔW  =  B · A

A :  r × 4096        (compresses 4096 numbers down to r)
B :  4096 × r        (expands r numbers back up to 4096)

with the rank  r = 8  as a typical choice
```

Multiply them and you get `4096 × 4096`, exactly the shape of `ΔW`.

Now count the parameters:

```text
Learning ΔW directly:       4096 × 4096              ≈ 16,777,216
LoRA with r = 8:            4096 × 8  +  8 × 4096    =     65,536

                            ≈ 256× fewer, for this one layer
```

> **A note on conventions.** Throughout this handbook we've written layers as `Y = XW`, with the input as a row vector. Research papers (and the LoRA paper) usually write `y = Wx` with a column vector, which is why the product is written `BA`, with `B` as the expanding matrix (`d_out × r`) and `A` as the compressing one (`r × d_in`). In our row convention the same two matrices appear in the opposite order. The roles don't change: **`A` compresses, `B` expands.** We use the papers' naming because that's what you'll meet in libraries and checkpoints.

### Why does this work? "Low rank", explained

Why can two tiny matrices stand in for a giant one? Because of what the update actually looks like.

Some patterns are simple, and some are complex. Drawing a straight line takes two numbers: slope and intercept. Drawing a perfect circle takes three: centre and radius. Drawing random television static, or an ink splash, takes thousands or millions. Researchers discovered that the changes needed to turn a general model into a medical model are often closer to the circle than to the static.

The reasoning is natural. The model already knows English, programming, mathematics and reasoning. Teaching it medicine is **not rewriting English**. You are mostly teaching new associations, new terminology and new behaviours. The required update is much simpler than the original knowledge stored in `W`. It's like editing a 1000-page book: you make a few hundred edits, you don't rewrite it. **LoRA assumes that fine-tuning is mostly editing, not rewriting.**

That's what **low rank** actually means. It does *not* mean "a small matrix". It means:

> **The update has much lower complexity than the original matrix.**

The **rank** of a matrix is the number of independent directions (independent pieces of information) it can represent. A rank-4096 update can be arbitrarily complicated. A rank-8 update must be built from just 8 independent directions.

### Why *two* matrices? The bottleneck

Why not just learn one small matrix directly? Because the update has to be the full `4096 × 4096` size so it can be added to `W`. The factorization is what lets us get a full-sized update from few parameters. And it works through a **bottleneck**:

```text
Input vector                      4096 numbers
      │
      ▼   A  (compress)
Bottleneck                        8 numbers
      │
      ▼   B  (expand)
Correction                        4096 numbers
```

Everything the correction does has to pass through those 8 dimensions. The model cannot invent an arbitrarily complex correction. That bottleneck *is* what enforces a low-rank update.

You can see the bottleneck in a toy example with rank **1**. Keep `X = [2 5]` and `W` as before, and let the correction be built through a single number:

```text
compress:  X → one number         [2 5] · [0.1; 0.2]  =  1.2
expand:    that number → 2 values  1.2 × [2  1]       =  [2.4  1.2]

Y = X·W  +  correction  =  [17 24] + [2.4 1.2]  =  [19.4  25.2]
```

The full matrix this builds is `ΔW = [0.2 0.1; 0.4 0.2]`. On a 2 × 2 matrix there is no saving (4 entries, described by 2 + 2 = 4 numbers), but the structure is the point: every row of `ΔW` is a multiple of the same direction `[2 1]`. On a 4096 × 4096 matrix, the same trick describes 16.7 million entries with only 65,536 numbers at rank 8.

```text
Without LoRA:   all 4096 dimensions are free to change   □□□□□□□□□□□□□□□□
With LoRA:      changes pass through an 8-dimensional bottleneck   ■■
```

And with the LoRA update in place, attention still works exactly as before:

```text
Q = X · (W_Q + B·A)
```

Softmax still works. The KV cache still works. Everything you've learned stays the same. Only the *effective* weight matrix changes slightly.

### A frequent misconception

> "LoRA changes only 1% of the neurons."

No. Remember `Y = X(W + BA)`. Every input still multiplies the whole effective weight matrix, and `BA` is a full-size matrix, so **every output changes**. The adapter is small, but its influence is network-wide.

An analogy: an orchestra. Traditional fine-tuning replaces every musician. LoRA keeps the orchestra exactly the same, and gives the conductor slightly different instructions. The orchestra plays differently even though almost nobody changed.

Another way to see it: the engine is already excellent, so instead of replacing it, install a small turbocharger.

### Which matrices get an adapter?

A Transformer block has many learned matrices: `W_Q`, `W_K`, `W_V`, `W_O`, and the FFN's `W1` and `W2` (plus the embeddings and the LM head). Where should LoRA attach? Think like an engineer. The options:

```text
Option 1:  attach LoRA to every matrix     → maximum flexibility, but more parameters,
                                              more GPU memory, slower training
Option 2:  attach LoRA nowhere             → fast, but the model learns nothing
Option 3:  find the important matrices     → this is what researchers did
```

A good instinct is "every *critical* parameter", and that is almost exactly how the field evolved. The reasoning: attention is where the model decides *where to look*. `W_Q`, `W_K` and `W_V` determine what each token asks, what counts as relevant, and what gets retrieved. Change them and the model begins attending differently. The FFN is more like internal processing after attention has gathered the information. It matters, but attention usually has more influence on behaviour.

Researchers experimented: only `W_Q` worked surprisingly well; `W_Q + W_V` was better; `W_Q + W_K + W_V` was better still for some tasks; and eventually people tried everything. The lesson: **more isn't always better.** Some matrices contribute far more than others, so common configurations target a subset, often `W_Q` and `W_V`, sometimes also `W_K`, `W_O` and the FFN layers. It depends on the model and the task.

It's like tuning a car: you don't replace every bolt, you modify the parts with the biggest effect. In code, this is why frameworks ask you for something like:

```python
target_modules = ["q_proj", "v_proj"]
# or
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

Those names map directly onto the matrices you've been studying.

### How does LoRA "generate" the adapter?

A common question, and the answer is simpler than people expect:

> **Nothing generates the adapter.** There is no adapter-generator and no second model. `A` and `B` are learned by gradient descent, exactly like every other weight.

Before training, `A` and `B` hold starting values. (In the standard recipe, `A` starts with small random numbers and `B` starts at **zero**. That makes `BA = 0` at the beginning, so the model begins *exactly* equal to the base model, and drifts away from it only as training pushes it.) Then training is the loop you already know:

```text
Forward pass          using the effective weight  W + B·A
      ↓
Cross-entropy         (with the loss mask)
      ↓
Backpropagation       gradients flow through the whole network
      ↓
AdamW                 updates ONLY A and B
      ↓
Repeat                millions of times
```

Walk through one step. The example is `Translate Hello to French → Bonjour`. The model, using `W + BA`, predicts `Bonjor`. The loss is high. Backpropagation starts. Which parameters are trainable? `W`: **no**, frozen, never updated. `A` and `B`: **yes**. AdamW nudges `A` and `B` slightly, so `BA` is now slightly better. Repeat across the dataset, and `A` and `B` *become* the adapter. It emerges through backpropagation, like every learned matrix in this handbook.

Two small details worth knowing:

- **Gradients still flow *through* the frozen `W`.** An adapter in layer 1 needs error signals that travelled back through all the later layers. What is skipped is *computing and storing a gradient for `W` itself*, and the AdamW state (momentum and variance, Chapter 11) that would go with it. That is where most of the memory saving comes from.
- LoRA implementations usually multiply the correction by a scale factor (`α / r`), so the strength of the adapter can be tuned independently of its rank.

### Is it the same adapter for every block?

No. **Every layer gets its own LoRA matrices.** If the model has 32 blocks and LoRA is attached to `W_Q` and `W_V`:

```text
Block 1:   W_Q₁ → (A_q₁, B_q₁)     W_V₁ → (A_v₁, B_v₁)
Block 2:   W_Q₂ → (A_q₂, B_q₂)     W_V₂ → (A_v₂, B_v₂)
...
Block 32:  W_Q₃₂ → (A_q₃₂, B_q₃₂)  W_V₃₂ → (A_v₃₂, B_v₃₂)
```

That is natural: different blocks learn different things (early blocks tend to capture basic syntax, later ones relationships and longer-range structure; each block refines the contextual embeddings further, see Chapters 7 and 8), so their corrections must differ. A LoRA file is therefore a collection of hundreds of small matrices, one pair per targeted matrix per block. That's why LoRA files are tens or hundreds of megabytes, not kilobytes.

### At inference time

Nothing is "generated" at inference either. The adapter is already trained, and the engine simply uses:

```text
W_effective = W + B·A       (for every matrix with an adapter)
```

then runs the normal forward pass. The Transformer doesn't know whether a weight came from the original model, from full fine-tuning, or from LoRA. It just multiplies matrices. You can either keep `W` and `BA` separate (so you can swap adapters in seconds), or **merge** them once into a single matrix so there is zero extra cost at run time.

### Why LoRA changed the field

Before LoRA, specializing a large model meant massive GPUs, huge storage and expensive deployment. After LoRA you distribute *only the adapter*:

```text
Base model          70 GB
+ Medical adapter  100 MB
+ Legal adapter    100 MB
+ Finance adapter  100 MB
```

One base model and many tiny adapters, instead of four full 70 GB copies. This is also what you're downloading when you pull something like `medical-lora` from Hugging Face: not another model, just the `BA` matrices. The framework loads *base model + adapter → effective model*. That is why LoRA became the standard in the open-source community.

It also raises an interesting question: can you load a medical adapter *and* a finance adapter together, or merge them into one model? **Sometimes yes, sometimes spectacularly no**, because both adapters push on the same weight matrices, and their updates can interfere. This is an active research area (adapter composition and merging).

### The pattern behind modern LLM engineering

Look at what each technique keeps fixed:

| Technique | What stays fixed | What changes |
|---|---|---|
| Transformer | the architecture | the learned weights |
| Instruction tuning | the architecture | the training data |
| LoRA | the base weights | a tiny adapter |
| Inference | the model | the decoding strategy |

Modern LLM engineering is surprisingly conservative. Instead of constantly inventing new architectures, big improvements often come from changing **how** we train, adapt, or use the same underlying Transformer.

---

## Part 6 — QLoRA: LoRA on a compressed model

LoRA shrinks the number of **trainable** parameters. But the frozen base model still has to sit in GPU memory while you train, and a 70-billion-parameter model in 16-bit numbers is roughly 140 GB. That is still too big for most hardware.

**QLoRA** attacks that remaining cost by combining two ideas you already have, or will have soon:

1. **Quantize the frozen base model** to 4-bit numbers (Chapter 14 explains number formats and quantization). The base weights are never updated, so they can be stored very compactly: roughly a quarter of the 16-bit size.
2. **Train LoRA adapters on top**, kept in higher precision (16-bit), since those are the only numbers that actually change.

```text
Frozen base model     stored in 4-bit   (small)
        │
        ▼   each weight is converted back to 16-bit on the fly,
        │   only for the moment it is being multiplied
        ▼
Forward pass   Y = X · (W_4bit→16bit  +  B·A)
        │
        ▼
Backprop       updates only A and B  (in 16-bit)
```

The adapter doesn't care that the base weights were compressed. It learns to correct for whatever the compressed model produces. The result is that very large models can be fine-tuned on a single consumer-grade or workstation GPU, with quality close to ordinary LoRA. (The QLoRA paper also introduces a 4-bit format designed for the bell-shaped distribution of weights, and tricks to save optimizer memory. The core idea stays the one above.)

> **Remember:** LoRA saves memory on the *training* side (few trainable parameters, tiny optimizer state). QLoRA additionally saves memory on the *frozen model* side. Neither changes how the adapted model works afterwards.

---

## Part 7 — RLHF: teaching a model what humans *prefer*

### Where SFT runs out

After SFT, our pipeline looks like this:

```text
Internet → Pre-training → Base Model → Instruction data → SFT → Instruction Model
```

Is that enough? Ask `Tell me a joke.` and the SFT model replies `Why did the chicken...`. Good. But now ask `How do I build a bomb?`, where there isn't one obviously correct response. Or ask `Write me an email declining a job offer.` There are perhaps 100 excellent answers. Which one should the model learn?

Remember how SFT works: **one question, one answer**, and cross-entropy trains the model to imitate that exact answer. But for `Write a professional email`, response A (`Good morning...`) and response B (`Hello...`) can both be perfectly fine. SFT cannot easily express *"both are acceptable, but A is slightly better."*

### Humans think in "better / worse", not "correct / wrong"

Rather than label answers as correct or wrong, it's far more natural for a human to compare two:

```text
Prompt:  Summarize this article.

Answer A:  clear, concise, accurate
Answer B:  too long, repeats itself, less clear

Human picks:  A
```

That's all. No gradients, no token labels. Just a **preference**. Collect thousands (or millions) of these and you have a **preference dataset**:

```text
Prompt → Answer A → Answer B → human picks the better one
```

### The Reward Model: a teacher that never sleeps

The clever idea: instead of training the LLM directly on these comparisons, train **another model** that predicts how good an answer is:

```text
Prompt + Answer   →   Reward Model   →   a score (one number)
```

It's a teacher. The LLM writes an essay, and instead of a human grading every essay, the reward model says `9/10` or `3/10`. Much faster.

How is the reward model trained? From the preference data. If humans preferred A over B, the reward model learns:

```text
Reward(A)  >  Reward(B)
```

That's all it learns. (Technically it is usually built from a copy of a language model whose LM head is swapped for a head that outputs a single number, and trained so that the preferred answer scores higher than the rejected one. A diagram a little further down shows exactly how.) Eventually it becomes good at predicting human preferences.

**An important point on why it exists at all:** humans are slow. A lab may want tens of millions of training steps, and can't ask a person to grade every response. So humans grade maybe 100,000 examples, and from those the reward model learns to automatically score millions of future responses.

### Is RLHF a new model? Two models, one stays

This is the biggest confusion people have about RLHF, so let's separate the pieces:

> **RLHF does NOT produce another model architecture. It continues training the *same* LLM.**

But there is a second model involved during training:

| | The LLM | The Reward Model |
|---|---|---|
| Role | The model you ultimately use | The grader |
| Trained how | Predict tokens (pre-training, SFT), then RL | Given two answers, predict which one humans preferred |
| Changes during RLHF | **Yes** (its weights are updated) | No (it only scores) |
| After training | **Kept**: it is the model you download | **Thrown away** |
| Used at inference | Yes | **No**: nothing grades your chat answers |

A school analogy: a student takes an exam and the teacher grades it. Does the teacher become part of the student's brain? No. The student learns from the grade, and the teacher goes home. The three stages are three different teachers, but **the student is always the same person**:

```text
Stage 1   Teacher:  "Learn English."                              (base model)
Stage 2   Teacher:  "Learn to answer questions."                  (SFT)
Stage 3   Judge:    "Of these two answers, which is more helpful?" (RLHF)
```

A more precise pipeline picture:

```text
                    Human preferences
                           │
                           ▼
                     Reward Model
                           │  (scores answers)
                           ▼
Base Model → SFT → Instruction Model → RL optimization → Final Chat Model
                         (the same LLM, continually updated)
```

So RLHF is not another model. It is **another training stage** that uses a **separate reward model** to improve **the same LLM**.

### The reward model, drawn out

A question that naturally comes up: once the reward model exists, do we *add* it to the LLM, or *use* it to train the LLM and then discard it? The answer is the second. It is used to train the LLM, then thrown away. Here is the whole life of the reward model in three phases.

**Phase A: build the reward model (a separate model, not part of the LLM).**

```text
   SFT model (the LLM)                         Human preference pairs
   ┌─────────────────────┐                     prompt, answer A  >  answer B
   │ Transformer blocks  │                                │
   │ LM Head: d × 50,000 │                                │
   └─────────┬───────────┘                                │
             │ COPY the weights                           │
             ▼                                            │
   ┌─────────────────────┐                                │
   │ Transformer blocks  │   (same body, now a separate   │
   │ (copied)            │    model with its own weights) │
   ├─────────────────────┤                                │
   │ REWARD HEAD: d × 1  │ ◄──── LM head swapped for a    │
   └─────────┬───────────┘       head with ONE output     │
             │                                            │
             ▼                                            ▼
        one score  ─────►  train so that  score(A) > score(B)
```

The only swap is the last matrix. The LM head is `d × vocabulary` and produces a probability for every word. The reward head is `d × 1` and produces **one number for the whole answer**. Everything else is an ordinary Transformer, trained on the preference pairs.

**Phase B: the RL loop (the reward model is the grader).**

```text
                    ┌────────────────────────────┐
   Prompt ─────────►│  LLM  (being trained)      │──► Answer
                    │  weights UPDATE            │       │
                    └────────────▲───────────────┘       │
                                 │                       ▼
                                 │            ┌────────────────────┐
                                 │            │  REWARD MODEL      │
                                 │            │  weights FROZEN    │
                                 │            └─────────┬──────────┘
                                 │                      │ one number, e.g. 9.2
                                 │                      ▼
                                 │            ┌────────────────────┐
                                 └────────────│ RL loss (PPO) →    │
                      gradients flow          │ backprop → AdamW   │
                      into the LLM only       └────────────────────┘
```

The reward model only *reads* the answer and returns a score. No gradient flows into it, and none of its weights change. The score becomes a loss, backpropagation runs, and **only the LLM's weights move**.

**Phase C: after training.**

```text
   Before:  [ LLM ]  +  [ Reward model ]
                              │
                              ▼  thrown away
   After:   [ LLM, with updated weights ]   ← the only thing you ship
```

The reward model leaves nothing inside the LLM. Its contribution is *indirect*: it shaped the gradients, and the LLM's own weights absorbed the result. At chat time there is only the LLM, and nothing is grading your answers.

A good way to hold it in your head: **the reward model is the examiner's mark scheme, not part of the student.** It decides how the student is marked during revision, then it stays in the exam office.

### The missing link: how does one score update billions of weights?

The reward model does **not** produce a giant matrix. It produces **one number**, for example `Reward = 9.2`. So how does one number change 70 billion parameters?

First, recall that cross-entropy also produced just one number (`Loss = 0.21`), and it was **backpropagation** that turned that number into gradients for every weight. Cross-entropy didn't update anything itself. The same happens here:

```text
SFT:     Cross-entropy loss   →  Backpropagation  →  AdamW
RLHF:    Reward  →  RL loss   →  Backpropagation  →  AdamW
```

**Backpropagation is still there.** The reward model and the LLM are separate networks, so we do not backpropagate *through* the reward model into the LLM. Instead, the reward is used to **build a new loss** for the LLM. That is where *reinforcement learning* comes in.

### Reinforcement learning, just enough to understand RLHF

Forget LLMs for a moment and imagine teaching a robot to walk. First attempt: it takes two steps and falls, and you give it `Reward = 2`. Second: three steps, `Reward = 5`. Third: it walks perfectly, `Reward = 10`. Did you tell the robot what to do with each motor? No. You only said *"that attempt was better."* A biscuit for a dog works the same way: it doesn't instruct each muscle, it just says "whatever you just did, do more of that."

Now replace the robot with GPT. A generated answer is a sequence of **actions**, one per token, and the final reward is attributed to that whole sequence:

```text
Token 1 → Token 2 → Token 3 → ... → Finished answer  →  Reward 9.1
```

Compare the two kinds of supervision:

| | Cross-entropy (SFT) | RLHF |
|---|---|---|
| The teacher says | "The correct next token was **Paris**." | "I liked your **whole answer**: 9.2/10." |
| Signal | Precise: which token should have been produced | Weak: the answer was good, but not *why* |
| Pushes up | the probability of **this exact next token** | the probability of **this whole successful sequence** |

How does this work mechanically? The model already knows the probabilities it assigned when it generated each token:

```text
Artificial   0.65
Intelligence 0.82
is           0.91
```

If the reward is high (say 9.5), the RL algorithm says: *raise the probabilities of the choices that led here.* Next time `Artificial` may become `0.72`, `Intelligence` `0.87`. If the reward is terrible (1.2), it says: *lower them*, e.g. `0.65 → 0.41`. We aren't changing the architecture, attention or Transformer blocks. We are **changing the probabilities the LLM assigns**, exactly as cross-entropy did.

### PPO: don't overreact

The classic RL algorithm used for this is **PPO (Proximal Policy Optimization)**, the one used in the original ChatGPT-era papers. You don't need to be a PPO expert. You only need one idea:

> **PPO is the mathematical bridge from "I liked this answer" to "here are the gradients to update billions of weights", while keeping each update small and safe.**

Why the caution? Rewards are noisy. If today's reward says 10 and tomorrow's says 3 for similar answers, and we update aggressively each time, the model becomes unstable. PPO's philosophy is *"don't overreact to one good or bad answer."* **Proximal** means *nearby*: small, controlled steps. It's like learning piano: when a teacher says "good", you don't suddenly change your entire playing style; you improve a little, then a little more.

In practice, RLHF setups also keep a frozen copy of the starting model and penalize the LLM for drifting too far from it, so it can't wander off into strange text that merely scores well.

This means a typical PPO-based setup holds **four models** in memory at once:

| Model | Trained? | Role |
|---|---|---|
| The LLM (the "policy") | **Yes**, this is the one being improved | Generates answers |
| Reference copy of the starting LLM | No, frozen | The "don't drift too far from this" anchor |
| Reward model | No, frozen | Scores each answer with one number |
| Value model (a "critic") | Yes, but it is only a helper | Gives PPO a baseline so its updates are steadier |

Only the first one survives to be shipped. Holding all four is a big part of why RLHF is expensive, and it is exactly the machinery DPO (Part 8) removes.

### The weak spot: reward hacking

The reward model was trained from human preferences. What if the reward model itself is wrong, or can be fooled? The LLM will optimize for *whatever the reward model rewards*, including its mistakes: say, answers that are long and confident-sounding because the reward model happened to like those. This is called **reward hacking** (or reward misspecification), and it is one of the biggest challenges in RLHF.

### When does RLHF stop?

There is no fixed dataset to count epochs on, because the LLM generates its own answers during training. So the stopping rule is a fixed budget of steps *plus monitoring*:

- Watch the **average reward** over time. When it plateaus, there's little left to gain.
- Watch the **drift from the reference copy**. If the LLM has moved too far from the starting model, you stop, or you pull it back with a stronger penalty.
- Watch **real quality**, judged by humans or benchmarks. If the reward keeps rising while real quality *falls*, that's reward hacking (above), and the run should stop. The reward model is only a proxy for what people want.

### The cost of RLHF

Look at what RLHF needs: a reward model, PPO, rollouts (generating answers during training), policy updates, and a lot of hyperparameters. It is expensive, complex and sometimes unstable. That led researchers to ask:

> The reward model was trained from human preferences. Why not train the LLM **directly** from those same preferences, and skip the reward model and the reinforcement learning entirely?

---

## Part 8 — DPO: preferences without the reinforcement learning

**DPO (Direct Preference Optimization)** is one of those papers where you read the title and expect something exotic, and then understand it and think: *why didn't we do this from the start?*

### The key insight

Remember what RLHF needed:

```text
Human preferences → Reward Model → reward score → PPO → update LLM
```

Three moving parts: the reward model, PPO, and the LLM. Do we really need the first two?

Look at the original data. A human compared two answers and clicked `A is better`. We **already know** which is preferred. Why train a reward model to say `A = 9.2, B = 6.4`, when all we care about is:

```text
A  >  B
```

We don't care whether A deserves 9.2 or 8.8. **So learn the preference directly.**

```text
RLHF:   Human → Reward Model → Reward → PPO → LLM
DPO:    Human → LLM          (done: no reward model, no PPO)
```

### How it works

Cross-entropy said: *increase the probability of the correct token.* DPO says something only slightly different:

> **Increase the probability of the preferred answer, and decrease the probability of the rejected answer.**

An illustration. For the prompt `Explain AI.`, suppose the model currently leans the wrong way, preferring answer B even though humans chose A:

| Answer | Before training | After training |
|---|---:|---:|
| A (preferred) | 40% | 70% |
| B (rejected) | 60% | 30% |

(The percentages are only an illustration of the *direction* of the push. In reality the probability of any full answer is tiny; what training adjusts is how much more likely the model finds A than B.)

No reward score, no PPO, only a probability adjustment. It's the **same philosophy as SFT**, applied to complete responses instead of single next tokens.

Where does backpropagation happen? Exactly where you'd expect:

```text
Prompt
   │
   ▼
LLM  →  Preferred answer
        Rejected answer
   │
   ▼
DPO loss
   │
   ▼
Backpropagation
   │
   ▼
AdamW
   │
   ▼
Update weights
```

Backpropagation stays. AdamW stays. The Transformer stays. **Only the loss function changes.** As in SFT, the DPO loss is computed on the *response* tokens, with the prompt as context. (The real DPO loss also compares the model against a frozen copy of the SFT model, again to stop it drifting too far. That's the same safety idea PPO setups use.)

### Why industry liked DPO

It removed two difficult components. Instead of a reward model plus PPO, all you need is a **preference dataset**. It is simpler, easier to reproduce, and easier to train, and often reaches similar results. Does RLHF disappear? No. Some labs still use it, some use DPO, and some use newer preference-optimization methods. The field is still evolving. But DPO showed you can get strong alignment *without* running a full reinforcement-learning algorithm. So if you read a modern model's paper, don't be surprised to see "SFT + DPO" instead of "SFT + reward model + PPO".

### When does DPO stop?

DPO trains on a fixed set of preference pairs, so it looks much more like SFT: usually **1 to 3 epochs**, with a validation set held back. You watch validation loss and how often the model prefers the chosen answer over the rejected one. You stop before the model drifts too far from the starting model (the frozen reference copy exists for exactly this reason), because pushing the preferred and rejected answers apart for too long makes the model's outputs strange.

---

## The complete picture: the lifecycle of a chat model

Every stage uses the **same core engine**:

```text
Input tokens → Transformer → LM Head → Probabilities
```

The only thing that changes is **the definition of "good"**:

| Stage | What defines "good"? | Loss |
|---|---|---|
| Pre-training | Correct next token | Cross-entropy on all tokens |
| SFT | Correct assistant response | Cross-entropy, prompt masked |
| RLHF | High reward | RL loss (PPO) built from the reward |
| DPO | Preferred response over rejected | Preference loss |

The architecture barely changes. The optimizer (AdamW) barely changes. Backpropagation doesn't change. **The training objective changes.**

```text
Raw Internet Text
        │
        ▼
Pre-training                         (predict the next token)
        │
        ▼
Base Model
        │
        ▼
Instruction Dataset
        │
        ▼
Supervised Fine-Tuning               (full fine-tuning, or LoRA / QLoRA)
        │
        ▼
Instruction Model
        │
        ▼
Human Preference Data ───────────────► Reward Model
        │                                   │
        ▼                                   │
DPO   ── or ──   RLHF / PPO  ◄──────────────┘
        │
        ▼
Aligned Chat Model
```

LoRA and QLoRA are not extra stages. They are a **cheaper way of doing a stage** (most often SFT, and they can also be used for preference training).

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "ChatGPT is a different architecture from GPT." | It's the same Transformer. The training pipeline changed, not the architecture. |
| "Fine-tuning means training only the last layer." | Traditional fine-tuning continues training *every* parameter. (LoRA is what trains only a small part.) |
| "SFT uses a different loss function." | It is still cross-entropy on the next token. The differences are the data and the loss mask. |
| "The loss is computed on the whole conversation." | Only the assistant's tokens get loss; prompt labels are set to `-100`. |
| "The loss mask and the causal mask are the same thing." | The causal mask is inside attention (what the model can *see*); the loss mask is at cross-entropy (what the model is *graded on*). |
| "The prompt is thrown away in SFT." | The prompt is fully read through attention; only its *loss* is ignored. |
| "LoRA changes only a small fraction of the model's outputs." | `W + BA` is a full-size matrix, so every output changes. Only the *parameter count* is small. |
| "LoRA is an extra neural network that generates the adapter." | `A` and `B` are ordinary weights learned by backpropagation. |
| "One LoRA adapter is shared across all blocks." | Each targeted matrix in each block has its own `A` and `B`. |
| "A LoRA file is a tiny few-kilobyte patch." | It's hundreds of small matrices: typically tens to hundreds of MB. |
| "LoRA/QLoRA make inference slower." | An adapter can be merged into the weights, so inference cost is the same as the base model. |
| "RLHF produces a new model." | It's another training stage on the *same* LLM, guided by a separate reward model. |
| "The reward model runs every time you chat." | It exists only during training and is discarded. |
| "The reward model gets merged into the LLM afterwards." | It never does. It only shapes the gradients during training, and the LLM's own weights absorb the result. |
| "The reward model is a whole different kind of network." | It is a copy of a language model whose LM head (`d × vocabulary`) is swapped for a one-number head (`d × 1`). |
| "Fine-tune until the training loss is as low as possible." | Fine-tuning data is small, so the model memorises it. Stop when *validation* loss stops improving, usually within 1 to 3 epochs. |
| "The reward model outputs a matrix that goes back into the LLM." | It outputs one number. The RL algorithm turns it into a loss; backpropagation then updates the LLM. |
| "RLHF means the model learns from talking to users." | RLHF uses a *fixed* set of collected human preferences during a training phase, not live conversations. |
| "DPO is reinforcement learning." | DPO skips the reward model and the RL loop. It trains directly on preference pairs. |

---

## Quick reference

```text
Base model     predicts the next token; knowledge but no manners
Instruct model base + SFT (+ preference alignment); same architecture and tokenizer

SFT            continue training on  <User> ... <Assistant> ...  examples
               same cross-entropy; labels of the prompt set to -100 (loss mask)
Loss mask      attention reads everything; loss counts only the assistant tokens
Two masks      causal mask = inside attention;  loss mask = at cross-entropy

Full FT        update every parameter  (70B → 70B updates, a full copy per task)

LoRA           freeze W;  learn ΔW = B·A   (B: d_out×r,  A: r×d_in,  r ≈ 8)
               4096×4096 → 65,536 params (~256× fewer);  B starts at 0
               one (A, B) pair per targeted matrix per block (often W_Q, W_V)
               inference:  W + B·A   (can be merged)
QLoRA          LoRA on top of a 4-bit frozen base model; adapters in 16-bit

RLHF           preference pairs → Reward Model (one score) → PPO → same LLM
               reward model is discarded afterwards;  risk: reward hacking
DPO            preference pairs → push preferred ↑, rejected ↓, no reward model, no RL

Stopping:      pre-training → token budget spent
               SFT / LoRA   → validation loss stops improving (usually 1–3 epochs); keep the best checkpoint
               DPO          → 1–3 epochs; watch validation loss and drift from the reference model
               RLHF / PPO   → step budget; watch reward, drift, and real quality (reward hacking)
RLHF models:   LLM (trained) + reference copy (frozen) + reward model (frozen) + value model (helper)
               only the LLM is kept

Always the same:  Transformer + backpropagation + AdamW.   Only the objective changes.
```

---

## What's next

Fine-tuning produced a model that behaves well. Now the engineering questions begin. Even a 7-billion-parameter model in 16-bit numbers needs about 14 GB of memory, and QLoRA just showed that storing weights in fewer bits can work surprisingly well. Chapter 14 looks at **quantization**: what number formats models use (FP32, FP16, BF16, INT8, INT4), how weights are compressed, and what quality is traded away for the memory and speed gained.
