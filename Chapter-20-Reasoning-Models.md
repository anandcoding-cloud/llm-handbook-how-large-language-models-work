# Chapter 20 — Reasoning Models: Test-Time Compute, Thinking Budgets and Reasoning Effort

## Introduction

So far, we have treated a language model as a machine that reads a prompt and starts writing the answer straight away, one token after another. Chapter 18 showed how to make such a model *bigger* without making it slower, and Chapter 19 showed how to give it new senses. This chapter is about a completely different way to make a model better at hard problems:

> **Let it think before it answers.**

If you have used one of the newer "reasoning" or "thinking" models, you have probably noticed two things:

1. The model shows (or at least spends time producing) a block of **reasoning text** before it gives its final answer.
2. You can often choose **how hard it thinks**: low, medium or high.

Both observations are the surface of one idea, and both have a precise explanation. This chapter builds it from the ground up:

1. **Why thinking longer helps**, even though the model is "just predicting tokens".
2. **Test-time compute**: a new way to scale a model, besides making it bigger.
3. **How a reasoning model is trained**: reinforcement learning with *verifiers*, and a concrete training iteration.
4. **Verifiers**: what they are when there is no calculator.
5. **Spending more compute at answer time**: longer chains, many samples, search.
6. **Reasoning effort**: what the Low / Medium / High control really does, and why the reasoning is sometimes visible.
7. **The costs**: KV cache, memory, latency.
8. **Combining it with quantization**, and where reasoning ends and agents begin.

One clarification up front, because it is the most common misunderstanding:

> **A reasoning model usually has no new architecture.** It is the same Transformer as before. What changes is how it was *trained* and how it is *used at inference time*.

---

## Part 1 — Why thinking longer helps

### The same model, two behaviours

Take a simple question: **What is 17 × 24?**

A model that answers directly might produce:

```text
17 × 24 = 408
```

A reasoning model might produce:

```text
17 × 24
24 = 20 + 4
17 × 20 = 340
17 × 4  = 68
340 + 68 = 408
Therefore the answer is 408.
```

Both are just tokens, generated one at a time by the loop from Chapter 1. There is no special "reasoning engine" hidden in the Transformer. So why would the second approach be more reliable?

### A Transformer does a fixed amount of work per token

Recall the structure of the model (Chapter 8): a token passes through **N blocks**, and then the LM head produces the next token. That is a fixed number of steps, no matter how hard the question is. The model cannot loop inside a single forward pass until it is satisfied. For a question that needs many sequential steps (such as multi-digit arithmetic, a logic puzzle or a multi-step proof), N blocks may simply not be enough to get from the question straight to the answer in one go.

Now look at what happens when the model writes out intermediate steps. Each generated token is **fed back in as context** (Chapter 12), so the next token can attend to it:

```text
Question
   ↓
Step 1  (a few tokens)       ← becomes context
   ↓
Step 2  (reads Step 1)       ← becomes context
   ↓
Step 3  (reads Steps 1 and 2)
   ↓
Answer  (reads all of it)
```

The model has created a **scratchpad using its own context**. The weights do not change during inference. Instead, the model *computes through a sequence of generated representations*: each new token can use the results of the earlier ones. In effect, writing more tokens buys more sequential computation, because every token is another pass through all N blocks.

That is the core reason "thinking out loud" works:

```text
Direct answer:    Question ───────────────────────────────► Answer
                  (one pass; fixed amount of computation)

With reasoning:   Question ► step ► step ► step ► step ► Answer
                  (many passes; each can use what came before)
```

(This idea predates "reasoning models". In 2022, **chain-of-thought prompting** showed that simply showing a large model a few worked examples with intermediate steps, or even just asking it to "think step by step", improved its performance on arithmetic and logic problems a great deal. What is new is that models are now *trained* to do this on their own.)

---

## Part 2 — Test-time compute: a new way to scale

Until now, there were three ways to make a model stronger:

```text
More parameters       (7B → 70B → 400B)           how much the model can hold
More training         (more data, more compute)   how well it was taught
```

Reasoning models add a third dimension:

```text
More computation at INFERENCE time, spent on the particular problem in front of it
```

This is called **test-time compute** (or inference-time compute). Different problems deserve different amounts of effort:

```text
Easy:    "What is 2 + 2?"          → a very short answer
Hard:    "You have 5 boxes..."     → a long chain of reasoning, with checking
```

So inference becomes **adaptive**: easy questions get short reasoning and hard ones get long reasoning. It is worth separating the three kinds of "scale":

| | What it measures | Example |
|---|---|---|
| **Model size** | How much learned capacity the model has | 7B → 70B |
| **Training compute** | How much computation was spent *teaching* it | Weeks on thousands of GPUs |
| **Test-time compute** | How much computation it spends *on your problem* | Low → High reasoning |

Research has found this trade is real. One study found that on problems where a smaller model has a non-trivial success rate, spending extra compute at inference time in a well-chosen way could match or beat a model 14 times larger. This is a very different scaling strategy from "just make the model bigger".

A connection to Chapter 23: speculative decoding asks *"how do we generate the same tokens faster?"* Reasoning asks the opposite: *"when does it pay to generate more tokens on purpose?"*

---

## Part 3 — How a reasoning model is trained

### The limit of ordinary fine-tuning

With normal supervised fine-tuning (Chapter 13) we show the model a question and a good answer and train it to reproduce the answer. To teach reasoning that way, we would need examples of **good reasoning**:

```text
Question:   If 3x + 5 = 20, what is x?
Reasoning:  3x = 15,  x = 5
Answer:     5
```

We can use such examples, and they do teach reasoning patterns. But there's a problem: **people cannot write millions of high-quality reasoning examples**, and imitating human-written steps may not be the best way for a model to reason anyway. So researchers looked for a different source of supervision.

### The idea: reward correct answers, not good-looking steps

For many problems, we can check automatically whether the final answer is correct:

```text
Math:        17 × 24 = ?   → a calculator or solver checks 408
Code:        write a sorting function → run the tests → pass or fail
Chess:       a game engine
Puzzles:     a known solution or checker
```

That automatic checker is a **verifier**. Now we can use the reinforcement-learning machinery you met in Chapter 13, but with a different reward source:

```text
RLHF  (Chapter 13):   LLM → answer → reward model trained from HUMAN PREFERENCES → reward
RLVR  (this chapter): LLM → reasoning + answer → VERIFIER ("is it actually correct?") → reward
```

This is called **reinforcement learning with verifiable rewards (RLVR)**. The reward can be as simple as:

```text
Correct = 1        Wrong = 0
```

The model is then trained to produce *reasoning that leads to correct answers*. And just as with RLHF, a human is no longer in the loop for every example.

Why is this so powerful? **Human preference is subjective.** For "write a good explanation", two answers can both be good, and a perfect reward is hard to define. But `17 × 24 = ?` has an objective answer, and "does this program pass its tests?" can be checked by running them. In domains where correctness can be checked, the training signal is much stronger and cannot easily be fooled.

### One complete training iteration, with a tiny model

Take our TinyGPT (the tiny model of Appendix A) and follow one iteration:

> **There are 3 boxes with 4 apples in each. How many apples?**

```text
 1. Tokenize the question            → token IDs
 2. Embeddings → Transformer blocks → LM head    (nothing special so far)
 3. The model GENERATES reasoning, one token at a time:
        "There are 3 boxes. Each box has 4 apples. 3 × 4 = 12. Therefore the answer is 12."
 4. The VERIFIER compares the answer with the expected one:   12 = 12  →  correct
 5. Reward  R = 1
 6. RL objective:  "make this kind of trajectory more likely"
 7. Backpropagation → gradients → AdamW → the model's weights change
```

Every reasoning token is an ordinary next-token prediction. The only new ingredient is the *reward*, which tells the training procedure whether the whole trajectory was a good one. Over many examples, the probability of reasoning patterns that lead to correct answers goes up. Given the weights, the same prompt will then tend to produce more of that kind of thinking.

Notice also what the verifier is **not**: it is **a training tool**. After training, a user's question goes straight to the reasoning model, which reasons and answers. The verifier isn't there. It is exactly like the reward model in RLHF, which Chapter 13 showed is discarded after training. There are three separate roles:

```text
LLM          generates the reasoning
Verifier     judges the result
RL algorithm uses that judgment to change the LLM
```

### What does the RL algorithm look like? GRPO

Chapter 13 showed classic RLHF with PPO, which needs four models in memory, including a "value model" (a critic) that estimates how good each situation is. For reasoning, a simpler method became popular, called **GRPO** (Group Relative Policy Optimization), introduced for training mathematical reasoning. Its trick is to drop the critic and use a **group of answers to the same question** as its own baseline.

For each question, the model samples a group of, say, 4 attempts, and each gets a reward from the verifier. Then each attempt's **advantage** is how much better or worse than the group average it was:

```text
advantage  =  (reward − group average) / (spread of the group)
```

Two examples with 4 attempts each:

```text
Attempts:  A   B   C   D                          Attempts:  A   B   C   D
Rewards:   1   0   1   0     average 0.5          Rewards:   1   1   1   0     average 0.75
Advantage: +1  −1  +1  −1                         Advantage: +0.58 +0.58 +0.58 −1.73
```

The training signal then says: *raise the probability of the tokens in attempts with positive advantage, and lower it for those with negative advantage.* Notice the second group. Three attempts got it right, so each is only mildly encouraged, but the one that got it wrong is pushed down hard. If **all** attempts are right (or all wrong), every advantage is zero, and there is nothing to learn from that question. This also keeps the updates small and stable, with a penalty that stops the model drifting too far from where it started, in the same spirit as PPO's caution in Chapter 13. The group average replaces the critic, which is why GRPO needs less memory than PPO.

### The recipe in practice: DeepSeek-R1

The best-known public example is **DeepSeek-R1**, a model trained to reason mostly with this kind of reinforcement learning. The story has two parts:

- **R1-Zero:** reinforcement learning applied directly to a *base* model with simple rule-based rewards (is the answer right, and is the output in the expected format), with no human-written reasoning examples at all. Behaviours such as checking its own work, noticing an error and changing approach **emerged during training**, without anyone writing them in. Nobody hard-coded "check your answer". It's behaviour that earns reward.
- **R1:** because R1-Zero's output could be hard to read, the full recipe added a small "cold start" of curated reasoning examples before RL, then further stages (more RL, extra supervised fine-tuning on good samples, and a final stage for general helpfulness). The reasoning ability could then also be **distilled** into much smaller models by fine-tuning them on the larger model's reasoning traces.

You will recognize this shape in Chapter 21's description of Qwen3's post-training: *cold start → reasoning RL → merging thinking and non-thinking modes → general RL*. That is the same pattern with different details.

A last point on training: the reasoning ability **can be taught by imitation, too**. Fine-tuning a model on a small, carefully curated set of long reasoning examples (as little as a thousand, in one study) already gives a large boost. RL is the way to push further, without needing humans to write the examples.

---

## Part 4 — Verifiers: what if there is no calculator?

The calculator is the easy case. The natural question is: *"not every answer can be checked by a program, and a verifier has to work across many kinds of problems to scale. How does it work?"*

The honest answer is that **there is no single universal verifier**. Modern training uses different kinds, depending on the task:

### A. Programmatic verifiers (exact and cheap)

```text
Math        → calculator, symbolic solver, comparing with a known answer
Code        → run the tests
Chess       → a game engine
Puzzles     → a checker for the known solution
```

These give a clean `1` or `0`. The signal is strong and hard to fool. This is why **math and coding** are such attractive domains for reasoning RL: you can often turn "is this correct?" into an executable test.

### B. Learned verifiers (a model that judges)

For "is this physics explanation logically sound?", no simple program exists. So you train *another model* to judge, from examples labelled `correct` and `flawed`. The verifier can itself be an LLM. But a learned verifier **can be wrong**, and if the reasoning model discovers how to exploit its blind spots, you get **reward hacking** (Chapter 13): answers that score well without being good.

### C. Process verifiers (checking the steps, not just the answer)

Instead of checking only the final answer, a process verifier scores each step of the reasoning:

```text
Step 1: 3x + 5 = 20   ✅
Step 2: 3x = 15       ✅
Step 3: x = 5         ✅
```

This is richer feedback than a single tick at the end. The distinction is called **outcome supervision** (reward the final answer) versus **process supervision** (reward each step). A well-known study found that process supervision beat outcome supervision for training models to solve competition-level maths problems, and released a dataset of 800,000 step-level human labels to support more research.

### Combining them

In practice a system may combine several:

```text
                    Reasoning model
                          │
                          ▼
                       Solution
                          │
                 ┌────────┴────────┐
          Programmatic         Learned
           verifier            verifier
                 └────────┬────────┘
                          ▼
                       Reward
```

The guiding principle is:

> **Make correctness easier to evaluate than it was to produce.**

Where that is possible (maths, code, logic) reasoning RL is on its strongest ground. Where it isn't (creative writing, strategy, open-ended advice) it's much harder, and models lean more on learned judgments and human preferences.

---

## Part 5 — Spending more compute at answer time

The trained model can also use extra compute in several ways when it answers. They fall along a ladder of increasing sophistication.

### 1. A longer single chain

The simplest: let one reasoning path run longer, with more steps, more checking, and more self-correction (`Wait, let me re-check that...`). Training with RL tends to teach the model to do exactly this when a problem is hard.

### 2. Many independent samples: self-consistency

One reasoning path can go wrong from a bad assumption. More thinking won't fix it if the chain is built on a mistake. So instead, sample **several independent paths** and see which answer they agree on:

```text
                 Question
                    │
          ┌─────────┼─────────┐
        Path A    Path B    Path C
          │         │         │
          42        47        42
          └─────────┼─────────┘
                    ▼
            Most common answer
                    ▼
                   42
```

This is **self-consistency**: if several independent attempts reach the same answer, confidence goes up. It really does help. The original paper reported gains of around 18 points on a grade-school maths benchmark, and several points on others, without changing the model at all.

A feel for why it works (an idealized case): suppose each independent attempt is right 60% of the time, and the wrong answers are spread across different values. Then a majority vote is right about 68% of the time with 5 samples, 83% with 21, and 98% with 101. Real attempts are not fully independent, so the gains are smaller, but the principle holds. The trade-off is that you pay for every extra sample.

A related method is **best-of-N with a verifier**: generate N candidates, score each with a verifier (Part 4), and keep the best one.

### 3. Search: branch, evaluate, prune

Why stop at independent paths? We can let the model **explore different possibilities and deliberately choose between them**. A plain chain is one line:

```text
Question → Think → Think → Think → Answer
```

If the model reaches a decision point ("use Method A or Method B?"), it can **branch** instead of committing, giving a tree:

```text
             Question
                 │
             Decision
              /     \
            A         B
            │         │
           ...       ...
```

This is the idea behind **Tree of Thoughts**. In a puzzle that needs planning and look-ahead (the "Game of 24"), the paper reported that GPT-4 with ordinary chain-of-thought solved only about 4% of cases, while a tree-search version solved about 74%.

The catch is that the tree explodes: you can't explore everything. So you need a way to decide which branches are worth continuing, which is the verifier's job again. Candidates are scored, bad ones are **pruned**, and good ones are explored further:

```text
                  Question
                     │
              Generate candidates
                 /   │   \
                A    B    C
                │    │    │
              score score score
               0.9   0.3   0.8
                │          │
              keep A     keep C      (B is pruned)
                │          │
             explore    explore
```

That is already much closer to *search* than to ordinary text generation: **generate, explore, evaluate, prune, verify**. And the underlying model can remain exactly the same Transformer.

### "But isn't the model sequential? Wouldn't exploring several paths take several times longer?"

This is a sharp question, and the answer is: **sequential within a path, parallel across paths.**

The model is *autoregressive* inside one path: you can't produce token 2 before token 1. But Path A, Path B and Path C are independent, so they can be **batched** and processed at the same time on the GPU, which is the same idea as batching different users in Chapter 22:

```text
A₁ → A₂ → A₃ → A₄
B₁ → B₂ → B₃ → B₄        three sequences in one batch: each step advances all three
C₁ → C₂ → C₃ → C₄
```

And the paths often share a common beginning (the question and some early reasoning), so a smart system can compute that shared prefix **once** and branch from it. This uses the prefix caching and block sharing from Chapter 22.

To correct the original phrase: the Transformer is not "non-parallel". Inside each forward pass, it is massively parallel (matrix multiplications, attention). What is sequential is the *generation of tokens along one path*. So reasoning effort can include more exploration, not just a longer chain.

### Which is better: think longer, or sample more?

They spend compute differently, and neither always wins:

```text
More reasoning tokens:    one path, ─────────────►  longer     (depth)
More reasoning samples:   many paths, in parallel               (breadth)
```

Research on test-time compute found that the best mix depends on **how hard the problem is**, which is why adaptive allocation, spending more on hard questions and less on easy ones, matters.

---

## Part 6 — Reasoning effort: what the Low / Medium / High control really does

You noticed two things when using a reasoning model. Now we can explain both.

### Observation 1: "The model shows reasoning text before the answer"

A reasoning model generates something like:

```text
[reasoning tokens ...]
[final answer ...]
```

The reasoning tokens are **ordinary generated tokens**. The model has typically been trained to mark where thinking ends and the answer begins, so it learns when to stop thinking and start answering. Whether *you* see them is a decision of the product:

```text
Model generates:   reasoning tokens ──► final answer
                                ▲
                       the interface decides what you see:
                       show it all, show a summary, or hide it
```

So visible reasoning is partly a **product and interface decision**, not a different kind of model. Providers differ: some show the raw reasoning, some show only a summary, some hide it. Two cautions follow. What you see may not be a full transcript of everything the model computed. And a model's written reasoning is not guaranteed to be a faithful account of *why* it reached its answer, so it is useful evidence, not proof.

### Observation 2: "I can pick the level of reasoning"

That is a **reasoning-effort** control. The important word is **budget**: *how much computation the system may spend on this problem.* But "budget" is only part of the story, and it is worth looking at how the control really works, because most people now use reasoning models most of the time, and the settings change both what you pay and how the model behaves.

First, the key point to hold on to: when you move the effort selector, **you are not switching to a different model**. It's the same weights, run under a different policy. The question is where that policy lives, how the model decides when to stop, and what changes in its behaviour.

### Three places where the policy can live

A reasoning-effort setting is not one mechanism. It is made of up to three layers, and different products combine them differently:

```text
LAYER 1   THE MODEL ITSELF         (soft control)
          Learned behaviour. The setting is part of the prompt, and the model was
          trained to think more or less because of it, and to decide when to stop.

LAYER 2   THE SERVING LAYER        (hard control)
          Ordinary software around the model: token limits, and forcing the
          model to end its thinking when a budget is used up.

LAYER 3   THE SYSTEM AROUND IT     (extra compute)
          Several attempts, voting, verifiers, search (Part 5).
```

| Layer | What it does | Example |
|---|---|---|
| **1. The model (soft)** | The effort level is text in the context. The model reads it and behaves accordingly, because training taught it to. | An open model whose system prompt says "Reasoning: high" (or low, or medium) |
| **2. The serving layer (hard)** | Enforces a ceiling, regardless of what the model "wants". | A maximum number of tokens; cutting thinking off and forcing the answer when a thinking budget is reached |
| **3. The system around it** | Spends compute on more than one path. | Sampling several answers and voting; verifier-guided search |

The rest of this part takes them in turn. Since layer 1 contains the most surprising idea, we start by asking how the model decides to stop thinking at all.

### How does the model decide to stop thinking?

Here is the thing that surprises most people: **there is no separate "stop controller" inside the model.** Thinking has a beginning and an end, and both are marked by special tokens, such as `<think>` and `</think>` in many open models. (Other systems use differently named markers or separate "channels", but the idea is the same.) Ending the thinking is simply **producing the end-of-thinking token**, and that token is chosen the way every token is chosen: the LM head produces scores for the whole vocabulary, softmax turns them into probabilities, and the sampler picks one (Chapters 8 and 12).

So at every single step of the reasoning, the vocabulary includes the end-of-thinking token, and the model assigns it a probability. A made-up but realistic illustration, at two moments in the 17 × 24 example:

```text
Moment A:  the model has just computed  "340 + 68 = 408"  but hasn't checked it
           next token:   "Wait"      40%      ← "let me double-check"
                         "So"        25%
                         "Let"       15%
                         "</think>"   5%      ← stop thinking and answer
                         other       15%

Moment B:  the model has just confirmed  "24 × 17 = 408 as well, so it matches"
           next token:   "</think>"  80%      ← stop thinking and answer
                         "Wait"      10%
                         other       10%
```

At moment A, the model usually keeps going, so it writes something like "Wait, let me verify". At moment B, the check has passed, so it usually stops. Nothing is "deciding" in a separate part of the network. The same Transformer, reading everything it has written so far, simply finds that ending the thinking is more or less likely at this point.

### Why training teaches it when to stop

The stopping behaviour is learned through the reinforcement learning of Part 3, and the reward structure pushes in two opposite directions:

- **Stopping too early is punished:** the answer is wrong, so the reward is 0.
- **Never stopping is punished too:** training runs under a maximum length. A response that runs out of room before it gives an answer earns nothing. (In real products the same thing is visible to you: if the token limit is hit during thinking, you can pay for all the reasoning tokens and get no visible answer at all.)

The model therefore learns a *calibrated* habit: think long enough to get the answer right, then stop. Because the best length depends on the problem, the habit is **adaptive**. An easy problem gets a high stop probability early, while on a hard one the model keeps producing the kind of tokens that lead to checking and exploring. Some recipes also add an **explicit length reward**, for example rewarding the model for finishing close to a requested length, as in the L1 research below.

The phrases that keep thinking going ("Wait", "Hmm", "Let me double-check") are learned habits too: self-checking behaviour that RL rewarded because it caught mistakes. This is also why a research method called *budget forcing* works: to make the model think longer, you append the word "Wait" and let it carry on from there.

There are failure modes, and they are all failures of stopping:

| Failure | What happens |
|---|---|
| **Never stops** | The model loops or repeats and never emits the end-of-thinking token. Greedy decoding makes this more likely (see below). |
| **Stops too early** | It answers before it has really worked the problem out, so it gets the answer wrong. |
| **Overthinking** | It keeps checking an answer it already had, and uses up tokens (or talks itself out of a correct answer). |

### What the model does differently at each level: one question, three runs

Before looking at the machinery, here is what the difference looks like from the outside. Take the question **"What is 17 × 24?"** and imagine the same model answering at three settings. (These are illustrative transcripts, not real output, but they show the typical pattern.)

```text
LOW      →  408                                              about 10 tokens
            (answers almost directly; no checking)

MEDIUM   →  17 × 24 = 17 × 20 + 17 × 4 = 340 + 68 = 408.     about 60 tokens
            Check: 24 × 17 = 408. ✓  Answer: 408
            (works it out, does one check, then stops)

HIGH     →  Plan: split 24 into 20 + 4.                       about 250 tokens
            17 × 20 = 340. 17 × 4 = 68. 340 + 68 = 408.
            Check 1: 24 × 17 = 24 × 10 + 24 × 7 = 240 + 168 = 408 ✓
            Check 2: 17 × 24 ≈ 20 × 24 − 3 × 24 = 480 − 72 = 408 ✓
            Let me re-read the question: it asks for the product. ✓
            Answer: 408
            (works it out, then checks it by two different methods, then re-reads the question)
```

All three arrive at the same answer. What differs is **how much the model does after it has an answer**: how many times it double-checks, whether it tries a different method, whether it explores alternatives. That is the heart of reasoning effort: it changes how many checking and exploring steps the model is willing to do before it commits.

The level also interacts with how hard the problem is. Illustrative reasoning lengths:

| | Low | Medium | High |
|---|---:|---:|---:|
| **Easy question** (17 × 24) | ~10 tokens | ~60 | ~250 |
| **Hard question** (a multi-step puzzle) | ~400 | ~1,500 | ~6,000 |

Notice that the *hard question at low effort* (about 400 tokens) takes longer than the *easy question at high effort* (about 250). The level scales the amount of thinking, but the problem sets the baseline, which is why a setting never corresponds to a fixed number of tokens.

### The mechanism, step by step: how a label becomes more reasoning

Now to the question you asked: *what internal mechanism makes the model reason more?* The answer has two halves, and the first one is a surprise.

**First, what does not change:** nothing in the architecture. There are no extra layers, no bigger model, no special reasoning module that switches on. Each token still passes through the same N blocks (Chapter 8) with the same weights. As Part 1 explained, a Transformer does a fixed amount of work *per token*. So "reasoning more" can only mean one thing: **generating more tokens**, each one a further pass through the network, each one building on what came before. The question is only what makes the model write more of them.

**Second, how the setting changes what gets written.** The effort setting reaches the model as part of its input, as ordinary tokens (the serving layer may turn a setting into text or special tokens placed in the system message). From there it works through the mechanisms you already know:

```text
1. The effort label is in the context, at the start.
        │
        ▼
2. At EVERY later position, attention (Chapter 4) can read it,
   just as it reads any instruction in the prompt.
        │
        ▼
3. So at every position the hidden vector is nudged slightly,
   depending on the label.
        │
        ▼
4. So at every position the LM head's probabilities change slightly:
   P("Wait, let me double-check")  vs  P("</think>"  = stop and answer)
        │
        ▼
5. A small change at each decision point COMPOUNDS over the chain:
   many more checks → a much longer chain of thought.
```

So the label doesn't "turn up a dial" on the computation. It **steers the choice made at each step between "keep going" and "stop"**, and the cumulative effect of hundreds of such choices is a very different amount of reasoning. The same applies at the very start: whether the model begins by thinking at all (an opening thinking marker, or a direct answer) is itself just the first token's probability, and the label shifts it too.

### Why the model learned to respond to the label: the price of a token

Why would a model respond sensibly to such a label? Because it was **trained to**. One family of published methods (length-aware rewards, as in the L1 and BudgetThinker research) makes the idea very concrete. The reward for a reasoning attempt combines being right with the price of being long:

```text
reward  =  (1 if the answer is correct, else 0)   −   λ × (tokens used / 1000)
                                                       ▲
                                    the "price per token", set by the effort level
```

The effort level decides the **price of a token** (λ). Suppose two kinds of attempts at some problem: a **short** attempt (300 tokens) that is right 70% of the time, and a **long, careful** attempt (3,000 tokens, with checks) that is right 95% of the time. Which behaviour does the reward favour?

| Effort level | Price λ | Short attempt: 0.70 − λ×0.3 | Long attempt: 0.95 − λ×3 | Which wins? |
|---|---:|---:|---:|---|
| **Low** (tokens are expensive) | 0.5 | 0.55 | −0.55 | **Short**, by a mile |
| **Medium** | 0.1 | 0.67 | 0.65 | **About equal**: the balance point |
| **High** (tokens are cheap) | 0.02 | 0.69 | 0.89 | **Long**, clearly |

(The numbers are made up to show the principle.) Read the middle row: at the medium price, an extra round of checking is **worth its cost only when it buys enough accuracy**. That is exactly the in-between behaviour of a medium setting. At each point in its reasoning the model effectively asks, *"is another check worth its price?"*:

- Right after computing an answer that hasn't been checked, another check buys a lot (it could catch a mistake), so even at a medium price it continues.
- After two independent checks agree, a third buys almost nothing, so at a medium price it stops, whereas at a cheap (high-effort) price it may still do one more.

Different providers use different recipes (for example, supervised examples of reasoning at each length, or reinforcement learning with a price like this), and the exact recipes for commercial models are not public. But the behaviour that comes out is consistent with this picture.

### So how does "medium" know where to stop?

This is the heart of your question: *if I choose medium, how does the model know it has used more than the Low amount, but not as much as High, and that it should stop around here?*

The honest answer is: **it doesn't count.** There is no number stored inside the model like "medium = 800 tokens" that it compares itself against. Instead, three ingredients combine, and the first is the main one.

**Ingredient 1: stopping probabilities that depend on where it is in the reasoning.** The model reaches natural *checkpoints* in its reasoning, such as "I have a plan", "I have an answer", "my first check passed", "a second, independent check passed". At each checkpoint the model has a probability of stopping, and the effort level sets how high it is. Here is an illustration, using the 17 × 24 example. Each cell is the chance of stopping *at that checkpoint*, given the model got that far:

```text
Checkpoint:                  C1        C2          C3            C4              C5
                          plan made  answer     first check   second check    alternative
                                     computed   passed        passed          method agrees
Chance of stopping here:
   LOW                       5%       65%         90%           95%             99%
   MEDIUM                    2%       20%         65%           90%             97%
   HIGH                      1%        5%         25%           60%             90%
```

Because it is a chain of chances, the outcome is a **spread**, not a single point. Working it out, here is where each run would likely end, and the typical length:

| | Stops at C1 | C2 | C3 | C4 | C5 | Typical length |
|---|---:|---:|---:|---:|---:|---:|
| **Low** | 5% | 62% | 30% | 3% | ~0% | ~120 tokens |
| **Medium** | 2% | 20% | 51% | 25% | 3% | ~180 tokens |
| **High** | 1% | 5% | 24% | 42% | 25% | ~280 tokens |

This is how "in between" works. Medium is **not** "stop after exactly N tokens". It is "stop *after the first check, usually*, sometimes earlier, sometimes after a second one". Its runs overlap with Low's and with High's: a medium run can occasionally be as short as a low one, or as long as a high one. And it explains why **the same question at the same setting gives different lengths on different runs**: the stop decision is a probability, and the model's sampler (Chapter 12) rolls the dice at every step.

It also explains the difficulty effect from earlier. The stop probabilities depend on the **state of the reasoning**. On a hard problem, the model doesn't reach a "verified, confident" checkpoint quickly, so the stop probability stays low for longer, and even a medium run goes on for a long time.

**Ingredient 2: a rough sense of how much it has written.** The model can see its own output, so it has some sense of how long it has been going. The position information in the model (Chapter 3, and the rotary positions of Chapter 16) lets attention work out roughly *how far back* the start of the thinking is. Language models are poor at *exact* counting but decent at rough lengths. This matters most when the setting is given as a **length target**, as in the L1 research, where the prompt says how long to think and training rewards landing close to it. Models trained this way get *near* the target, not exactly on it.

**Ingredient 3: an explicit countdown, where the system provides one.** The most direct way for a model to know how much budget is left is to be *told*. We cover this next.

### When the budget is a number: countdown and control tokens

Some systems turn the budget into something the model can literally read. Two examples:

- **A countdown marker in the context.** In one provider's "task budget" feature, the server injects a running countdown into the conversation, showing how many tokens remain in the current task and updating as the model generates thinking, tool calls and output. The model sees it only as part of its context (the API response doesn't expose it), and it uses it to **pace itself and finish gracefully as the budget runs down**. The thinking naturally scales down as the budget depletes. It is advisory rather than a hard cap, and a budget that is clearly too small for the task can make the model scope the task down aggressively or stop early, because it sees from the countdown that it cannot finish.
- **Control tokens inserted at intervals.** The BudgetThinker research periodically inserts special tokens into the model's own output that tell it how much of its budget remains, and trains the model with supervised examples followed by reinforcement learning with a length-aware reward so that it adapts its reasoning to what is left and wraps up as the budget runs out.

In both cases the "internal mechanism" is nothing mysterious: **the remaining budget becomes text (or special tokens) in the context, and attention reads it like any other token**. When it is low, the stop-probability pattern shifts toward wrapping up. That is the same machinery as the effort label, with a number that changes along the way.

### Putting the mechanisms together: what is probably at work for each kind of setting

| What you control | What the model can "see" | How it decides where to stop |
|---|---|---|
| **A level (Low / Medium / High)** | A label in the context | Learned stopping behaviour: label-dependent stop probabilities at each checkpoint (Ingredient 1) |
| **A length target ("think for about N tokens")** | The target number in the prompt | Learned approximate length sense (Ingredient 2), trained with a length-aware reward |
| **A budget with a countdown** | A running "tokens remaining" value | Reads the remaining budget and wraps up as it shrinks (Ingredient 3) |
| **A thinking on/off switch** | A flag (`/think`, `/no_think`) | Chooses whether to open a thinking block at all |
| **A hard token limit** | Nothing (the model doesn't know) | It doesn't. The serving layer cuts it off (next section) |

### Variants you will meet in real products

The same ideas show up in several forms:

- **Thinking on or off.** The Qwen3 models (Chapter 21) have a switch in the chat template, `/think` or `/no_think`. In the "off" case the model still writes an empty thinking block and goes straight to the answer. It is a two-level version of the mechanism above, and the models were trained to handle both modes in one set of weights ("thinking-mode fusion").
- **A level written in the system prompt.** Some open models read the level from the system message (a line such as `Reasoning: high`, with low, medium and high), which is the label mechanism in its plainest form.
- **A length target.** The L1 model is prompted with the number of tokens to think for.
- **The model decides whether to think at all (adaptive thinking).** Some providers let the model decide, for each request, whether to think and for how long, with the effort level as a signal. At lower effort it may skip thinking entirely for an easy request, and at higher effort it thinks more readily and at greater length. The same provider says that even at a low setting the model still thinks on a sufficiently difficult problem, only less than it would at a higher setting. It also notes that, in a tool-using loop, most of the thinking tends to happen on the first step after new input, and that thinking per request tends to decrease as a conversation grows longer.
- **Effort that covers more than thinking.** In at least one provider's design, the effort setting affects *all* the output tokens: the explanations and the tool calls too. Lower effort means fewer and terser tool calls and less preamble. Higher effort means more tool calls, a plan explained before acting, and more detailed summaries afterwards. So "reasoning effort" is better understood as *how much work the model puts into the whole response*, of which thinking is one part.

### Hard limits: when the serving layer steps in

The behaviours above are *soft*: the model chooses. The serving layer can also enforce *hard* limits, and it is useful to see how they differ.

**Forcing the end of thinking.** In the thinking-budget mechanism of Qwen3, the serving code counts the reasoning tokens. When the count reaches the limit, it stops the model's thinking, **inserts a fixed sentence** telling the model that time is up and it has to answer based on the thinking so far, adds the end-of-thinking token, and lets the model carry on generating the answer. The model was never explicitly trained for this. The report says the ability emerges from training it to handle both thinking and non-thinking modes. Forcing can also run the other way: appending "Wait" to extend the thinking (budget forcing).

```text
SOFT STOP:   ... reasoning ... reasoning ... [</think>]  answer ...
                                              ▲
                             the model chose to stop

HARD STOP:   ... reasoning ... reasoning ... reasoning  ✂  [“time's up, answer now” + </think>]  answer ...
                                                         ▲
                                         the serving layer cut in; the model continues
                                         from the inserted text
```

**Token limits.** Every request also has a maximum number of output tokens, and **thinking counts toward it** (and is billed as output), even when you can't see the thinking. If the limit is reached mid-thought, there may be no visible answer at all. That is why settings that think a lot need a generous limit. Providers differ on thinking *budgets*: some treat a requested budget as a **target** rather than a strict cap, so the model may stop earlier, and the separate maximum length is the true hard ceiling.

### What you actually see change, setting by setting

Putting it together, here is what generally differs as you move up the settings. These are tendencies, not guarantees, and the exact numbers vary by model:

| | **Low** | **Medium** | **High** (and above) |
|---|---|---|---|
| **Reasoning length** | Short, or none for easy requests | Moderate | Long, up to many thousands of tokens |
| **Self-checking and alternatives** | Rare | Some | Frequent: verifies, backtracks, tries other approaches |
| **Tool calls and explanation** (where effort covers them) | Fewer, terser | Balanced | More calls, more explanation |
| **Time before the first visible answer word** | Short | Longer | Can be long, because thinking comes *before* the answer |
| **Cost** | Lowest (fewest output tokens) | Moderate | Highest |
| **Best for** | Simple lookups, formatting, routing, quick chat | Everyday tasks | Hard maths, code, multi-step planning |

Two practical points follow from this table. First, **latency is mostly thinking time**. The tokens come out at the same speed as always (Chapter 12), but the answer you wanted only starts after the reasoning has finished, so a high setting makes the first visible word arrive much later. Second, **more is not always better**. There are diminishing returns, and at the very top setting some simple or structured tasks only cost more, or even get worse through overthinking.

### Why sampling settings matter more for reasoning models

Chapter 12's sampling controls (temperature, top-p, top-k) interact with the stopping behaviour. Some reasoning-model families recommend moderate temperature and top-p (for one family, around 0.6 and 0.95) and explicitly warn **against greedy decoding**. The reason follows from the section on stopping: greedy always takes the single most likely token, and a long chain of thought can get caught in a loop of repeating itself. If the model never picks the end-of-thinking token, it never finishes. A little randomness lets it break out of loops, and lets it find the "stop" token when that token is a probable but not the single most probable choice.

### Choosing a setting in practice

- **Start from the default**, then try the next level up or down on your own tasks, rather than assuming the highest is best. Test with a sample of real questions.
- **Use low** for simple, well-specified work and for high-volume or latency-sensitive tasks. **Use high** for hard multi-step problems where correctness matters more than speed.
- **Give high settings room.** Set a generous maximum token limit so that the thinking does not use it all up before the answer begins.
- **Watch time-to-first-answer and cost, not only quality**, since both grow with thinking length.
- **Keep the setting stable inside one long conversation** if you rely on prompt caching, because with some providers, changing it invalidates the cached prefix (Chapter 12).
- **Match the setting to the problem**, ideally automatically: easy requests at low effort, hard ones at high (see "Routing" in Part 8).

### Putting the whole picture together

```text
                      TRAINING
                         │
            ┌────────────┴────────────┐
     Reasoning examples            Verifiers
            └────────────┬────────────┘
                         ▼
                 Reinforcement learning
                         ▼
                  Reasoning model
                         │
                      INFERENCE
                         ▼
       Question ──► Reasoning budget (Low / Medium / High)
                         ▼
                  Reasoning tokens ─► KV cache ─► Final answer
```

There is one more distinction here. **Training-time improvement** gives the model *better reasoning capability*. **Test-time compute** gives it *more opportunity to use that capability*. The strongest systems combine both.

---

## Part 7 — The costs: tokens, KV cache and latency

Reasoning is not free, and the cost follows directly from what you already know.

### Reasoning tokens are still tokens

They go through exactly the same pipeline as any other token:

```text
Token → Embedding → Transformer → KV cache → next token
```

So every reasoning token adds to the KV cache (Chapter 12). Using the 128 KiB per token of an 8B-class model:

```text
20 reasoning tokens    →  about 2.5 MiB of cache
2,000 reasoning tokens →  about 250 MiB
3 paths × 2,000        →  about 750 MiB
```

Add up the effects of higher effort:

```text
Higher reasoning effort → more tokens → more compute
                                      → more KV cache
                                      → more latency and memory
                                      → possibly a better answer
```

You are literally buying quality with **latency, GPU time, memory, energy and money**. This is why reasoning models create new serving and hardware challenges (Chapters 22 and 23): long outputs keep large caches alive for longer.

### "Do we really need to keep the reasoning tokens?"

A fair question. If reasoning is only an intermediate step, why store it, or its KV cache?

- **While the model is still generating**, it needs them. Producing `340 + 68 = 408` requires reading the earlier lines, so their Keys and Values must be there. Throw them away early, and the model loses the thread.
- **After the final answer**, the reasoning text and its cache need not be kept as permanent memory. The system can discard the cache, or keep only the question and the answer if the conversation continues. (Whether an application feeds earlier reasoning back in on later turns is a design choice.)
- It is also useful *during* the answer. A long, detailed final answer often benefits from the reasoning that preceded it, which is why the cache isn't dropped the moment the reasoning ends.

There is active research on **compressing reasoning**: summarizing or compressing the state, or reasoning in hidden vectors rather than in written tokens, so that a model doesn't carry thousands of tokens forward when a smaller representation would do. It is the same trade-off as in Chapter 12's "could we compress the cache?": exact context costs memory, compression risks losing detail, and it is not a solved problem.

### Another cost: overthinking

More thinking is not always better. A reasoning model can spend a lot of tokens on an easy question, and long chains can sometimes talk themselves into a wrong answer. Adaptive effort exists partly to avoid this.

---

## Part 8 — Combining reasoning with quantization and routing

### "Can reasoning make up for a quantized model?"

An intriguing idea: Chapter 14 showed quantization costs a little quality. If thinking longer improves answers, can extra reasoning compensate?

```text
FP16 model, answering directly                100  (illustrative)
Q4 model, answering directly                   97
Q4 model, with more reasoning/verification     99
```

(The numbers only illustrate the principle.) Yes, it can recover **some** of the loss: you are trading numerical precision for extra inference computation. But there is a ceiling, and it's an important one:

> **Reasoning helps a model use the capability it has. It cannot recreate knowledge that quantization (or a smaller size) destroyed.**

If the quantized model no longer reliably *knows* something, telling it to think harder doesn't bring the fact back. So: can reasoning recover some of the loss? Yes. Everything? No.

A second route is **distillation**: use a strong model to produce reasoning traces, then fine-tune a small model on them (as with R1's distilled variants). The small model inherits useful reasoning behaviour, and you can then quantize it and give it a thinking budget. The stack for a capable local model becomes: *better training + smaller model + quantization + reasoning + test-time compute.*

### Routing: spend effort where it pays

This leads to a practical way of thinking. Instead of "which is the strongest model?", ask: *what is the cheapest model that is good enough when given the right amount of reasoning, tools, retrieval and verification?*

```text
Easy request   →  small (quantized) model  →  low reasoning
Medium request →  medium model             →  medium reasoning
Hard request   →  strong model             →  high reasoning
```

That is the seed of **model routing**, which Chapter 24 takes up.

---

## Part 9 — Reasoning models, tools and agents

These terms get mixed up constantly. They are related but not the same, so here is the boundary:

```text
Normal LLM         Prompt → LLM → Answer

Reasoning model    Prompt → LLM → Reasoning → Final answer
                   (still ONE model generating tokens)

Reasoning + tools  the LLM can decide to call a calculator, web search or code,
                   get the result back, and continue

Agent              Goal → Plan → Act → Observe → Re-plan → ... → Result
                   (a persistent execution LOOP over many steps)
```

| System | Main capability |
|---|---|
| LLM | Generate |
| Reasoning model | Generate and reason |
| Tool-using LLM | Generate and use tools |
| Agent | Plan, execute, observe and adapt |

As soon as you add search, verifiers, memory and tools around the model, you move from a reasoning *model* to a reasoning *system*, and the boundary with an agent gets thin. But keep one line clear:

> **Reasoning is a capability of the model and the inference process. Agentic behaviour is a capability of the surrounding execution system.**

An agent doesn't have to use a reasoning model: it can wrap a normal LLM in a loop. And a reasoning model doesn't need an agent: a question can go in, reasoning happen, and an answer come out. They complement each other, and Chapter 24 covers the system side.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "Reasoning models have a different architecture." | Usually the same Transformer. They differ in training (RL with verifiers) and in how they are used (long reasoning at inference). |
| "There is a special reasoning engine inside." | Reasoning is ordinary next-token generation. The generated tokens become context that later tokens can use. |
| "The verifier runs every time you ask a question." | The verifier is a *training* tool, like the RLHF reward model. At inference there's usually just the model. |
| "Reasoning tokens are a new kind of token." | They are normal tokens, with a normal embedding, KV cache and cost. |
| "Low / Medium / High selects three different models." | It's the same weights under a different inference policy (budget, stopping, sampling and search). The exact mechanism varies by provider. |
| "A separate part of the model decides when to stop thinking." | There's no stop controller. Stopping is the model emitting the end-of-thinking token, chosen by the same softmax as any token, and learned through RL. |
| "The model counts its tokens to know when it has reached Medium." | There is no stored token number. Stopping is a probability at each checkpoint, shifted by the effort label, so Medium runs vary in length and overlap with Low and High. |
| "Medium means a fixed number of reasoning tokens." | The level scales the *amount* of checking and exploring, while the problem sets the baseline. A hard question at Low can take longer than an easy one at High. |
| "Extra reasoning comes from extra layers or a special reasoning module." | Nothing in the architecture changes. More reasoning means more tokens, each a normal forward pass through the same blocks. |
| "The effort setting only changes the length of the thinking." | It can also change whether the model thinks at all, how often it checks itself, and (in some designs) how many tool calls it makes and how much it explains. |
| "A thinking budget is something the model plans around." | Often it's enforced from outside: the serving layer cuts in, inserts a short "answer now" text and the end-of-thinking token, and the model continues from there. |
| "Greedy decoding is the safest setting for reasoning." | It can trap a long chain of thought in a loop that never reaches the end-of-thinking token. Reasoning models are usually run with moderate temperature. |
| "The thinking budget is a strict cap." | It's often a target. The model may stop early, and a separate maximum length is the hard limit. |
| "The visible reasoning is a full, faithful record of the model's thinking." | It's what the product chooses to show (raw, summarized or hidden), and it isn't guaranteed to explain why the model answered as it did. |
| "More thinking is always better." | It costs time, memory and money, and it can overthink easy problems. Effort should match difficulty. |
| "One verifier can judge everything." | Programmatic verifiers work for maths, code and puzzles. Elsewhere learned verifiers are used, and they can be exploited (reward hacking). |
| "Exploring several paths means waiting for each one in turn." | Tokens are sequential within a path, but separate paths can be batched in parallel, and shared prefixes can be cached. |
| "GRPO is a different model." | It's a training algorithm: PPO-style RL that uses a group of attempts as its baseline instead of a separate value model. |
| "Reasoning can restore everything lost to quantization." | It can recover some quality, but cannot bring back knowledge the compressed model has lost. |
| "A reasoning model is an agent." | A reasoning model is one model that thinks. An agent is a loop of planning, acting and observing around it. |

---

## Quick reference

```text
Why it works:  a Transformer does a fixed amount of work per token (N blocks);
               each written token becomes context → more sequential computation (a scratchpad)
Test-time compute:  spend more computation per problem at inference (depth and/or breadth)
               scaling axes: model size | training compute | test-time compute

Training:      RLVR = RL with VERIFIABLE rewards (correct = 1, wrong = 0)
               verifiers: programmatic (math, code) | learned (a judge model) | process (per step)
               outcome supervision (final answer) vs process supervision (each step)
               GRPO:  sample a group per question;  advantage = (reward − group mean) / group std
                      no value model → less memory than PPO
               DeepSeek-R1: R1-Zero (pure RL, emergent self-checking) → R1 (cold start, RL, SFT, RL)
                            → distill into small models
               verifier is discarded after training

Answer-time:   longer chain | self-consistency (majority vote) | best-of-N + verifier
               | search (Tree of Thoughts: branch → score → prune)
               sequential within a path, parallel across paths (batched; shared prefix cached)

Effort control: same model, different inference policy, in three layers:
                1 model (soft): effort label in the prompt, learned stopping ("Reasoning: high", /think, length target)
                2 serving (hard): token limits; forced "answer now" + end-of-thinking token; budget = target
                3 system: more samples, voting, verifiers, search
Stopping:       no controller: the model emits the end-of-thinking token (</think>), chosen by softmax
Mechanism:      label (text) in context → attention reads it at every step → hidden state nudged →
                P("Wait") vs P("</think>") shifts → compounds over the chain. NO architecture change;
                more reasoning = more forward passes (tokens).
Why it learned: reward = correct − λ × length; effort sets the PRICE λ of a token
                (low = expensive, high = cheap; medium = a check is worth it only if it buys enough accuracy)
Medium stops:   no stored token count. P(stop) at checkpoints (plan, answer, check 1, check 2 …) depends on
                the label and on the state of the reasoning → a spread of lengths that overlaps Low and High.
                Also: a rough length sense (positions; length-target training) and, where provided,
                an explicit countdown / control tokens in the context.
                RL teaches it: too early = wrong, never = no answer; effort label shifts P(stop) vs P("Wait")
                failures: never stops (loops, greedy), stops too early, overthinks
Settings:       low → short, few checks, fewer tool calls;  high → long, many checks, slow first word, costly
                latency ≈ thinking time;  avoid greedy decoding;  keep effort stable for cached conversations
Visible reasoning: product decision (raw / summary / hidden); not guaranteed faithful

Costs:         more tokens → more KV cache (128 KiB/token for an 8B-class model → 2,000 tokens ≈ 250 MiB)
               + latency, GPU time, energy, money;  overthinking is possible
Combos:        quantization + reasoning recovers SOME quality (not lost knowledge);
               distillation; routing: easy → small/low effort, hard → strong/high effort

Boundary:      reasoning = capability of the model/inference process;
               agent = capability of the execution system around it
```

---

## What's next

That completes the model itself: how it is built (Chapters 2 to 8), trained (Chapters 9 to 11), run, adapted and compressed (Chapters 12 to 14), redesigned (Chapters 15 to 19) and taught to reason (this chapter). Chapter 21 puts it all together by opening up two real model families, Qwen and DeepSeek, so you can see how these ideas (modern building blocks, attention variants, MoE and reasoning-focused training) combine in practice. After that, the handbook turns from the model to everything around it: serving it, scaling it, and building systems on top of it.
