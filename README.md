# The LLM Handbook: How Large Language Models Work, from Tokenization to Production

> A free, plain-Markdown guide to how large language models work: tokenization, embeddings, attention, the Transformer, training, fine-tuning (LoRA, RLHF, DPO), quantization, Mixture of Experts, reasoning models, serving (llama.cpp, vLLM) and production AI. It teaches by derivation: **Why, How, What**.

**Start here:** [Chapter 1: The Big Picture](Chapter-01-The-Big-Picture.md) · [Contents](#contents) · [Learn with an AI assistant](#learning-with-an-ai-assistant) · [Appendix A: TinyGPT by Hand](Appendix-A-TinyGPT-by-Hand.md) · [References](References.md)

**Prefer a PDF?** [Download the full handbook as one PDF](https://github.com/anandcoding-cloud/llm-handbook-how-large-language-models-work/raw/main/pdf/LLM-Handbook-How-Large-Language-Models-Work.pdf) (298 pages, with a clickable table of contents). It is generated from these Markdown files.

---

## What this handbook is

A guide to how large language models work: how they are built, trained, adapted, compressed, served and used. It is written for readers who want to understand *why* things are the way they are, without needing a research background. If you can follow a simple diagram and a little arithmetic, you can follow this.

It is **not** a tour of the Transformer architecture, listing each part with its formula. It is closer to a guided reconstruction. You meet each part of an LLM at the moment you need it, because the previous step left a problem unsolved.

---

## How it teaches: Why, then How, then What

Most material shows the finished architecture: here are the parts, here is what each one does. This handbook goes the other way. It **derives** the model, one problem at a time. For every new piece, it asks the same four questions before moving on:

1. **What problem do we have?**
2. **Why is what we have so far not enough?**
3. **How does the new piece solve it?**
4. **What does it produce?** (concretely, with numbers)

The *why* comes first, and it gets the most space. The *how* and the *what* then feel like consequences, not rules to memorize.

Here is an example from Chapter 4 (attention):

| Question | Answer |
|---|---|
| **What problem?** | The meaning of a word depends on the words around it ("it" in "The cat drank the milk because *it* was thirsty"). |
| **Why isn't the previous step enough?** | An embedding is the same vector for a word in every sentence. It knows nothing about its neighbours. |
| **How is it solved?** | Each word asks a question (a query), compares it with what every other word offers (keys), and blends their information (values). |
| **What does it produce?** | A new, context-aware vector for each word. The chapter computes it with real numbers. |

The whole first half of the handbook is one chain of such problems, each created by the solution before it:

```text
Computers need numbers, not text            →  Tokenization              (Chapter 2)
Token IDs are meaningless addresses         →  Embeddings                (Chapter 3)
Embeddings know meaning, not context        →  Attention                 (Chapter 4)
One attention pattern is not enough         →  Multi-head attention      (Chapter 5)
Gathering information is not thinking       →  The feed-forward network  (Chapter 6)
Layers must not lose the input or blow up   →  Residuals and LayerNorm   (Chapter 7)
One block is not deep enough                →  Stacking blocks, the LM head (Chapter 8)
The random weights know nothing yet         →  Loss, gradients, AdamW    (Chapters 9 to 11)
```

From Chapter 12 on, the same method is applied to real engineering problems. Generating text recomputes everything, so we get the KV cache (12). A base model only continues text, so we fine-tune it (13). The weights are too big, so we quantize them (14). Attention costs too much on long text (17). Every token pays for the whole model, so we split the FFN into experts (18). One pass is not enough for hard problems, so models learn to reason (20). One GPU serves one user, so serving systems batch (22). Each technique is the answer to one specific, expensive problem.

---

## A habit used throughout: fixed, learned or computed?

Whenever a new component appears, the handbook says what kind of thing it is. Almost every confusing moment in learning LLMs comes from not knowing which of these three it is:

| Kind | Meaning | Examples |
|---|---|---|
| **Fixed by the architect** | Chosen before training, never trained | Embedding size, number of layers, number of heads, the GELU formula |
| **Learned during training** | Starts random, shaped by backpropagation | The embedding table, `W_Q`, `W_K`, `W_V`, `W_O`, the FFN weights |
| **Computed on every forward pass** | Built fresh from the input | Token IDs, Q, K, V, attention scores, the next-word probabilities |

When you wonder "where did this number come from?", the answer is always one of these three. Chapter 3 introduces the idea, and the chapters that follow use it.

---

## What you will find in every chapter

- **The problem first.** Each chapter opens with what was left unsolved, and why that matters.
- **An analogy, then the math.** Intuition comes before any formula, and the formula is introduced only when the intuition is solid.
- **Small worked examples.** Real numbers small enough to follow by hand. Appendix A runs one complete training step on a tiny GPT with every number visible.
- **The engineering view.** Tensor shapes, memory, cost, and how the idea shows up in real systems such as llama.cpp and vLLM.
- **A running diagram.** Chapters 2 to 8 each add one piece to a single picture ("The architecture so far"), so that by Chapter 8 you hold the whole model.
- **Common misconceptions.** These come from real questions asked while learning, not from a list of generic myths.
- **A quick reference and a pointer to the next chapter.**

### How this differs from a typical textbook or course

- **Derived, not catalogued.** A textbook lists components. Here each one earns its place by solving a problem the previous one created.
- **Real questions are answered.** Questions that learners actually get stuck on have their own sections: *Why can't we just compare the raw embeddings?* (Chapter 4), *Why not just one giant block?* (Chapter 8), *Why do we cache only Keys and Values, and can't the cache be compressed?* (Chapter 12). Most books skip questions like these.
- **The order follows understanding, not the usual table of contents.** For example, the feed-forward network comes before residual connections and LayerNorm, because the need for those only becomes visible once the block has two halves. Modern attention variants come after inference and fine-tuning, because their purpose (a smaller KV cache) only makes sense once you have seen the cache. Serving comes near the end, because it applies to any model.
- **Nothing is hidden.** Every number in a worked example comes from the step before it, and Appendix A shows the whole chain, including the backward pass.
- **Theory and systems in one thread.** The same document that explains attention also explains why a 70B model needs several GPUs, and what a `Q4_K_M` file is.

---

## How to read it

- **In order the first time.** Each chapter builds on the one before it.
- **As a reference afterwards.** Jump straight to the chapter you need. Each one opens with what it covers and why it matters.
- **Numbers are examples unless stated otherwise.** Many worked examples use small, made-up numbers to show a principle, and they say so. Figures taken from papers, model cards or documentation are listed in [References](References.md).
- **The first eleven chapters follow the Why → How → What pattern most closely.** The later chapters apply it to larger, more engineering-heavy topics, so they lean more on explanation and comparison tables.

## Learning with an AI assistant

These files are plain Markdown, so you can give them to an AI assistant (ChatGPT, Claude or similar) and use it as a tutor. The whole handbook is about 140,000 tokens, which is more than many assistants can hold at once, so load the Introduction plus the chapter you are on. Use the Markdown files rather than the PDF for this: they are smaller and cleaner for an AI to read. [llms.txt](llms.txt) is a one-page topic index, and [AGENTS.md](AGENTS.md) tells an assistant how to use the material.

A prompt to start with (attach this file and the chapter):

```text
Use the attached handbook as your main source. Teach me this chapter in its
own style: Why, then How, then What. Go one idea at a time, use its examples,
and ask me a short question after each section to check I understood.
For anything newer than October 2026, or about a specific model, price or
version, search the web and tell me which parts come from the handbook and
which from the web.
```

---

## Contents

### Build a GPT
1. [The Big Picture](Chapter-01-The-Big-Picture.md): an LLM as a next-token predictor in a loop
2. [Tokenization](Chapter-02-Tokenization.md): turning text into numbers
3. [Embeddings](Chapter-03-Embeddings.md): turning numbers into meaning, and adding position
4. [Attention](Chapter-04-Attention.md): queries, keys, values and the causal mask
5. [Multi-Head Attention](Chapter-05-Multi-Head-Attention.md): many attention patterns at once
6. [The Feed-Forward Network](Chapter-06-The-Feed-Forward-Network.md): processing what attention gathered
7. [The Transformer Block](Chapter-07-The-Transformer-Block.md): residual connections and LayerNorm
8. [Stacking Blocks into GPT](Chapter-08-Stacking-Blocks-into-GPT.md): the full model, from text to a next-word probability

### Train it
9. [Training I: Learning from Text and Measuring Error](Chapter-09-Training-Data-and-Loss.md)
10. [Training II: Gradients and Backpropagation](Chapter-10-Gradients-and-Backpropagation.md)
11. [Training III: Gradient Descent, AdamW and the Training Loop](Chapter-11-Gradient-Descent-and-AdamW.md)

### Run it, adapt it, shrink it
12. [Inference](Chapter-12-Inference.md): the KV cache and sampling
13. [Fine-tuning](Chapter-13-Fine-tuning.md): SFT, LoRA, RLHF and DPO
14. [Quantization](Chapter-14-Quantization.md): storing the same knowledge in fewer bits

### How modern models are built
15. [The Transformer Family](Chapter-15-The-Transformer-Family.md): encoder, decoder, encoder–decoder
16. [Modern Building Blocks](Chapter-16-Modern-Building-Blocks.md): RoPE, RMSNorm, SwiGLU
17. [Attention Variants](Chapter-17-Attention-Variants.md): MQA, GQA, MLA, sliding window, sparse, hybrid, FlashAttention
18. [Mixture of Experts](Chapter-18-Mixture-of-Experts.md): a huge model that uses only part of itself
19. [Multimodal Models](Chapter-19-Multimodal-Models.md): letting a language model see and hear

### How models reason, and real models
20. [Reasoning Models](Chapter-20-Reasoning-Models.md): spending compute on thinking
21. [Inside Modern Models: Qwen and DeepSeek](Chapter-21-Inside-Modern-Models-Qwen-and-DeepSeek.md)

### Interfacing the model with its users
22. [Serving a Model](Chapter-22-Serving-a-Model.md): GGUF, llama.cpp and vLLM
23. [Speed and Scale](Chapter-23-Speed-and-Scale.md): speculative decoding, parallelism and hardware

### In production
24. [Production AI](Chapter-24-Production-AI.md): tools, memory, retrieval, agents, routing and reliability

### Appendix and sources
- [Appendix A: TinyGPT by Hand](Appendix-A-TinyGPT-by-Hand.md): one complete training step with every number visible. The program that produced the numbers is included: [`Appendix-A-tinygpt.js`](Appendix-A-tinygpt.js) (run it with `node Appendix-A-tinygpt.js`).
- [References](References.md): papers, documentation, model cards and the study material behind the handbook.

## A note on how current it is

The first fourteen chapters describe ideas that change slowly. Chapters 16 to 23 describe a fast-moving area: model names, sizes, defaults and prices in particular may be out of date by the time you read them, so treat specific figures as a snapshot and the ideas as the lasting part. The References list the sources to check.
