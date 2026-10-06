# Chapter 24 — Production AI: Tool Calling, MCP, RAG, Agents and Reliability

## Introduction

For twenty-three chapters we have studied the model itself: how it reads tokens, how it is trained, adapted, compressed, redesigned and served. But a model on its own is not a product. Nobody uses a bare next-token predictor. They use a chat assistant that remembers the conversation, looks things up, runs code, and calls other services. All of that is built *around* the model, and it is mostly ordinary software engineering.

The idea to keep in mind, which Chapter 1 introduced and every chapter since has reinforced, is this:

> **The model only ever reads tokens and writes tokens.** Everything else a "smart" system does is software that decides *what tokens to put in*, and *what to do with the tokens that come out*.

Memory, tools, retrieval and agents sound like new capabilities of the model. They are not. They are ways of **filling the context window** before the model runs, and **acting on its output** afterwards. The model's loop never changes.

This chapter is deliberately a tour rather than a manual. It covers, in order:

1. **One message, end to end:** the whole stack in one picture.
2. **The context is the interface:** what goes into the prompt, and why it is costly.
3. **Tool calling:** how a text predictor "uses" a calculator, a search engine or a database.
4. **Memory:** working memory versus persistent memory.
5. **Retrieval-augmented generation (RAG):** giving the model knowledge it was never trained on.
6. **Agents:** a loop of planning, acting and observing.
7. **Routing and cost:** spending the right amount on each request.
8. **Reliability:** logging, evaluation and guardrails.

---

## Part 1 — One message, end to end

Let's trace a single user message through the entire system. Everything in the left column is something you have already learned. Everything in the right column is what this chapter adds.

```text
 INSIDE THE MODEL (Chapters 2 to 12)                  AROUND THE MODEL (this chapter)

                                                      You type a message
                                                              │
                                                              ▼
                                                      Application builds the CONTEXT
                                                      (system prompt + history + retrieved
                                                       documents + tool definitions)
                                                              │
 Tokenizer                    (Chapter 2)  ◄──────────────────┘
      ▼
 Embeddings + positions       (Chapter 3)
      ▼
 Transformer blocks × N       (Chapters 4 to 8, 16, 17, 18)
      │  attention reads the KV cache; a cached
      │  prefix can be reused                (Chapters 12, 22)
      ▼
 LM head → logits             (Chapter 8)
      ▼
 Sampling: temperature, top-p (Chapter 12)
      ▼
 One token ───────────────────────────────────────►  Streamed to you as it is produced
      │
      └── repeat until the model emits an end token,
          or a TOOL-CALL request                     ───►  Application runs the tool, adds the
                                                            result to the context, and the
                                                            loop runs again (Part 3)
                                                              │
                                                              ▼
                                                      Final answer shown to you
                                                              │
                                                              ▼
                                                      Application updates memory (Part 4)
```

Three things in this picture are worth underlining.

- **The application builds the context before the model ever runs.** What the model "knows" in a given moment is *exactly* what the application chose to include.
- **The model's output is either an answer or a request.** A model can end its turn with text for you, or with a structured request to use a tool. The application decides what happens next.
- **Everything is a loop.** Generation is a loop over tokens (Chapter 1). A tool-using system is a loop over *requests to the model*. An agent (Part 6) is a loop over *both*.

---

## Part 2 — The context is the interface

The model has no memory of its own between requests (Chapter 12): every request starts from whatever text is sent to it. So the context window is the *only* way anything gets to the model. A typical request assembles:

| Piece | What it is | Why it is there |
|---|---|---|
| **System prompt** | Standing instructions: role, tone, rules | Shapes behaviour for the whole conversation |
| **Conversation history** | Earlier user and assistant turns | The model can't recall them otherwise |
| **Retrieved knowledge** | Passages looked up for this question (Part 5) | Facts the model was never trained on |
| **Tool definitions** | Names and schemas of the tools it may call (Part 3) | So it knows what actions exist |
| **Tool results** | What the tools returned | Fresh information from the outside world |
| **Reasoning tokens** | The model's own thinking (Chapter 20) | Its scratchpad |

Three consequences follow, each of which connects back to something you already know.

**1. The context window is a budget.** Everything above has to fit (Chapter 12), and every token costs: compute during prefill, memory in the KV cache, and money. Adding "just in case" content is not free.

**2. Position matters.** Models do not use all positions equally well. Research on long contexts found that models tend to use information at the **beginning and end** of the context more reliably than information buried in the **middle**. So put the most important material where it will be used well, and don't assume a bigger window means everything inside is equally "seen".

**3. A stable prefix is cheap.** Recall prefix caching (Chapters 12 and 22): if the *start* of a prompt is identical to one already processed, its KV cache can be reused. So order the context from **most stable to most changeable**:

```text
[ system prompt ][ tool definitions ][ long reference document ][ history ][ new question ]
  ◄──────────── identical every request: cache hits ───────────►  ◄── changes ──►
```

If you put something that changes (like a timestamp) at the very start, you invalidate the cache for everything after it. Many providers also invalidate the cache when you change settings such as the reasoning level (Chapter 20).

---

## Part 3 — Tool calling: how a text predictor uses a calculator

A language model can't browse the web, query a database or run code. It only produces text. So how do assistants do these things? With a simple trick:

> **The model writes a structured *request* for a tool. The application runs the tool and tells the model the result.**

### The loop

```text
1. The application sends the model:  the conversation  +  a list of available tools
2. The model replies, not with an answer, but with a TOOL REQUEST:
        { "tool": "get_weather", "arguments": { "city": "Mumbai" } }
3. The application ──► checks the request and RUNS the tool itself
4. The application adds the tool's result to the context:
        { "tool_result": "31°C, humid" }
5. The model reads the result and continues, possibly calling another tool,
   or finally writing the answer:  "It's 31°C and humid in Mumbai."
```

The model **never runs anything**. It produces a piece of text shaped like a function call, and the application, which is ordinary code, does the real work. That has an important consequence: the application is the one that can check, limit, log and refuse.

### What the model is given

Each tool is described to the model, usually with a name, a plain-language description, and a schema for its arguments:

```text
Tool:        get_weather
Description: Get the current weather for a city.
Arguments:   { "city": string }          (required)
```

These definitions are just text that goes into the context (Part 2). The model decides *whether* a tool is needed and *which* one, from the descriptions and the conversation. So **clear descriptions matter a lot**: they are the model's only manual.

### How models learn to do this

Tool use is a skill acquired in fine-tuning (Chapter 13): models are trained on many examples of conversations that include well-formed tool requests and the results. At run time, systems often also constrain the output so that the request is **valid structured text** (valid JSON that matches the schema), by restricting which tokens the sampler may choose at each step (a use of Chapter 12's sampling).

### Practical points

- **Parallel calls.** A model can request several independent tools at once. The application runs them together and returns all the results.
- **Errors are just results.** If a tool fails, the error message goes back as a result and the model can try something else.
- **Trust nothing blindly.** The arguments come from a model, and the *content* the model reads may come from untrusted places (a web page, a document, an email). Text inside a tool result can contain instructions aimed at the model. This is called **prompt injection**, and it is the main security risk of tool-using systems. So: give tools the least privilege they need, validate arguments, require human confirmation for actions with side effects (sending, paying, deleting), and treat all tool output as *data*, not as commands.

### MCP: a standard plug

Every application used to wire up its tools its own way. The **Model Context Protocol (MCP)** is an open standard for connecting AI applications to external systems, and it is often described as a USB-C port for AI: one standard connector instead of a custom cable for every device. A tool provider writes an **MCP server** that exposes its tools (and also data "resources" and reusable prompts), and any MCP-aware application can connect to it. That turns an "N applications × M integrations" problem into "N + M".

```text
            ┌── MCP server: files ──┐
Application ┼── MCP server: database ┼──  standard protocol
 (the host) └── MCP server: search ──┘
```

Nothing about MCP changes how the *model* works. It standardizes how the *application* finds and calls tools.

---

## Part 4 — Memory: three different things that get confused

When people say an assistant "remembers", they may mean three very different things, and keeping them apart clears up a lot:

```text
PERSISTENT MEMORY    the conversation history, notes, facts and preferences, stored by the
(a database, files)  APPLICATION. Permanent until deleted. The model never sees it directly.
        │
        ▼   for each request, the application CHOOSES what to include
CONTEXT WINDOW       the tokens the model sees right now. Limited in size. Assembled fresh per request.
        │
        ▼   the model processes that context
KV CACHE             the Keys and Values for those tokens (Chapter 12). Temporary working
(GPU memory)         memory for this run. Not your chat history.
```

The KV cache is working memory, the context window is what the model is looking at, and persistent memory is what the *application* knows. Only the middle layer reaches the model.

### What to do when history outgrows the window

A long conversation will not fit forever. The usual options, each with a trade-off:

| Approach | How it works | Trade-off |
|---|---|---|
| **Truncate** | Drop the oldest turns | Simple, but forgets early details |
| **Summarize** | Replace old turns with a short summary | Saves space; the summary can lose detail |
| **Retrieve** | Store everything; pull in only the relevant pieces for this question (Part 5) | Scales well; depends on retrieval quality |
| **Structured memory** | Extract durable facts ("prefers metric units", "works on project X") and store them as notes | Compact and precise; needs care about what to save |
| **Start fresh** | New context, carrying forward only essentials | Cleanest, loses continuity |

Real systems combine them: recent turns verbatim, older turns summarized, durable facts stored, and relevant older material retrieved on demand.

Memory brings its own risks. Stored facts can be **stale** or wrong, they can contain **private** information that must be handled carefully, and they can be **poisoned**: if the system saves something from untrusted content, an attacker's text may come back in future conversations.

---

## Part 5 — RAG: giving the model knowledge it was never trained on

A model's knowledge is frozen at training time, it knows nothing about your private documents, and it sometimes produces confident but false statements (hallucination). **Retrieval-augmented generation (RAG)** addresses this by *looking things up first* and putting what it finds in the context:

```text
Question ──► find the most relevant passages in a knowledge base ──► put them in the prompt ──► model answers from them
```

RAG is the practical application of Part 2: instead of hoping the model remembers a fact, you put the fact in front of it. The original RAG work combined a generative model with a searchable index of documents, so that knowledge could be updated by changing the index, not retraining the model.

### How the search works: embeddings and similarity

How do we find the "relevant" passages? With embeddings (Chapter 3). A separate **embedding model**, usually an *encoder-only* model of the kind in Chapter 15, turns every passage into a vector, so that passages with similar meaning land close together. The question is turned into a vector the same way, and we find the nearest neighbours. "Closeness" is usually measured by **cosine similarity**, the angle between vectors. A tiny example with 2-dimensional vectors:

```text
Question: "How do I get my money back?"      q = [0.9, 0.1]

Passage A  "Refund policy"            [0.8, 0.2]    similarity with q = 0.99   ← closest
Passage C  "Returns and refunds"      [0.7, 0.3]    similarity with q = 0.96   ← close
Passage B  "Shipping times"           [0.2, 0.9]    similarity with q = 0.32
Passage D  "Office opening hours"     [0.1, 0.95]   similarity with q = 0.21

Take the top 2 (A and C) and put them in the prompt.
```

Notice that the question shares *no words* with "Refund policy", yet it still matches. That is the advantage of meaning-based search over keyword search.

### The pipeline

```text
INGEST (done ahead of time)
   Documents ──► split into chunks ──► embed each chunk ──► store vectors in an index

QUERY (done for each question)
   Question ──► embed ──► find nearest chunks ──► (re-rank) ──► insert into the prompt ──► model answers,
                                                                                         ideally citing sources
```

### The choices that decide quality

- **Chunking.** Too big, and a chunk contains a lot of irrelevant text and wastes the context. Too small, and a chunk loses the context needed to make sense. Chunks of a few hundred tokens, often with some overlap, are common.
- **How many to retrieve (top-k).** Too few misses the answer. Too many fills the window with noise (and the "lost in the middle" effect of Part 2 makes extra text actively harmful).
- **Hybrid search.** Pure meaning-based search can miss exact terms such as product codes or names. Many systems combine it with keyword search.
- **Re-ranking.** Retrieve a generous set of candidates cheaply, then use a stronger model to re-order them and keep the best few.
- **Instructing the model.** Tell it to answer from the supplied passages and to say so when they don't contain the answer.

### RAG, fine-tuning, or long context?

| | Best at | Limits |
|---|---|---|
| **RAG** | Facts that change or are private; citing sources; large knowledge bases | Quality depends on retrieval; adds latency and moving parts |
| **Fine-tuning** (Chapter 13) | Changing style, format and behaviour | Poor for injecting many facts; needs retraining to update |
| **Long context** | Small documents you can just paste in | Costly (Part 2); a bigger window doesn't mean everything is used well |

They are complements: fine-tune for behaviour, retrieve for facts.

### How RAG fails

Mostly in the *retrieval*: the right passage wasn't found, irrelevant passages were found, or the answer is split across chunks. Then the generation step: the model ignores the passages, or blends them with its own memory. The documents themselves can also contain **prompt injection** (Part 3). When debugging, always check first *what was retrieved*. Often the model did exactly what it was asked, with the wrong evidence.

---

## Part 6 — Agents: a loop of planning, acting and observing

Chapter 20 ended with a distinction, which we can now complete:

```text
LLM                Prompt → LLM → Answer
Reasoning model    Prompt → LLM → Reasoning → Answer
Tool-using LLM     the model can request tools; the application runs them
Agent              Goal → Plan → Act → Observe → Re-plan → … → Result
```

An **agent** is a system that pursues a goal over **many steps**, deciding at each step what to do next, using tools, and looking at the results. The core is a loop:

```text
          ┌─────────────────────────────────────────────┐
          ▼                                             │
   Goal + state ──► MODEL decides the next action       │
                          │                             │
              ┌───────────┴───────────┐                 │
              ▼                       ▼                 │
        "I'm finished"          a TOOL REQUEST          │
              │                       │                 │
              ▼                       ▼                 │
        Final result       Application runs the tool ───┘
                           (result added to the context)
```

A widely used pattern, **ReAct**, has the model interleave short reasoning with actions: *think about what is needed, act, observe the result, think again*. A tiny trace:

```text
Goal: "Find the cheapest of these three laptops that has 16 GB of RAM."
 Thought: I need the specs and prices for each.     Action: search("laptop A specs")
 Observation: 8 GB RAM, $720.
 Thought: A has only 8 GB, so rule it out.          Action: search("laptop B specs")
 Observation: 16 GB RAM, $850.
 Thought: B qualifies. Check C.                     Action: search("laptop C specs")
 Observation: 16 GB RAM, $790.
 Thought: C is cheaper than B. Answer: C.           → Final answer
```

An agent has four parts: **a model** (decides), **tools** (act), **memory** (carries state, Part 4), and **a controller** (the loop that enforces limits and decides when to stop). As Chapter 20 stressed, an agent doesn't need a reasoning model, and a reasoning model doesn't need to be an agent. They complement each other.

### Why agents are hard to make reliable

- **Errors compound.** If each step is right 95% of the time, ten steps in a row are right only about 60% of the time (0.95¹⁰ ≈ 0.60). At 99% per step it is about 90%. At 90% per step, only about 35%.
- **Loops and drift.** Agents can repeat the same failing action, or wander away from the goal.
- **Cost grows fast.** Every step resends the growing context (Part 7).
- **Actions have side effects.** Unlike text, a deleted file or a sent email can't be un-sent.

The standard defences are the controller's job: a **maximum number of steps**, a **budget**, a check after each step that the goal is still on track, **human approval** before irreversible actions, narrow tools rather than powerful general ones, and detailed logs. Where a task can be done as a fixed sequence of steps, a plain workflow is often more reliable than an agent that decides everything itself. Use an agent when the path really isn't known in advance.

Larger systems sometimes use several agents, each with a narrow role, coordinated by another. The same principles apply, with an extra cost: more model calls and more places for errors to creep in.

---

## Part 7 — Routing and cost: spend the right amount on each request

The earlier chapters gave you several dials that trade quality against cost, and production is about setting them per request:

| Dial | Chapter | Effect |
|---|---|---|
| **Which model** (size, quantized or not) | 14, 18 | Cheaper and faster vs more capable |
| **Reasoning effort** | 20 | More thinking tokens vs latency and cost |
| **Context size** | 12 | Fewer input tokens, lower cost |
| **Prompt caching** (a stable prefix) | 12, 22 | Cheaper repeated input |
| **Output limits** (max tokens, step limits) | 12, 20 | A hard cap on runaway spend |
| **Streaming** | 12 | Doesn't cut cost, but the user sees text sooner |

**Routing** means choosing among them per request:

```text
Easy request    →  small (maybe quantized) model   →  low reasoning effort
Medium request  →  mid-size model                  →  medium effort
Hard request    →  strongest model                 →  high effort
```

A router can be rules (by task type, length or user tier), a small classifier, or even a small model that judges difficulty. Add **fallbacks** too, so that if one provider fails or is slow, the request goes to another.

### What a request costs

A rough model is:

```text
cost  ≈  (input tokens × input price)  +  (output tokens × output price)
         with reasoning tokens counted as OUTPUT (Chapter 20), and cached input discounted
```

An illustrative example (made-up prices of $2 per million input tokens and $10 per million output tokens):

```text
Input:   8,000 tokens × $2 / 1M            = $0.016
Output:  1,500 tokens × $10 / 1M           = $0.015      (e.g. 1,000 of them reasoning)
                                    Total  ≈ $0.031 per call

With prompt caching, if 5,000 of the 8,000 input tokens are a cached prefix billed at a tenth of the price:
Input:   3,000 × $2/1M  +  5,000 × $0.20/1M = $0.007     (about 56% cheaper than $0.016)
```

### Why agent loops get expensive quickly

An agent calls the model many times, and each call **resends everything so far**. Suppose each step adds about 1,000 tokens to the context:

```text
Call 1:  1,000 input tokens
Call 2:  2,000
Call 3:  3,000
 ...
Call 10: 10,000        Total input over 10 calls:  1k + 2k + … + 10k = 55,000 tokens
                       (not 10,000)
```

The total grows with the *square* of the number of steps. Prompt caching helps a great deal here, since each call's context starts with the previous call's, which is exactly a stable prefix. Context trimming and summarization (Part 4) help too. This is why step and budget limits (Part 6) are not optional.

---

## Part 8 — Reliability: observing, testing and guarding

A system built around a probabilistic model needs more engineering around it than ordinary software, not less. Four practices cover most of it.

**Observability.** Record, for every request: the full assembled prompt, the tool calls and results, the retrieved passages, token counts (including reasoning tokens), latency (and time to the first token), and errors. When an answer is wrong, you can only find out *why* if you can see what the model actually saw. Tracing a whole multi-step run as one unit is especially valuable for agents.

**Evaluation.** Because outputs vary, you test on **sets of examples**, not single cases. Keep a collection of representative and tricky questions with expected behaviour, and re-run it whenever you change the model, the prompt, the retrieval, the tools, or the reasoning setting. Scoring can be exact (checking an answer, as with the verifiers of Chapter 20), or done by another model acting as a judge. Judge models are convenient, but they have biases and blind spots of their own, so spot-check them against humans. For RAG, evaluate retrieval (did the right passage come back?) separately from generation (did the answer follow the passages?).

**Guardrails.** Constraints that hold regardless of what the model says: input and output checks (for unsafe content or leaked sensitive data), least-privilege tools, confirmation before irreversible actions, limits on steps, spend and time, and treating everything the model reads as untrusted. These sit in the application, outside the model, so they can't be talked out of by a clever prompt.

**Resilience.** Timeouts, retries with back-off, fallbacks to another model or provider, and graceful degradation (for example, answering without retrieval if the search service is down). Version your prompts and model choices so that you can roll back, because a "small" prompt change can alter behaviour in ways you only notice later.

```text
A healthy loop:   build → run → LOG → EVALUATE on your example set → change one thing → repeat
```

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "The model uses tools itself." | It only writes a structured request. The application validates it, runs the tool, and returns the result. |
| "The model remembers past conversations." | It has no memory between requests. The application stores history and re-sends what it chooses. |
| "KV cache, context window and memory are the same." | KV cache = temporary GPU working memory; context window = what the model sees now; persistent memory = the application's storage. |
| "A bigger context window means the model uses everything in it well." | Information in the middle of a long context tends to be used less reliably, and every token costs. |
| "RAG gives the model new knowledge permanently." | It adds passages to *this request's* context. Nothing in the model's weights changes. |
| "RAG is a replacement for fine-tuning." | RAG is for facts; fine-tuning is for behaviour and style. They complement each other. |
| "If the answer is wrong, the model hallucinated." | In RAG systems the cause is often retrieval: the wrong or missing passage. Check what was retrieved. |
| "MCP makes the model smarter." | It standardizes how applications connect to tools and data. The model is unchanged. |
| "An agent is a smarter model." | An agent is a loop of model calls, tools and state around a model. |
| "An agent loop costs N times one call." | Each call resends the growing context, so total input grows roughly with the square of the number of steps. |
| "Guardrails can live in the prompt." | A prompt can be overridden or ignored. Hard limits belong in the application. |
| "Text returned by a tool is safe to follow." | It may contain instructions planted by an attacker (prompt injection). Treat it as data. |

---

## Quick reference

```text
Core idea:   the model only reads and writes tokens. Production = software that
             (1) decides what goes into the context, and (2) acts on what comes out.

Context:     system prompt + history + retrieved passages + tool definitions + tool results
             budget: tokens cost compute, KV cache and money; ends and beginnings are used best
             order stable → changeable for prefix-cache hits

Tool calling: model emits a structured request → APPLICATION validates and runs it →
              result added to context → model continues.  MCP = open standard plug for tools/data.
              Risks: prompt injection → least privilege, validate, confirm side effects.

Memory:      persistent (app database) → chosen into context window → KV cache (temporary).
             Overflow: truncate | summarize | retrieve | structured notes | fresh start.

RAG:         ingest: chunk → embed → index.   query: embed → nearest chunks (cosine) → (re-rank) → prompt.
             levers: chunk size, top-k, hybrid search, re-ranking.   Fails mostly in retrieval.
             RAG = facts;  fine-tuning = behaviour;  long context = small documents.

Agents:      goal → model decides → act (tool) → observe → repeat, with a CONTROLLER (max steps, budget).
             errors compound (0.95^10 ≈ 0.60);  cost grows ~ steps²;  prefer a fixed workflow if the path is known.

Routing:     choose model, reasoning effort, context size, caching, limits per request; add fallbacks.
             cost ≈ input×price_in + output×price_out (reasoning counts as output) − cache discount.

Reliability: log everything; evaluate on example sets (judge models have biases);
             guardrails in the application, not the prompt; timeouts, retries, fallbacks, versioning.
```

---

## What's next

That completes the main body of the handbook: you now have the whole path from a sentence to a system, from tokenization through the Transformer, training, serving and reasoning, up to the application built around the model. The **Appendix** makes everything concrete one last time. It follows a complete forward pass and a complete backward pass through a tiny GPT using small real numbers, so you can see every matrix multiplication, softmax and gradient that the chapters described in words.
