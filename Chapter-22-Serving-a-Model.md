# Chapter 22 — Serving a Model: GGUF, llama.cpp and vLLM

## Introduction

By now you can explain how a model is built (Chapters 2 to 8), trained (Chapters 9 to 11), run one token at a time (Chapter 12), adapted (Chapter 13) and compressed (Chapter 14), and you have seen how modern designs, from attention variants and Mixture of Experts to reasoning, build on that plain GPT. What we haven't asked is a very practical question:

> **You have a trained model. How do you actually get it running, for yourself on a laptop, and for thousands of users on a server?**

Three pieces of software answer it, and each one is a direct application of something you already know:

| Piece | What it is | The problem it solves |
|---|---|---|
| **GGUF** | A file format | How do you *package* a (quantized) model so it loads fast and is self-contained? |
| **llama.cpp** | An inference engine | How do you *run* that file efficiently on ordinary hardware? |
| **vLLM** | A serving engine | How do you *serve* one model to many users at once without wasting the GPU? |

Nothing in this chapter changes the model. There is no new attention, no new training. The Transformer from Chapters 2 to 8 is exactly what runs. This chapter is about **systems engineering around it**: files, memory, scheduling.

A note on where this chapter sits. Everything before it was about *the model*: its architecture and how it behaves. This chapter and the next are about **interfacing the model with its users**, and they are largely **model-independent**. The same ideas of packaging, memory and scheduling apply whether the model is a plain GPT, a Mixture of Experts or a multimodal model.

One framing to hold on to throughout:

> **llama.cpp is built to run a model efficiently. vLLM is built to serve that model efficiently to many users.**

---

## Part 1 — What is a "model", as a file?

From the computer's point of view, a model is not "an AI". It is just:

```text
Weights   +   Metadata
```

The **weights** are the huge collection of numbers from the earlier chapters: the embedding table, `W_Q`, `W_K`, `W_V`, `W_O`, the FFN matrices, the norm vectors, the LM head. The **metadata** is everything else the software needs to *use* those numbers: how many layers there are, how wide each one is, how long the context can be, and how text is turned into tokens.

### How models are stored for training

Models that come out of training, and that you download from Hugging Face, are usually stored in one of two formats:

- **PyTorch `.bin` files**: the older format, based on Python's `pickle`. Loading one can run arbitrary code, which is a security risk.
- **`.safetensors`**: the modern replacement. A safe, simple format: a small header describing each tensor, followed by the raw tensor bytes. It can also be memory-mapped (we'll see what that means in Part 3).

Both are excellent for training and for frameworks like PyTorch. But a safetensors model is spread across several files, and the information about the model lives next to the weights rather than inside them: the configuration is in a separate `config.json`, the tokenizer in other files, and so on.

### What an inference engine like llama.cpp wants

An engine that runs models on laptops and phones has different priorities:

- **One self-contained file**, so you can copy and share a model easily.
- **Fast loading**, even for a 30 GB model.
- **Quantized weights stored natively**, in the block formats from Chapter 14.
- **Everything needed to run it inside the file**: the tokenizer and the chat format included.

So the llama.cpp project created its own format: **GGUF**. A useful analogy: `.docx` is a good format for *editing* a document, and `.pdf` is a good format for *reading* it. Same content, different optimization. Safetensors is for training and sharing; GGUF is for running.

(GGUF replaced the project's earlier formats, GGML, GGMF and GGJT. The key improvement was extensibility: hyperparameters are stored as flexible key–value pairs, so new model types and new fields can be added without breaking old files.)

---

## Part 2 — Inside a GGUF file

A GGUF file has a simple, sequential layout:

```text
┌───────────────────────────────────────────────────────┐
│ HEADER                                                │
│   magic          "GGUF"                               │
│   version        3                                    │
│   tensor_count   how many tensors the file holds      │
│   kv_count       how many metadata entries            │
├───────────────────────────────────────────────────────┤
│ METADATA   (key → value pairs)                        │
│   general.architecture        = "llama"               │
│   llama.context_length        = 8192                  │
│   llama.block_count           = 32                    │
│   llama.embedding_length      = 4096                  │
│   llama.attention.head_count  = 32                    │
│   tokenizer.ggml.tokens       = [ ...every token... ] │
│   tokenizer.chat_template     = "..."                 │
├───────────────────────────────────────────────────────┤
│ TENSOR INFO   (one entry per tensor)                  │
│   name, shape, quantization type, offset in the file  │
├───────────────────────────────────────────────────────┤
│ padding to an alignment boundary (default 32 bytes)   │
├───────────────────────────────────────────────────────┤
│ TENSOR DATA   (all the weights, as raw bytes)         │
└───────────────────────────────────────────────────────┘
```

So GGUF is **not just weights**. It is everything needed to run the model. Let's look at each part.

### The header

Four small fields. The **magic** is the four bytes `G G U F`, which lets software recognize the file type immediately. Then a **version number** (currently 3), the **number of tensors**, and the **number of metadata entries**. That's all the program needs to know how to read the rest.

### The metadata

Metadata is a list of named values. Keys are hierarchical, written like `llama.context_length`. This is where the model's architecture is described, in exactly the terms you've learned:

| Metadata key (example) | What it says | Where we learned it |
|---|---|---|
| `general.architecture` | Which model family (`llama`, `qwen2`, …) | Chapter 15 |
| `llama.block_count` | Number of stacked Transformer blocks | Chapter 8 |
| `llama.embedding_length` | Width of each token vector, d | Chapter 3 |
| `llama.attention.head_count` | Number of query heads | Chapter 5 |
| `llama.attention.head_count_kv` | Number of K/V heads (GQA) | Chapter 17 |
| `llama.feed_forward_length` | Hidden width of the FFN | Chapters 6 and 16 |
| `llama.context_length` | Maximum context | Chapter 12 |
| `llama.rope.*` | RoPE settings | Chapter 16 |
| `tokenizer.ggml.tokens` and merges | The whole vocabulary and merge rules | Chapter 2 |
| `tokenizer.ggml.bos_token_id`, `eos_token_id` | Special tokens | Chapter 2 |
| `tokenizer.chat_template` | How to format "system / user / assistant" turns | Chapter 13 |

Notice that the **tokenizer travels inside the file**. With safetensors you need to keep track of separate tokenizer files; with GGUF, one file is the model, the vocabulary and the chat format.

### Tensor info and tensor data

For every tensor, the file stores its **name**, its **shape**, its **data type**, and its **offset** (where its bytes start inside the data section). The data type is where quantization enters: it can be `F32`, `F16`, or one of the quantized block formats (`Q8_0`, `Q4_K`, `Q6_K`, …) from Chapter 14. Different tensors in one file can use different types, which is how a "mixed" format like `Q4_K_M` is stored.

Tensors follow a standard naming scheme, so a Transformer's parts are easy to recognize:

| Tensor name | What it is | Chapter |
|---|---|---|
| `token_embd.weight` | The embedding table | 3 |
| `blk.N.attn_norm.weight` | Norm before attention in block N | 7, 16 |
| `blk.N.attn_q.weight` | `W_Q` | 4 |
| `blk.N.attn_k.weight` | `W_K` | 4 |
| `blk.N.attn_v.weight` | `W_V` | 4 |
| `blk.N.attn_output.weight` | `W_O` | 5 |
| `blk.N.ffn_norm.weight` | Norm before the FFN | 7, 16 |
| `blk.N.ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight` | The SwiGLU FFN | 16 |
| `output_norm.weight` | The final norm | 8 |
| `output.weight` | The LM head | 8 |

Reading a GGUF file is therefore like reading the table of contents of the Transformer you've been building. (A Mixture-of-Experts model, Chapter 18, simply has many extra per-expert FFN tensors in each block, which is why its file is large even though only a few experts are used per token.) Here is what it looks like for an 8-billion-parameter Llama-style model (32 blocks, `d = 4096`, 32 query heads and 8 K/V heads of size 128, FFN width 14,336, vocabulary 128,256):

```text
token_embd.weight          128,256 × 4,096     (the embedding table)

for each of the 32 blocks:
  attn_q.weight             4,096 × 4,096
  attn_k.weight             4,096 × 1,024      (8 K/V heads × 128: GQA makes K smaller)
  attn_v.weight             4,096 × 1,024
  attn_output.weight        4,096 × 4,096
  ffn_gate.weight           4,096 × 14,336
  ffn_up.weight             4,096 × 14,336
  ffn_down.weight          14,336 × 4,096
  attn_norm.weight, ffn_norm.weight    4,096 each

output_norm.weight          4,096
output.weight              128,256 × 4,096     (the LM head)
```

That is 9 tensors per block, so `32 × 9 + 3 = 291` tensors in total (a few models add a handful of extras). Adding up the matrices gives about 8.0 billion parameters, and the K and V matrices are visibly four times narrower than `W_Q`, which is GQA, from Chapter 17, written in a file. The small norm vectors are usually kept in `F32` even in a 4-bit file, since they are tiny and sensitive.

### Alignment

Between the tensor info and the data, the file is padded so that every tensor starts at a multiple of 32 bytes (by default). This isn't wasteful. It lets the CPU and GPU read the data in the aligned chunks they are fastest at, and it is what makes the next trick possible.

---

## Part 3 — Memory mapping: why a 30 GB model starts quickly

### The slow way

A simple program loads a model like this:

```text
Disk  →  read the whole file into RAM  →  copy it into the program's own memory  →  ready
```

For a 30 GB model, that means reading 30 GB and copying 30 GB before the first token. It is slow, and it needs enough RAM to hold *two* copies for a while.

### Memory mapping

Operating systems offer a better tool, **memory mapping** (`mmap`). Instead of reading the file, the program asks the operating system to *map the file into its address space*, as if the whole file were already sitting in memory:

```text
Disk
  ↓
The OS maps the file into the program's memory (instantly, nothing is read yet)
  ↓
When the program touches a part of a tensor, the OS loads just that part ("page fault")
  ↓
Pages stay cached in RAM, and are dropped by the OS when memory gets tight
```

The weights are loaded **on demand**, a small piece at a time, by the operating system. There is no second copy, start-up is fast, and if several processes load the same model file they can **share** the same cached pages. It works so well because a Transformer's weights are read-only: nothing ever writes to them, and the alignment from Part 2 means each tensor can be used straight from the mapped file.

### The honest caveats

- **The first tokens can be slow.** If the model is on a slow disk and hasn't been read yet, the first pass through the model triggers many page loads. Once the pages are cached, it's fast.
- **If RAM is too small, the OS will drop and re-read pages**, and generation becomes painfully slow. Mapping doesn't make a model *fit*; it makes loading *lazy*.
- Flags let you choose: llama.cpp maps the file by default, `--no-mmap` loads it normally, and `--mlock` asks the OS to keep the model locked in RAM so it can't be paged out.

### GPU offload: layers split between VRAM and RAM

If you have a GPU, the engine can place some or all of the **layers** in GPU memory (the `-ngl` option sets how many). A model too big for your GPU can still run with, say, 24 layers on the GPU and 8 on the CPU: slower than all-GPU, but much faster than all-CPU. Because Chapter 14's quantization shrinks every layer, more layers fit.

---

## Part 4 — From Hugging Face to GGUF: what conversion does

You will often start with a model released as safetensors and need a GGUF. The process has two steps:

```text
Hugging Face model            convert_hf_to_gguf.py           llama-quantize
 (safetensors + config   ─────────────────────────────►  model-F16.gguf  ──────►  model-Q4_K_M.gguf
  + tokenizer files)        (repack, no precision loss)    (16-bit)             (apply Chapter 14)
```

1. **Convert.** A script reads the safetensors tensors and the config and tokenizer files, renames the tensors to the standard GGUF names (`model.layers.7.self_attn.q_proj.weight` becomes `blk.7.attn_q.weight`), rearranges weights into the layout llama.cpp expects, writes the config into metadata keys, and embeds the tokenizer. The result is a 16-bit GGUF with no loss of precision.
2. **Quantize** (optional). A second tool compresses the weights into a quantized type, applying everything from Chapter 14: block-wise scales, mixed precision, and so on.

Three things are *not* touched. Conversion does not retrain the model, it does not change what the model computes, and (apart from the optional quantization step) it does not modify the weights. It repackages them. In practice you can also just download a ready-made GGUF, since many people publish models in several quantization levels, which is why the filenames from Chapter 14 (`Q4_K_M.gguf` and friends) are so common.

> **GGUF in one sentence:** an inference-optimized file format that packages the model's weights, tokenizer, metadata and quantization information so that engines like llama.cpp can load and run the model efficiently.

---

## Part 5 — llama.cpp: running the Transformer you already know

**llama.cpp** is an inference engine written in C and C++, built on a small tensor library called **ggml**. It has no Python and no heavy dependencies, which is why it runs almost anywhere: laptops, phones, Raspberry Pis, servers. It has **backends** for CPUs (using vector instructions such as AVX and ARM NEON), for NVIDIA GPUs (CUDA), for Apple GPUs (Metal), and for others (Vulkan and more), and it can split a model across them.

### What happens at start-up

When you launch it with a model:

```text
llama-cli -m model.gguf
```

it does roughly this:

```text
Open the GGUF file
        ↓
Read the metadata            (architecture, layer count, widths, context length)
        ↓
Load the tokenizer           (vocabulary, merge rules, chat template)
        ↓
Memory-map the weights       (Part 3), offloading some layers to the GPU if asked
        ↓
Allocate the KV cache        (sized from the context length you chose, Chapter 12)
        ↓
Wait for a prompt
```

Notice what *isn't* there: no training, no weight changes. Start-up only prepares the weights for fast inference. The KV cache is reserved up front for the context length you ask for (the `-c` option), using exactly the size formula from Chapter 12, so memory use is predictable.

### What happens when you type a prompt

Take `What is AI?`. The whole run is the pipeline you've been studying, with each step labelled by its chapter:

```text
"What is AI?"
     │
     ▼  Tokenizer (Chapter 2)               text → token IDs, e.g. [3923, 374, 15592, 30] (illustrative)
     ▼  Embedding lookup (Chapter 3)        each ID → a vector
     ▼  Transformer blocks × N              (Chapters 4 to 7, in the modern form of Chapter 16)
     │      attention uses Q, K, V;  the K and V of every token go into the KV cache (Chapter 12)
     ▼  LM head (Chapter 8)                 → logits
     ▼  Sampling (Chapter 12)               temperature, top-k, top-p → one token
     │
     └──► the new token is fed back in; the KV cache is reused, so only the NEW token
          goes through the expensive part. Repeat until the end-of-text token.
```

The first pass over the whole prompt is the **prefill** (compute-heavy). Every pass after that is a **decode** step (memory-bound). Chapter 12 explained why, and llama.cpp's benchmarking tool reports them as two separate numbers: prompt-processing speed and token-generation speed.

Two things make it fast:

- **It runs quantized weights directly.** As Chapter 14 described, the matrix multiplications use kernels that unpack the 4-bit blocks and multiply in one go, rather than converting everything back to 16-bit first.
- **It uses all the hardware you give it:** vector instructions on the CPU, layers offloaded to the GPU, and a FlashAttention-style attention kernel (Chapter 17) where supported.

### The important realization

> **llama.cpp does not invent a new algorithm. It executes the Transformer from Chapters 2 to 8, very efficiently.**

Everything you spent time learning is exactly what runs. It is *"the loop of Chapter 1"* implemented in highly optimized C.

### Beyond the command line: llama-server

llama.cpp also includes an HTTP server, **llama-server**, that speaks an **OpenAI-compatible API** (chat completions, embeddings and so on), so an application can use a local model through the same kind of requests it would send to a hosted one. It supports several parallel "slots" so more than one conversation can be handled at once, continuous batching (which you'll meet below), and prompt caching, which reuses the cached prefix of a prompt it has seen before (the prefix caching idea from Chapter 12). The context size is shared among the slots.

### What llama.cpp is best at

Running **one model for one person (or a few)** on whatever hardware you have, especially with quantized weights, CPUs, mixed CPU/GPU, or Apple silicon. It is the engine behind many desktop apps and personal assistants. But the design target changes when you have *hundreds* of users hitting a large GPU. That is vLLM's territory.

---

## Part 6 — Many users, one GPU: why batching matters

Picture a server with two users. User A asks for a poem. User B asks for an explanation of AI. A simple engine handles them like this:

```text
User A  ██████████████
                        User B  ██████████████
```

One after another. User B waits for all of A's tokens, and the GPU is mostly idle between and within steps. That is wasteful, and we can see exactly why from what you already know.

### Why batching is nearly free

Recall from Chapters 12 and 14 that the **decode** phase is **memory-bound**: for each new token the GPU has to *read all the model's weights* from memory, but does very little arithmetic with them. Now, here's the key observation: **reading the weights once can serve many requests at the same time.** If ten users each need their next token, we can stack their vectors into one bigger matrix and multiply by the weights in one pass.

An idealized example. A 7B model in 16-bit is about 14 GB. On a GPU that reads roughly 1 TB per second, one decode step costs at least `14 GB / 1000 GB/s ≈ 14 ms` just to read the weights.

```text
Batch of 1:    one step ≈ 14 ms  →   ~70 tokens/s total
Batch of 32:   one step ≈ 14 ms  →   32 tokens per step  →  ~2,300 tokens/s total
```

(The ceiling isn't reached in practice, because the KV caches also have to be read and the arithmetic grows, but the shape is the point.) **Batching multiplies throughput almost for free**, because GPUs love big matrix multiplications. A single request is a small, poorly used workload; thirty requests together are one large, efficient one.

So a serving engine's job is to keep the GPU's batch as full as possible, all the time. That raises two problems, which are exactly what vLLM is famous for solving.

---

## Part 7 — Continuous batching: keep every slot busy

### The problem with static batching

The first idea is **static batching**: collect, say, four requests, process them together, and start the next batch only when *all four* are done. But requests don't take the same time. One user wants a one-line answer; another wants an essay. Here is each request's slot over time (`█` = working, `░` = idle, waiting for the others):

```text
STATIC BATCHING          step:  1 2 3 4 5 6 7 8 9 10

slot 1   request A             █ █ █ █ ░ ░ ░ ░ ░ ░     A finished at step 4, slot idle
slot 2   request B             █ █ █ █ █ █ █ █ █ █
slot 3   request C             █ █ █ █ █ █ ░ ░ ░ ░
slot 4   request D             █ █ █ ░ ░ ░ ░ ░ ░ ░

new requests E, F, G wait until the whole batch ends        busy: 23 of 40 slot-steps (57%)
```

The whole batch moves at the speed of its slowest member, short requests leave idle slots behind them, and new users have to wait.

### The fix: continuous (iteration-level) batching

**Continuous batching** makes the scheduling decision *at every single decoding step* rather than once per batch. As soon as a request finishes, its slot is handed to a waiting request, immediately:

```text
CONTINUOUS BATCHING      step:  1 2 3 4 5 6 7 8 9 10

slot 1   A, then E             A A A A E E E E E E     E takes over the moment A finishes
slot 2   B                     B B B B B B B B B B
slot 3   C, then F             C C C C C C F F F F
slot 4   D, then G             D D D G G G G G G G

                                                        busy: 40 of 40 slot-steps (100%)
```

The GPU stays full, short requests finish quickly instead of waiting for long ones, and new requests start almost immediately. Think of a restaurant that seats a new party the moment any table frees up, instead of waiting for every table to finish before seating anyone.

A related detail worth knowing: processing a long new prompt (a big prefill) in one go could stall everyone else's generation. Modern engines handle this by splitting long prompts into chunks (**chunked prefill**) and mixing them into the same steps as other people's decoding.

---

## Part 8 — The memory problem: every user has a KV cache

Continuous batching creates a new problem. Each request has its own **KV cache** (Chapter 12): request A's cache, request B's, and so on, each one growing token by token. How much memory is that? The vLLM paper gives a concrete example: for a 13-billion-parameter model (OPT-13B), the cache costs about **800 KB per token**. A request that grows to 2,048 tokens therefore needs about **1.6 GB** of KV cache, for one request. (A modern model with grouped-query attention is much smaller per token, for instance 128 KiB for Llama-3-8B, as Chapter 12 showed, but with hundreds of requests the total still dominates GPU memory.)

### Why the simple approach wastes memory

The obvious way to store a request's cache is as **one contiguous block of memory**. But at the start, you don't know how long the answer will be. So the engine has to reserve space for the **maximum possible length**, say 2,048 tokens, even if the request ends after 50. This causes three kinds of waste:

```text
Reserved but unused     space held for tokens that haven't been generated yet
Internal fragmentation  space reserved for the maximum, but the request ends early
External fragmentation  gaps left between differently-sized blocks of memory
```

The paper measured it: in existing systems, **only about 20% to 38% of the KV-cache memory actually held token data.** The rest was waste. And since memory limits how many requests fit in a batch, and batch size decides throughput, wasted memory directly means wasted speed.

---

## Part 9 — PagedAttention: the cache as pages of memory

The vLLM authors noticed that this problem had been solved decades ago, by **operating systems**. A computer doesn't give each program one giant contiguous slab of RAM. It hands out small, fixed-size **pages**, which don't have to sit next to each other, and keeps a **page table** that maps the program's logical view to the physical pages.

**PagedAttention** applies exactly the same idea to the KV cache:

- The cache of each request is split into small **blocks**, each holding the Keys and Values of a fixed number of tokens (16 by default in the paper).
- Blocks are allocated **only as the request grows**: a new block is claimed when the previous one fills up.
- Blocks don't have to be contiguous in GPU memory. A **block table** records which physical block holds each part of the request's cache.

```text
Request A's cache, as the request sees it (logical blocks):

   [ logical 0 ]  [ logical 1 ]  [ logical 2 ]
        │              │              │
        ▼              ▼              ▼         block table of request A
        7              1              3         (logical block → physical block)

GPU memory, as it really is (physical blocks, in any order):

   physical:   0    1    2    3    4    5    6    7    8    9
             ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
             │ B0 │ A1 │free│ A2 │ B1 │free│free│ A0 │free│free│
             └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
```

Request A's three logical blocks live in physical blocks 7, 1 and 3, scattered. Request B's two live in 0 and 4. Neither needed a contiguous slab. The attention computation looks up the block table to find the Keys and Values it needs.

### Why it solves the problem

- **No big reservation.** A request only ever holds the blocks it has actually filled, so the only waste is the unused part of its *last* block (at most 15 tokens' worth).
- **No external fragmentation.** Any free block can be used by anyone.
- **Instant reuse.** When request A finishes, its blocks go straight back to the free pool for the next request.

So far, "like an operating system's memory manager". It also unlocks things a contiguous cache couldn't do.

### Sharing blocks between requests

Suppose one prompt is used to generate several different answers at once (parallel sampling), or many users start with the *same long system prompt*. Those requests have **identical** Keys and Values for the shared part. With paging, they can point their block tables at the **same physical blocks**, storing the shared prefix **once**:

```text
Request 1 block table:   [ P0 ] [ P1 ] [ X ]          X = request 1's own new block
Request 2 block table:   [ P0 ] [ P1 ] [ Y ]          Y = request 2's own new block
                           ▲      ▲
                           └──────┴── the same physical blocks, stored once
```

Each block keeps a **reference count**, the number of requests using it. If a request needs to *write* into a shared block (for example, the two answers diverge), the engine first makes a private copy of that block and writes into the copy. That is **copy-on-write**, borrowed straight from operating systems too.

The same idea extends across requests that arrive at different times. This is **automatic prefix caching**: each full block is identified by a hash of its tokens (and the blocks before it), so when a new request starts with text that has already been computed, such as a shared system prompt, the engine finds the existing blocks and skips recomputing them. Unused cached blocks are evicted when memory runs short, least recently used first. This is the engine-side machinery behind the "prefix caching" idea we met in Chapter 12.

### When memory runs out

Even with paging, a busy server can run out of blocks. The engine then **preempts** some requests, pausing them to free their blocks, and resumes them later. There are two ways to get the cache back: **swap** the blocks out to CPU memory and copy them back in later, or simply **recompute** the cache when the request is rescheduled (a fresh prefill). Either way, nobody's answer changes.

### The result

The paper reported **2× to 4× higher throughput** than the best existing systems at the same latency, with the biggest gains for longer sequences, larger models and more complex decoding. Not because the model got faster, but because wasted memory became useful batch capacity.

---

## Part 10 — vLLM as a whole

**vLLM** is the serving engine built around these ideas. Here is the loop that runs continuously on the server:

```text
 Incoming requests (many users)
        │
        ▼
 ┌───────────────┐      ┌──────────────────────────────┐
 │ WAITING QUEUE │ ───► │ SCHEDULER                    │
 └───────────────┘      │  every step, decides:        │
        ▲               │   • which requests run now   │
        │               │   • who gets which blocks    │
 preempted requests ◄── │   • who must be paused       │
                        └──────────────┬───────────────┘
                                       │ the current batch
                                       ▼
                        ┌──────────────────────────────┐
                        │ BLOCK MANAGER                │
                        │  free pool, block tables,    │
                        │  reference counts, sharing   │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │ GPU(s): one forward pass for │
                        │ the whole batch (PagedAttn.) │
                        └──────────────┬───────────────┘
                                       │ one new token per running request
                                       ▼
                         stream tokens back to each user
```

Its pillars are the ones you've just seen:

- **Continuous batching**: the batch is rebuilt every step, so slots never sit idle.
- **PagedAttention**: efficient, shareable, non-contiguous KV-cache memory.
- **Prefix caching**: shared prompts are computed once.
- **High GPU utilization**: the consequence of the three above.

On top of that it provides an **OpenAI-compatible HTTP server**, supports quantized models, and can spread a model across several GPUs with tensor and pipeline parallelism (the subject of the next chapter).

---

## Part 11 — llama.cpp or vLLM?

The difference is the design target, not what the model computes: in both cases it is the same Transformer. (llama.cpp's server also has parallel slots and continuous batching, so it can serve several users. It simply isn't built around squeezing the last bit of throughput from a large GPU.)

| | **llama.cpp** | **vLLM** |
|---|---|---|
| Built for | Running a model efficiently, anywhere | Serving a model to many users, efficiently |
| Typical hardware | Laptops, desktops, phones, CPUs, Apple silicon, small GPUs | Data-centre GPUs |
| Model format | GGUF (quantized, self-contained) | Hugging Face-style weights (safetensors), including GPTQ, AWQ and FP8 quantizations (GGUF loading exists but is not its main path) |
| Typical users | One person, or a few | Many concurrent users |
| Strength | Portability, low memory, easy setup | Throughput, memory management at scale |
| Typical use | Desktop apps, local assistants, edge devices | API servers, production deployments |

An application that sits on top of a model doesn't need to care which one is behind it: both expose OpenAI-style HTTP endpoints, so either can be one "model provider" among several.

> **One sentence to remember:** llama.cpp is optimized to run a model efficiently. vLLM is optimized to serve that model efficiently to many users.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "A model file contains only the weights." | A GGUF file also holds the architecture description, the tokenizer, the chat template and the quantization metadata. |
| "Converting to GGUF changes or retrains the model." | Conversion only repackages (and optionally quantizes) the weights. The model computes the same thing. |
| "llama.cpp can't load safetensors because it's a bad format." | Safetensors is fine for training and sharing. GGUF is a different optimization: one self-contained file with native quantized blocks. |
| "Memory mapping makes a big model fit in a small RAM." | It makes loading lazy and cheap. If the model doesn't fit, pages get dropped and re-read, and generation becomes very slow. |
| "llama.cpp uses a different algorithm to run models." | It runs the standard Transformer, just implemented very efficiently, with quantized kernels. |
| "llama.cpp only works for one user at a time." | llama-server supports parallel slots and continuous batching. It just isn't tuned for very large-scale GPU serving. |
| "Batching users together slows each one down a lot." | Decode is memory-bound, so reading the weights once serves the whole batch. Throughput rises sharply for a small cost per user. |
| "Static and continuous batching are the same thing." | Static batching waits for the slowest request in a batch. Continuous batching refills freed slots at every step. |
| "PagedAttention changes the attention math." | It changes only where the cached Keys and Values are stored. The computed attention is the same. |
| "vLLM is faster because it uses a faster model." | The same model. The gain comes from not wasting KV-cache memory, which allows larger batches. |
| "The KV cache is stored contiguously per request." | In a naive engine, yes (and it wastes memory). In vLLM it is split into fixed-size blocks that can be anywhere. |

---

## Quick reference

```text
Model file  =  weights + metadata
safetensors  training/sharing format (tensors only; config and tokenizer in other files)
GGUF         inference format: header | metadata | tensor info | (padding) | tensor data
             metadata includes architecture, tokenizer, chat template; tensors can be mixed-quantized
Tensor names token_embd | blk.N.attn_{q,k,v,output} | blk.N.ffn_{gate,up,down} | *_norm | output
mmap         map the file into memory; the OS loads pages on demand; no second copy
Convert      HF → convert script → F16 GGUF → llama-quantize → e.g. Q4_K_M

llama.cpp    C/C++ engine on ggml; CPU/CUDA/Metal/Vulkan; layer offload (-ngl);
             quantized kernels; KV cache preallocated for the context size; llama-server (OpenAI API)
             Run:  tokenize → embed → blocks (+KV cache) → LM head → sample → repeat

Batching     decode is memory-bound → one read of the weights serves the whole batch
Static       whole batch waits for the slowest request          (e.g. 57% busy)
Continuous   refill finished slots every step                   (e.g. 100% busy)

Naive KV cache   contiguous, reserved for max length → only ~20–38% holds real tokens
PagedAttention   fixed-size blocks (16 tokens), block table, allocate on demand
                 shared blocks + copy-on-write; prefix caching; preempt by swap or recompute
vLLM             scheduler + block manager + continuous batching + PagedAttention
                 (paper: 2–4× throughput)

llama.cpp = run a model efficiently      vLLM = serve a model to many users efficiently
```

---

## What's next

We now know how a model is packaged, run on one machine, and served to many users. Two natural limits remain. Generation still produces **one token per forward pass**, however clever the batching. And a big enough model **doesn't fit on one GPU** at all. Chapter 23 tackles both: **speculative decoding**, where a small model drafts several tokens and the large model checks them in a single pass, and **parallelism**, the ways a model (and its data) can be split across many GPUs. It also steps back to look at the hardware those GPUs sit on, and why the industry is building chips designed specifically for LLMs.
