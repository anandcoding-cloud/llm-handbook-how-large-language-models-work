# Chapter 15 — The Transformer Family: Encoder, Decoder, Encoder–Decoder

## Introduction

Over the previous chapters we built a GPT-style model piece by piece (tokens, embeddings, attention, the FFN, residuals, LayerNorm, stacked blocks, an LM head, and a full training loop), then ran it (inference), adapted it (fine-tuning) and compressed it (quantization). There's a fact about all of that which we have deliberately held back until now:

> **Everything we built was not "the Transformer". It was specifically one half of it, the *decoder*.**

The original Transformer, introduced in the 2017 paper *"Attention Is All You Need"*, had **two** halves, an *encoder* and a *decoder*. Different families of models keep one half, or both. This chapter explains what those words mean, how BERT, GPT and T5 relate, and why one of them ended up dominating modern LLMs. The next two chapters then look at how today's decoder-only models have been improved: the building blocks they swapped out (Chapter 16) and the attention variants they use (Chapter 17). A tour of real models such as Qwen and DeepSeek comes later, in Chapter 21.

---

## Part 1 — The original Transformer, and what "encoder" and "decoder" mean

The 2017 paper was about **machine translation**. Think about what happens in your own head when you translate "The cat sat on the mat" into French. You don't start speaking French immediately. First you *understand* the sentence, something like `animal = cat, action = sitting, place = on the mat`. That internal understanding isn't English and isn't French. Then you *express* it in French: `Le chat est assis sur le tapis.` Two different jobs:

```text
Job 1:  English  →  meaning          (understand)
Job 2:  meaning  →  French           (express)
```

Those two jobs are exactly the **encoder** and the **decoder**:

```text
English → Encoder → meaning (a vector for every word) → Decoder → French
```

A word of caution about the names. "Encoder" does **not** mean compression, like encoding an MP3 or a ZIP file. It means converting language into vectors that capture meaning and context. You've already seen what these vectors are: the contextual representations from Chapters 4 to 8, the ones that know about grammar, meaning and relationships between words.

- **The encoder** answers: *"What does this text mean?"* It turns text into contextual vectors, and stops there. No words are produced.
- **The decoder** answers: *"Given this meaning (or the text so far), what should I write next?"*

---

## Part 2 — The same block, with a different attention mask

Here's the part many people find surprising: **inside, an encoder block and a decoder block are almost identical.** Both contain attention and an FFN, wrapped in residuals and LayerNorm. The major difference is the attention mask.

```text
Encoder attention                    Decoder attention (causal)

      A  B  C                              A  B  C
  A   ✓  ✓  ✓                          A   ✓  ✗  ✗
  B   ✓  ✓  ✓                          B   ✓  ✓  ✗
  C   ✓  ✓  ✓                          C   ✓  ✓  ✓

Every token sees every token         Each token sees only itself
                                     and what came before
```

Why the difference? An **encoder** receives a sentence that already exists in full. There's nothing wrong with "cat" looking at "sat", and in fact that's desirable, because context from the right helps understanding. A **decoder** is *writing*. When it is producing `The cat ...`, the word `sat` doesn't exist yet, so it can't be looked at. That's the causal mask you met in Chapter 4.

So now we can look back and see what we built: our GPT-style model (the one that Appendix A runs by hand, "TinyGPT") was **a decoder**. When someone says "GPT uses a Transformer", what they really mean is *"GPT uses the decoder half of the original Transformer."*

---

## Part 3 — Three branches from one tree

After 2017, researchers asked a natural question: *do we really need both halves?* Sometimes yes, sometimes no. That led to three branches:

```text
                      Original Transformer (2017)
                        Encoder  +  Decoder
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   Encoder only          Decoder only        Encoder + Decoder
   (BERT)                (GPT, Llama,        (T5, BART, Whisper)
                          Qwen, DeepSeek)
   "Understand"          "Generate"          "Read fully, then write"
```

| | Encoder-only (BERT) | Decoder-only (GPT) | Encoder–decoder (T5) |
|---|---|---|---|
| Attention | Sees both directions | Sees only the past | Encoder: both directions. Decoder: past, plus the encoder |
| Main job | Understanding | Generating | Turning one sequence into another |
| Training task | Fill in hidden words | Predict the next word | Reconstruct or transform text |
| Typical uses | Search, classification, embeddings | Chat, coding, writing, reasoning | Translation, summarization, rewriting |

Notice that none of these invented a radically new neural network. They changed *how the Transformer is used*.

---

## Part 4 — BERT: the encoder-only model

BERT's training task is a fill-in-the-blank game. Take `The cat sat on the mat` and hide a word:

```text
The cat [MASK] on the mat.
```

Predict the missing word. The crucial detail is that BERT can look **both left and right** to do it:

```text
"The cat"   →   [MASK]   ←   "on the mat"
```

This changes what the model is good at. Take `The animal didn't cross the road because it was tired.` To work out what `"it"` refers to, BERT can read the whole sentence, including `because it was tired`, which settles the question. Or take a search query like `Apple store near me`. Does `Apple` mean the fruit or the company? Reading the whole query makes it obvious. That's why BERT-style models became so important for search and language understanding.

BERT can't do what GPT does, though. If you give it `Once upon a time...`, it has no idea what to do. It was built to understand text that already exists, not to continue it.

### Two precise points about training

**1. GPT is not trained on masked text.** It's easy to mix these up, so be careful. BERT *changes the input*: roughly 15% of the words are replaced with a special `[MASK]` token. GPT sees the **original, unchanged sentence**, but the *attention mask* (Chapter 4) stops each position from seeing the future. Two different kinds of "mask":

```text
BERT:  hides WORDS in the input     →  "The cat [MASK] on the mat"
GPT:   hides the FUTURE via attention →  "The cat sat" fed in unchanged,
                                          but each position can't look ahead
```

**2. They learn from different amounts of each sentence.** In Chapter 9 we saw that a GPT-style model gets a learning signal at **every** position, because every position predicts its next word. BERT only learns from the roughly 15% of positions that were masked. That's one reason decoder-only models turned out to be so efficient to train at scale.

---

## Part 5 — Encoder-only models in practice: when the vector is the product

You might wonder: if an encoder only outputs vectors and not readable text, what use is it? The answer is: **sometimes the vector itself is the product.**

Give an encoder `"The cat sat on the mat."` and instead of a sentence, you get something like `[0.42, -1.83, 0.91, ..., 0.15]`, maybe 768 or 1024 numbers that capture the meaning of the whole text. Texts with similar meanings produce similar vectors. That makes encoders the engine behind:

- **Semantic search.** Imagine one million documents and the query `How do I install Python?` You can't ask a large generative model to read a million documents. Instead, an encoder converts every document into a vector in advance, converts the query into a vector, and finds the nearest document vectors. Search by meaning, not just keywords.
- **Classification:** spam or not, positive or negative review.
- **Re-ranking and matching:** deciding which of several candidate results best fits a question.

Well-known encoder-only models include BERT, RoBERTa, DeBERTa, and the embedding models built on the same idea (sentence-transformers, and families such as E5 and BGE). Newer, modernized encoders such as ModernBERT also exist. The pattern is the same: **text in, meaning vector out.** (We'll meet these vectors again when we get to retrieval in Chapter 24.)

---

## Part 6 — Encoder–decoder models and cross-attention

When a task is fundamentally **one sequence in, a different sequence out**, such as translation or summarization, the original two-half design shines. The question is how the two halves connect. After all, the decoder isn't reading English; it's reading the encoder's *understanding* of the English. The answer is a second kind of attention.

### The decoder has two attention layers

A decoder block in the original Transformer looks like this:

```text
Masked self-attention   ("what have I already written?")
        ↓
Cross-attention         ("which part of the input matters right now?")
        ↓
FFN
```

A GPT block has only the first and last. The original has one extra layer.

### How cross-attention works

Suppose the encoder has read `The black cat sat on the mat`, and the decoder has written `Le chat` so far. It must now choose the next French word. Should it write `noir` (black) or `assis` (sat)? To decide, it asks the encoder: *"I'm writing the next word. Which English words matter most right now?"*

That is attention again, but across two sequences. And here is the beautiful part: **the attention formula hasn't changed at all.** Only the *source* of Q, K and V changes:

| Type | Q comes from | K and V come from |
|---|---|---|
| Self-attention | The same sequence | The same sequence |
| Cross-attention | The decoder | The encoder |

```text
Decoder  ──►  Q  ─┐
                   ├──► Attention(Q, K, V) ──► decoder receives information from the encoder
Encoder  ──►  K,V ─┘
```

The decoder asks the questions, and the encoder holds the knowledge. One practical consequence: the score grid has one row per decoder position and one column per *encoder* position. With 5 French words written so far and a 7-word English sentence, it is 5 × 7, not a square. Self-attention (masked, inside the decoder) is the part that keeps the output fluent. Cross-attention is the part that keeps it faithful to the input.

### The whole encoder–decoder model, drawn out

Putting the pieces together, here is the complete encoder–decoder model. The encoder reads the input and produces one vector per input word. The decoder writes the output one word at a time. The line down the right-hand side shows how the encoder's output reaches the decoder: it is fed into the **cross-attention layer inside every decoder block**.

```text
         Input text: "The black cat sat"
                         │
                         ▼
┌────────────────────────────────────────────────┐
│ TOKENIZER + EMBEDDINGS                         │
│  token IDs → vectors, plus position            │
└────────────────────────────────────────────────┘
                         │
                         ▼
╔════════════════════════════════════════════════╗
║ ENCODER   (one block, repeated N times)        ║
║  Self-attention: every word sees every word    ║
║  Add & LayerNorm                               ║
║  FFN                                           ║
║  Add & LayerNorm                               ║
╚════════════════════════════════════════════════╝
                         │
                         ▼
    Encoder output: one vector per input word ──────────┐
   (this one output feeds EVERY decoder block)          │
                                                        │
     Decoder input so far: "<start> Le chat"            │
                         │                              │
                         ▼                              │
┌────────────────────────────────────────────────┐      │
│ TOKENIZER + EMBEDDINGS                         │      │
│  token IDs → vectors, plus position            │      │
└────────────────────────────────────────────────┘      │
                         │                              │
                         ▼                              │
╔════════════════════════════════════════════════╗      │
║ DECODER   (one block, repeated N times)        ║      │
║  Masked self-attention (earlier words only)    ║      │
║  Add & LayerNorm                               ║      │
║  Cross-attention:  Q from the decoder,         ║◄─────┘
║                    K and V from the encoder    ║
║  Add & LayerNorm                               ║
║  FFN                                           ║
║  Add & LayerNorm                               ║
╚════════════════════════════════════════════════╝
                         │
                         ▼
┌────────────────────────────────────────────────┐
│ LM HEAD + SOFTMAX                              │
│  probabilities for the next French word        │
└────────────────────────────────────────────────┘
                         │
                         ▼
Next word: "est"  (appended to the decoder input, then repeat)
```

A few things to notice in the picture:

- **The encoder runs once.** It reads the whole input in one go, with every word seeing every other word, and its output is then fixed for the rest of the task.
- **The decoder runs once per output word.** It looks like the GPT decoder from earlier chapters, plus the extra cross-attention layer in the middle of each block.
- **The same encoder output feeds every decoder block.** Each block's cross-attention builds its Queries from the decoder's own vectors, and takes its Keys and Values from that one encoder output.
- **Each sublayer keeps its residual connection and LayerNorm,** exactly as in the Transformer block of Chapter 7.

### Why GPT dropped cross-attention

By now you can probably answer this yourself. GPT has only **one** sequence, which keeps growing: `The → cat → sat → ...`. There's no separate "input language" to look at, so there's nothing for cross-attention to attend to. It would have no purpose.

### Where encoder–decoder models still shine

Models such as T5, BART and Whisper (speech recognition) keep both halves. T5 popularized a neat idea: express *every* task as text in, text out (for example `"translate English to German: ..."` or `"summarize: ..."`) and train one model on a fill-in-the-spans objective. They remain strong when:

- the input is long,
- the output is a **different** sequence,
- and the output should stay tightly grounded in the input.

Typical cases: translation, summarization, grammar correction, rewriting, and speech recognition.

---

## Part 7 — "If GPT can already translate, why do we need the other two?"

This is exactly the question the field asked around 2019 to 2021, and it's worth answering carefully. A big enough GPT *can* translate: ask `Translate to French: The cat sat on the mat.` and it replies correctly. So why bother with an encoder?

Because **GPT solves translation indirectly, while an encoder–decoder solves it directly.** Picture two students:

```text
Student A (GPT):             ONE process does everything at once:
                             read the request → understand English → recall French
                             → write French, all inside one growing sequence

Student B (Encoder–Decoder): TWO dedicated stages:
                             understand the English completely → then write the French,
                             consulting notes (cross-attention) whenever needed
```

Or think of translating a 500-page novel. The GPT way is to keep the original open next to you and keep re-reading it as you write each sentence. The encoder–decoder way is to read and understand a chapter first, take notes, and then translate from the notes. Both can work, but the second is more naturally suited to "input sequence in, output sequence out."

So why did decoder-only models win anyway? Researchers discovered something unexpected:

> **Next-token prediction is an incredibly general objective.**

Translation? Predict the next translated word. Summarization? Predict the next summary word. Question answering, coding, chat? Each is just *"continue this text"*. Instead of building a specialized architecture per task, people simply made decoder-only models larger, trained them on more text, and added instruction tuning and reinforcement learning (Chapter 13, and Chapter 20 for reasoning models). Economically, one model that can do everything beats maintaining a translation model, a summarization model, a coding model and a chat model separately.

That's why the open models whose designs are published, such as Llama, Qwen, DeepSeek and Gemma, are all decoder-only, and most leading chat models are understood to be as well. It isn't that encoders and encoder–decoders stopped working. They remain excellent at their specialties. It's that scaling the decoder-only recipe covered so many tasks well enough that it became the default.

A good mental model for the three:

```text
BERT:             Read → understand → stop.            Excellent reader, cannot write.
GPT:              Read → think while writing.           Excellent writer, very good reader.
Encoder–Decoder:  Read → understand completely → write. Excellent translator and rewriter.
```

---

## The complete picture

```text
                Same ingredients inside every block:
                attention + FFN + residuals + LayerNorm

 What differs between the families is only WHICH attention is used and HOW the model is trained:

 BERT        full (bidirectional) self-attention            trained to fill in masked words
 GPT         causal self-attention                          trained to predict the next word
 T5 / BART   encoder: full self-attention                   trained to reconstruct / transform text
             decoder: causal self-attention + cross-attention
```

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "An encoder compresses text, like ZIP." | It converts text into contextual vectors that capture meaning. "Encode" here means "turn language into vectors". |
| "GPT is trained on text with words masked out, like BERT." | GPT sees the original sentence. The *attention mask* blocks the future. BERT is the one that replaces words with `[MASK]`. |
| "Encoder and decoder blocks are completely different designs." | They are almost identical. The main difference is the attention mask, plus an extra cross-attention layer in an encoder–decoder's decoder. |
| "Encoder-only models are useless because they can't generate text." | Often the vector itself is the product: search, classification, matching, and embeddings all rely on them. |
| "Cross-attention is a new, different kind of attention math." | It is the same formula. Only the source of Q (decoder) and K, V (encoder) changes. |
| "Decoder-only models won because encoder–decoders stopped working." | Encoder–decoders still excel at translation, summarization and speech. Decoder-only models won because next-token prediction scaled into one model that handles almost everything. |

---

## Quick reference

```text
Encoder:   full attention, outputs contextual vectors.   "What does this mean?"
Decoder:   causal attention, writes one token at a time.  "What comes next?"

BERT        encoder-only      fill-in-the-blank    search, classification, embeddings
GPT/Llama   decoder-only      next-token           chat, code, writing, reasoning
T5/BART     encoder-decoder   text-to-text         translation, summarization, rewriting

Cross-attention:  Q from the decoder, K and V from the encoder.
                  Same formula as self-attention; only the source of Q/K/V changes.

BERT masks WORDS in the input.  GPT masks the FUTURE with the attention mask.
```

---

## What's next

Since decoder-only models are the ones that dominate, we'll focus on them from here. Chapter 16 starts with the building blocks that modern decoder-only models have swapped out for better ones: how positions are represented, how normalization is done, and what the FFN looks like. Chapter 17 then covers how attention itself has been reworked to stay affordable on long text.
