# Chapter 19 — Multimodal Models: Letting a Language Model See and Hear

## Introduction

Everything in this handbook so far has been about text. A GPT reads tokens and writes tokens. But people want to show a model a photo, a screenshot or a chart, play it a recording, or hand it a video, and ask questions about them. How can a language model, which only understands vectors made from text tokens, deal with a *picture*?

The surprising answer is that **the language model itself barely changes**. Its core loop is still exactly the one from Chapter 1:

```text
Context  →  Transformer  →  LM head  →  next token
```

What changes is what gets put into the context. The recurring idea of this chapter is:

> **The Transformer doesn't need to understand everything directly. We can build an *encoder* that converts another kind of input (an image, a sound) into vectors the model can use, and place those vectors beside the text.**

This chapter follows one image on its journey into a language model:

1. **The problem**: a picture is just a grid of numbers, not tokens.
2. **The vision encoder**: cutting the image into patches and turning each into a vector.
3. **Teaching an encoder to understand pictures**: the CLIP idea.
4. **The projection**: translating those vectors into the language model's own space.
5. **Two ways to plug it in**: visual tokens in the sequence, or cross-attention.
6. **The cost**: how many tokens a picture is worth.
7. **How the combined model is trained**: how two models that grew up separately learn to understand each other.
8. **Audio, video, and what the model can produce.**

You'll also revisit two ideas from earlier chapters in a practical setting: the **encoder** and **cross-attention** from Chapter 15, and the **loss mask** from Chapter 13.

---

## Part 1 — The problem: pixels are not tokens

A text model sees this:

```text
Text  →  Tokenizer  →  Token IDs  →  Embeddings  →  Transformer  →  Text
```

Now suppose we give it an image plus a question. To the computer, an image is just numbers: a grid of pixels, each with red, green and blue values.

```text
one pixel:  [255, 120, 83]
a 224 × 224 colour image:  224 × 224 × 3  =  150,528 numbers
```

The Transformer has no idea what to do with 150,528 raw numbers. And we can't just treat every pixel as a token. A 224 × 224 image would be 50,176 tokens, and attention cost grows with the square of the length (Chapter 17). We need a smarter way to turn an image into a **short sequence of meaningful vectors**.

The plan, in three steps:

```text
Image  →  Vision encoder  →  Projection  →  vectors that look like text-token vectors  →  LLM
```

---

## Part 2 — The vision encoder: an image as a sequence of patches

### Why patches?

The trick behind the **Vision Transformer (ViT)**, whose paper is titled "An Image Is Worth 16×16 Words", is to cut the image into small square **patches** and treat each patch like a word. For a 224 × 224 image with 16 × 16 patches:

```text
┌────┬────┬────┬────┬ ... ┐
│ P1 │ P2 │ P3 │ P4 │     │      14 patches across × 14 patches down
├────┼────┼────┼────┼ ... ┤      = 196 patches
│ P15│ P16│ P17│ P18│     │
├────┼────┼────┼────┼ ... ┤      each patch is 16 × 16 pixels × 3 colours
│ ...│ ...│ ...│ ...│     │      = 768 numbers
└────┴────┴────┴────┴ ... ┘
```

Instead of 50,176 pixels, we now have 196 patches. This is the image's equivalent of the tokenizer: it chops the input into pieces.

### Turning a patch into a vector

Each patch is *flattened* into a list of numbers and multiplied by a learned matrix to become a vector of the model's width. This is the image's equivalent of the embedding lookup of Chapter 3. A tiny example with a 4 × 4 grayscale image and 2 × 2 patches:

```text
Image (grayscale pixel values)           Patches (2 × 2), each flattened

   10   20 │  30   40                     Patch 1 = [10, 20, 50, 60]
   50   60 │  70   80                     Patch 2 = [30, 40, 70, 80]
  ─────────┼─────────                     Patch 3 = [90, 100, 130, 140]
   90  100 │ 110  120                     Patch 4 = [110, 120, 150, 160]
  130  140 │ 150  160

Learned projection W (4 → 2 numbers), for illustration:
        [0.1  0  ]
   W =  [0    0.1]          Patch 1 · W = [10·0.1 + 50·0.1,  20·0.1 + 60·0.1] = [6, 8]
        [0.1  0  ]
        [0    0.1]
```

In a real model, `W` is a large learned matrix that maps each 768-number patch to a vector of hundreds or thousands of numbers, and its values are learned, not hand-picked.

### Position matters here too

Just like words in a sentence, patches have an order: the patch at the top-left is not the same as the one at the bottom-right. So, exactly as in Chapter 3, a **position embedding** is added to each patch vector so that the model knows *where* in the image the patch came from.

### The encoder itself

The sequence of patch vectors then passes through ordinary **Transformer blocks**: attention, FFN, residuals and normalization, exactly the block from Chapter 7. This is an **encoder** in the sense of Chapter 15: each patch can attend to *every other patch* (there is no causal mask, since there is no "next patch" to predict). So a patch showing a dog's ear can learn from patches showing the rest of the dog.

```text
Patch vectors (+ position)  →  Transformer blocks (every patch sees every patch)  →  N output vectors
                                                                                    (one per patch, now "aware" of context)
```

The result is a set of **visual tokens**: one rich vector for each patch.

```text
Image  →  Patch 1 → vector
          Patch 2 → vector
          Patch 3 → vector
          ...
          Patch N → vector          (like Text → Tokens → vectors)
```

---

## Part 3 — Teaching an encoder to understand pictures: the CLIP idea

A vision encoder with random weights would produce meaningless vectors. How does it learn what a "dog" looks like? One of the most influential answers is **CLIP**, which trains an image encoder and a text encoder **together** on about 400 million image–caption pairs from the internet. Its task is beautifully simple: *predict which caption goes with which image.*

Take a batch of three pairs. Encode every image and every caption, then compute a score for every possible image–caption pairing, giving a grid:

```text
                       caption 1         caption 2        caption 3
                      "a dog on        "a red car      "a bowl of
                       a beach"         at night"        soup"
   image 1 (dog)     ┌──────────────────────────────────────────────┐
   image 2 (car)     │   HIGH           low              low        │   ← the diagonal is
   image 3 (soup)    │   low            HIGH             low        │     the true pairs
                     │   low            low              HIGH       │
                     └──────────────────────────────────────────────┘
```

Training pushes the **diagonal scores up** (each picture should match its own caption) and **everything else down**. With millions of examples, the image encoder learns to produce vectors that land close to the vectors of the words describing them. That is the essential achievement: a shared meaning space where a photo of a dog and the words "a dog on a beach" end up nearby.

This is why a CLIP-style vision encoder is the common starting point for multimodal models. By the time it is plugged into a language model, it already knows how to turn pictures into vectors that capture *what is in them*. What it does not yet know is how to speak the particular language of *this* language model. That is the next step.

---

## Part 4 — The projection: translating into the language model's space

The vision encoder produces vectors in its own space, with its own width and its own meaning. The language model expects vectors in *its* space, of the same kind as the token embeddings of Chapter 3. A small **projection layer** (often a linear layer or a small MLP) translates one into the other:

```text
Image  →  Vision encoder  →  visual vectors  →  PROJECTION  →  vectors in the LLM's space
```

After the projection, the picture's vectors look to the language model just like word vectors. They can sit in the same sequence as the text:

```text
Image  →  Vision encoder  →  Projection  →  ┌─────────────────────┐
                                            │ visual tokens       │
"What is the dog doing?"  →  Tokenizer  →   │ + text tokens       │  →  LLM  →  Text
                                            └─────────────────────┘
```

### A full example

You upload a picture of a dog and ask *"What is the dog doing?"* The model receives one sequence:

```text
[visual token 1] [visual token 2] ... [visual token N]   What  is  the  dog  doing  ?
└─────────── the image, as vectors ────────────┘          └──────── the question, as vectors ───┘
```

The LLM processes all of it with its ordinary Transformer blocks and attention, and the text tokens can attend to the visual tokens. Then it generates, one token at a time as always:

```text
"The"  "dog"  "is"  "running"  "."
```

Nothing about the language model's own mechanism changed. It just has some extra vectors in its context.

---

## Part 5 — Two ways to plug the image in

The picture above, with the visual tokens placed in the sequence, is one of two common approaches. The second one uses the **cross-attention** of Chapter 15.

### Approach A: visual tokens go into the sequence

```text
Image → Vision encoder → Projection → visual tokens ┐
                                                    ├─►  one sequence  →  LLM
Text  → Tokenizer → text tokens ────────────────────┘
```

The language model treats image tokens like extra words. Its architecture is unchanged, and ordinary self-attention lets the text look at the image. This is simple and popular: open models such as LLaVA and many others follow it.

### Approach B: cross-attention

Recall the difference between the two kinds of attention (Chapter 15):

```text
Self-attention:    Q ← text,   K ← text,    V ← text      (the text looks at itself)
Cross-attention:   Q ← text,   K ← image,   V ← image     (the text looks at the image)
```

Here, the text representation asks: *"which parts of the image are relevant to me?"* For example, with the question "What colour is the car?":

```text
Question: "What colour is the car?"
                  │  Q
                  ▼
           CROSS-ATTENTION   ◄── K, V from the image features
                  │
        Image features:  sky ░   tree ░   road ░   car ████   ← high attention
```

In this design, extra cross-attention layers are added inside the language model, and the image enters through them rather than through the main sequence. Models in the style of Flamingo, which bridged a frozen vision model and a frozen language model with cross-attention (plus a "resampler" that squeezes many patches into a fixed number of vectors), took this route, and so do some more recent models.

```text
Text → ... ─► Q ─┐
                 ▼
          CROSS-ATTENTION ◄── K, V from image features
                 ▼
                LLM
```

### Comparing them

| | **A: visual tokens in the sequence** | **B: cross-attention** |
|---|---|---|
| How the image enters | As extra tokens in the context | Through added cross-attention layers |
| Change to the language model | None | New layers are added |
| Cost | Image tokens use up context length and KV cache | Image does not lengthen the text sequence |
| Can the original model stay frozen? | Often only the projector is trained at first | Often the original layers are frozen and only the new ones trained |
| Typical examples | LLaVA-style models | Flamingo-style models |

A third family trains everything **natively**: text and image tokens are mixed in the same training data from the start, so one model learns both together rather than gluing two models. Modern multimodal systems use variations and combinations of these three ideas. All of them still obey the principle from the introduction: other modalities are turned into vectors the Transformer can use.

---

## Part 6 — The cost: how many tokens is a picture worth?

With Approach A, every visual token takes up room in the context window, and in the KV cache (Chapter 12), exactly like a text token. So it matters how many a picture produces:

| Input | Typical tokens |
|---|---:|
| A 224 × 224 image, 16 × 16 patches | 196 |
| A 336 × 336 image, 14 × 14 patches (a common setting) | 576 |
| A high-resolution image split into several tiles (say 4 tiles plus an overview) | ≈ 2,900 |

An image of 576 tokens costs about as much context as a page of text. And in the KV cache of an 8B-class model at 128 KiB per token (Chapter 12), those 576 tokens take about **72 MiB**, for one image. A conversation with ten images is already hundreds of MiB, and attention cost grows with the square of the total length (Chapter 17). Higher-resolution images give the model finer detail, but cost many more tokens. This is why practical systems resize images, tile large ones, and sometimes compress the number of visual tokens before they reach the language model.

---

## Part 7 — How a multimodal model is trained

Here is the real puzzle. We have two models that grew up separately:

- A **vision encoder** that understands images but knows no language.
- A **language model** that understands text but has never seen a picture.

> If the LLM already understands language and the vision encoder already understands images, **how do we teach their two spaces to understand each other?**

The usual recipe works in stages, and it should look very familiar, because it's built from the fine-tuning tools of Chapter 13:

**Stage 1: Align the spaces (train the projection).** Keep the vision encoder and the language model **frozen**, and train only the small projection layer. The data are image–caption pairs, and the task is the usual next-token prediction: given the image tokens, predict the caption. Because almost nothing else is allowed to change, the projection is forced to learn the translation: *make this image's vectors land where the language model can read them*.

**Stage 2: Visual instruction tuning.** Now unfreeze the language model (fully, or with LoRA, Chapter 13) and train it together with the projector on **instruction data with images**: a picture, a question or instruction, and a good answer. This is exactly supervised fine-tuning, and it is what teaches the model to *follow instructions about images* rather than just caption them. A well-known example, LLaVA, generated much of this instruction data by asking a text-only GPT-4 to write questions and answers from image captions.

**Stage 3 (often): Preference tuning,** with the methods from Chapter 13, to make the answers more helpful and more reliable.

### The loss mask appears again

In Stages 1 and 2, the loss is cross-entropy on the next token, with the **loss mask of Chapter 13**: the model *reads* the image and the question, and is *graded only on the answer*.

```text
[ image tokens × 576 ]  "What is the dog doing?"   →   "The dog is running."
└── context only ──────┘  └── context only ───────┘      └── loss computed here ──┘
```

Nothing new is needed in the training algorithm. It's the same pre-training machinery, with different data and a loss mask.

---

## Part 8 — Audio, video, and what comes out

### Audio

The same recipe applies. An **audio encoder** turns sound into vectors, and a projection places them in the language model's space:

```text
Audio  →  Audio encoder  →  representations  →  projection  →  LLM
```

A typical audio encoder first converts the waveform into a **spectrogram** (a picture of how much energy there is at each frequency over time), then processes it with Transformer layers, like a vision encoder does for images. Speech encoders such as Whisper's produce roughly 50 vectors per second of audio, so a minute of audio is about 3,000 positions.

### Video

A video is a sequence of images, so the straightforward approach is to sample frames (say one per second) and encode each one. The cost grows fast: one frame per second at 576 tokens each is 576 tokens *per second*, or over 34,000 tokens for one minute. Video models therefore sample sparsely and compress visual tokens aggressively.

```text
Video  →  sampled frames  →  vision encoder  →  (compress)  →  visual tokens  →  LLM
```

### What comes out?

Most "multimodal LLMs" take images (and sometimes audio or video) **in** and produce **text** out. Producing images or audio is a different job. It is usually handled by a separate generator (for images, often a diffusion model), or by models that represent images and audio as their own kinds of discrete tokens so that the same next-token machinery can output them. The principle is the same either way: the Transformer works on sequences of vectors, and what those vectors *mean* is decided by the encoders and decoders around it.

### The pattern to remember

```text
TEXT   →  Tokenizer       →  text tokens ─────┐
IMAGE  →  Vision encoder  →  visual tokens ───┤
AUDIO  →  Audio encoder   →  audio tokens ────┤──►  Transformer  →  LM head  →  Text
VIDEO  →  Video encoder   →  video tokens ────┘
```

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "A multimodal model works with raw pixels." | It first converts the image into a short sequence of vectors with an encoder. Raw pixels would be far too long. |
| "The language model is redesigned to handle images." | In the common approach it is unchanged. The image arrives as extra vectors, produced by an encoder and a projection. |
| "Each image is one token." | An image is typically hundreds or thousands of tokens, and they use up context and KV cache like any other tokens. |
| "The vision encoder reads patches in order, like text." | It is an encoder: every patch can attend to every other patch, with no causal mask. |
| "CLIP generates captions." | CLIP only learns which captions match which images. It gives a shared meaning space and does not write text. |
| "The projection layer understands the image." | It is a small translator between two spaces. The understanding comes from the vision encoder and the language model. |
| "Multimodal training needs a new loss function." | It is the usual next-token cross-entropy on the answer, with the loss mask from Chapter 13. |
| "The model sees images and text in different ways internally." | After the projection, both are just vectors in the same sequence, processed by the same attention. |
| "Multimodal models generate images the same way they write text." | Most take images in and produce text. Producing images needs a separate generator, or a design where images are represented as tokens. |
| "Video is just one more image." | A video is many frames. The token cost grows with its length, so models sample and compress frames. |
| "Cross-attention and visual tokens are the same." | Visual tokens lengthen the sequence and use ordinary self-attention. Cross-attention adds layers where text queries the image. |

---

## Quick reference

```text
Problem:    an image is a grid of numbers (224×224×3 = 150,528); can't be raw tokens
Pipeline:   Image → Vision encoder → Projection → (visual tokens) → LLM → text
            Audio / video:  same idea with an audio encoder / sampled frames

ViT:        cut image into patches (16×16 → 196 patches for 224×224)
            flatten each patch → learned projection → vector;  + position embedding
            ordinary Transformer blocks; EVERY patch attends to every patch (encoder, no causal mask)

CLIP:       train image + text encoders on ~400M image–caption pairs;
            similarity grid: push the diagonal (true pairs) up, the rest down → shared meaning space

Projection: a small layer translating vision-encoder vectors into the LLM's embedding space

Two ways in:
  A  visual tokens in the sequence   (LLaVA-style; simple; costs context and KV cache)
  B  cross-attention: Q ← text, K,V ← image   (Flamingo-style; added layers; image not in sequence)
  C  native: train on mixed text and image tokens from the start

Cost:       224px/16 → 196 tokens;  336px/14 → 576 tokens (≈ 72 MiB KV cache at 128 KiB/token);
            video: 576 tokens × frames per second;  audio ≈ 50 positions per second

Training:   1) freeze encoder and LLM, train only the projection on image–caption pairs
            2) visual instruction tuning: train LLM (full or LoRA) + projection on image+instruction→answer
            3) preference tuning
            loss = next-token cross-entropy on the ANSWER only (loss mask, Chapter 13)

Output:     usually text; images/audio out need a separate generator or token-based design
```

---

## What's next

That completes the tour of how a model is built: how its blocks have been improved (Chapters 16 and 17), how its FFN can become a team of experts (Chapter 18), and how it can take in images and sound (this chapter). Next, Chapter 20 turns from *what the model is made of* to *how it behaves on hard problems*. **Reasoning models** don't have a new architecture at all. They are trained, and used at inference time, so that the model thinks at length before it answers. You'll see why that helps, how such models are trained with verifiers and reinforcement learning, and what it costs.
