# Chapter 14 — Quantization: Storing the Same Knowledge in Fewer Bits

## Introduction

Chapter 13 ended with a problem. Fine-tuning gave us a model that behaves well, but even a modest 7-billion-parameter model needs about 14 GB of memory just to hold its weights, and a 70-billion-parameter model needs about 140 GB. That's more than almost any single GPU, and far more than a laptop.

The question that opens this chapter is a simple one:

> **Do we really need 32 bits to store every weight?**

Chapter 13's LoRA asked "do we really need to *update* every weight?" and found the answer was no. Quantization asks the matching question about *storage*, and the answer is again no. A model's learned numbers can be stored far less precisely, in 8 bits or even 4, and the model barely notices. That one idea is the reason people can run Llama-class models on laptops and phones.

This chapter builds quantization from first principles, in the same intuition-first way we built attention:

1. **The problem**: a model is a pile of numbers, and numbers take memory.
2. **Number formats**: what FP32, FP16, BF16, INT8 and INT4 actually are.
3. **Quantizing by hand**: buckets, scale factors, and a worked example you can check with a calculator.
4. **Why one scale isn't enough**: block-wise quantization and the outlier problem.
5. **Why the model survives**: the question everyone asks about errors piling up through 80 layers.
6. **How the math runs**, and why smaller also means *faster*.
7. **Smarter methods**: GPTQ, AWQ, and mixed precision.
8. **Reading the names**: what `Q4_K_M` and its relatives mean.
9. **The engineering trade-off**: what you give up, and why almost everyone ships the quantized model.

Throughout, keep one thing in mind. **Quantization changes only how the learned numbers are stored.** It doesn't change the architecture, the tokenizer, attention, the Transformer blocks, the KV cache, or the LM head. It's the same model, written down more compactly.

---

## Part 1 — The problem: a model is a pile of numbers

### What one parameter is

When we say Llama has "70 billion parameters", what is one parameter? You've already met them. Remember `W_Q`, from Chapter 4:

```text
W_Q
 0.13   -0.52    1.71
 0.04    0.91   -0.12
-0.33    0.81    0.27
```

Each parameter is simply **one number**: a single entry of one of the learned matrices (`W_Q`, `W_K`, `W_V`, `W_O`, the FFN's matrices, the embedding table, the LM head). Every one of those numbers has to live in memory while the model runs.

So the question "how big is the model?" has a very mechanical answer:

```text
memory for weights  =  number of parameters  ×  bytes per parameter
```

The only thing we can change is the **bytes per parameter**.

### How many bits should one weight use?

Consider one weight, say `0.173421894`. How many bits should we spend storing it? The obvious answer is "as many as needed". But think about how you use numbers in everyday life. If I ask how much money you have and you say `₹100`, that's good enough. I don't need `₹100.0000000000000000000001`. The extra digits are real, but useless.

The same question applies to a neural network: do we need `0.17342189421563`, or would `0.17` do? Before we find out, let's see what the savings would be.

### The memory table

For a 7-billion-parameter model, just the weights:

| Format | Bits per weight | Bytes per weight | 7B model | 70B model |
|---|---:|---:|---:|---:|
| FP32 | 32 | 4 | ≈ 28 GB | ≈ 280 GB |
| FP16 / BF16 | 16 | 2 | ≈ 14 GB | ≈ 140 GB |
| INT8 | 8 | 1 | ≈ 7 GB | ≈ 70 GB |
| INT4 | 4 | 0.5 | ≈ 3.5 GB | ≈ 35 GB |

(Real files are slightly larger than these ideal numbers, as Part 4 explains, because each group of weights also stores some extra bookkeeping. A "4-bit 70B" model is typically in the region of 35 to 45 GB.)

Notice what we have *not* done. We didn't change the Transformer. We didn't change attention. We didn't retrain anything. We simply stored the **same learned numbers using fewer bits**. A 4-bit 70B model fits on hardware that a 16-bit one never could, and a 4-bit 7B model fits comfortably on a laptop or a phone.

### Weights are not the only memory user

One caution before we go on. The weights are the biggest fixed cost, but a running model also needs memory for:

- the **KV cache** (Chapter 12), which grows with every token of context,
- **activations**: the intermediate vectors flowing through the layers,
- the engine's own overhead.

So a model whose weights take 14 GB needs a GPU with comfortably *more* than 14 GB. We'll come back to this when we discuss quantizing the KV cache in Part 9.

---

## Part 2 — Number formats: what FP32, FP16, BF16, INT8 and INT4 really are

To quantize, we need to know what we're quantizing *from* and *to*. There are two families of formats: **floating point** (numbers that can be tiny or huge) and **integers** (whole numbers in a limited range).

### Floating point: sign, exponent, mantissa

A floating-point number is stored like scientific notation. `0.000173` is written as `1.73 × 10⁻⁴`: a few meaningful digits (the **mantissa**) and a power of ten (the **exponent**) that says where the decimal point goes. Computers do the same thing in base 2, plus one bit for the sign:

```text
FP32   ┌──┬──────────┬───────────────────────────┐
       │ 1│    8     │            23             │   32 bits = 4 bytes
       └──┴──────────┴───────────────────────────┘
        sign exponent        mantissa

FP16   ┌──┬─────┬────────────┐
       │ 1│  5  │     10     │                       16 bits = 2 bytes
       └──┴─────┴────────────┘

BF16   ┌──┬──────────┬───────┐
       │ 1│    8     │   7   │                       16 bits = 2 bytes
       └──┴──────────┴───────┘
```

The **exponent** bits decide the *range* (how big or small a number can get). The **mantissa** bits decide the *precision* (how finely you can tell nearby numbers apart).

| Format | Exponent bits | Mantissa bits | Largest value | Rough precision |
|---|---:|---:|---:|---|
| FP32 | 8 | 23 | ≈ 3.4 × 10³⁸ | about 7 decimal digits |
| FP16 | 5 | 10 | 65,504 | about 3 to 4 digits |
| BF16 | 8 | 7 | ≈ 3.4 × 10³⁸ | about 2 to 3 digits |

FP16 and BF16 both use 16 bits, but they spend them differently:

- **FP16** keeps more precision but has a small range (it overflows above 65,504). That can bite during training, where some values get very large.
- **BF16** ("brain float") keeps the **same range as FP32** and gives up precision instead. It rarely overflows, which is why it became the standard format for training and for shipping models.

Here is how our example weight survives each format:

```text
FP32    0.173421894   →   0.173421890   (essentially exact)
FP16    0.173421894   →   0.173461914
BF16    0.173421894   →   0.173828125
```

The differences are in the fourth decimal place, and that's all. FP16 and BF16 are widely used precisely because they preserve almost all of a model's quality while halving the memory compared with FP32. When you see a model released as "BF16 weights", this is what it means.

(Even smaller floats exist. **FP8** has two common variants, E4M3 and E5M2, and **FP4** has just 4 bits in total. Modern GPUs can do arithmetic directly in these. We'll see DeepSeek training with FP8 in Chapter 21.)

### Integers: a short ruler

Integers work very differently. An **n-bit integer** can represent only `2ⁿ` distinct values:

| Format | Distinct values | Typical (signed) range |
|---|---:|---|
| INT8 | 256 | −128 … 127 |
| INT4 | 16 | −8 … 7 |

There is no exponent, no mantissa, no fractions. Just a small ladder of whole numbers. Picture two rulers. FP32 is a ruler with markings every hair's width:

```text
0.171  0.172  0.173  0.174  0.175   ← lots of precision
```

INT4 is a ruler with very few markings:

```text
0.15        0.20        0.25        0.30        ← far fewer representable values
```

So with integers, quantization means we are **rounding** each weight to the nearest marking on a short ruler. A good image analogy: a photo with 16 million colours is beautiful. Reduce it to 16 colours and it's still recognizable, but subtle detail is lost. Quantization does the same thing to the weights.

But integers can't hold `0.173` at all. They only hold whole numbers. So how do we get a decimal weight onto an integer ladder? That is what the next part is about.

---

## Part 3 — Quantizing by hand

### The naive idea, and why it fails

Suppose a row of a weight matrix, in FP32, is:

```text
 0.12   -0.83    0.41    1.27   -0.36    0.91
```

and INT4 allows only 16 distinct values. The simplest idea is to round to the nearest whole number:

```text
0.91  →  1
0.83  →  1        ← oops
```

Different weights collapse into the same value, and almost all the information is lost. Rounding to whole numbers is far too coarse for numbers this small.

### Brilliant idea 1: buckets, and storing an index

Instead of rounding to whole numbers, look at the range the weights actually occupy and divide *that* range into 16 evenly spaced buckets. Suppose the weights run from −1.5 to 1.5:

```text
bucket:    0     1     2   ...    12    ...    15
value:   −1.5  −1.3  −1.1  ...   0.9   ...   1.5
```

Now `0.91` falls into the bucket at `0.9`. Here is the important realization:

> **We no longer store the weight. We store the *index* of its bucket.**

The model stores `12`, not `0.91`. During inference, "12" is translated back into "about 0.9". It's like a dictionary: the stored integer is just a key into a table of real values.

### Brilliant idea 2: a scale factor

But a table of 16 values per layer would be awkward. There's a neater way to say the same thing. Pick one number, the **scale**, and let the integer tell you *how many scale-steps* from zero the weight is:

```text
approximate weight  =  integer  ×  scale
```

For example, if the integer is `7` and the scale is `0.13`:

```text
7 × 0.13  =  0.91
```

So instead of storing `0.913482` (32 bits), we store `integer = 7` (4 bits) and share `scale = 0.13` across many weights. Recovering the number is called **dequantization**, and what comes back is an **approximation**, not the exact original.

How do we choose the scale? The simplest way (called **absmax** or *symmetric* quantization) is to make the largest-magnitude weight land exactly on the largest integer:

```text
scale  =  (largest absolute weight)  /  (largest integer)

integer  =  round( weight / scale )
weight ≈ integer × scale
```

(Some schemes also store a **zero point**, a shift that lets the integer ladder start somewhere other than zero. That helps when values aren't centred on zero, and it's called *asymmetric* quantization. The idea is otherwise identical.)

### The worked example

Take our row again: `0.12, −0.83, 0.41, 1.27, −0.36, 0.91`. The largest absolute value is `1.27`.

**INT8**, where the largest integer is 127:

```text
scale = 1.27 / 127 = 0.01

integers:     12     −83     41     127     −36     91
recovered:  0.12   −0.83   0.41    1.27   −0.36   0.91      ← exact!
```

(This row happens to land exactly on the 0.01 grid, so nothing is lost. In general INT8 has 256 levels, so the error is tiny but not zero.)

**INT4**, where the largest integer is 7:

```text
scale = 1.27 / 7 = 0.1814

integers:          1      −5      2       7      −2      5
recovered:     0.181  −0.907   0.363   1.270  −0.363  0.907
original:       0.12   −0.83    0.41    1.27   −0.36   0.91
error:         0.061   0.077   0.047   0.000   0.003  0.003
```

Now the rounding is visible: the weight `0.12` came back as `0.181`, and `−0.83` as `−0.907`. INT4 is a coarse ruler, and the biggest weight (which defined the scale) is the only one guaranteed to be exact.

What does that do to a real calculation? Take an input vector `x = [0.5, 1.0, −0.3, 0.2, 0.8, 0.4]` and compute the dot product with this row, as a neuron would:

```text
with the original weights:        −0.563
with the INT4 weights:            −0.599        (difference ≈ 0.036)
```

Individual weights were off by up to 0.077, yet the neuron's output is off by only about 0.036, because some errors push up and others push down and they partly cancel. A small, noisy change in a number: remember that for Part 5.

### What we're really storing

After quantization, a block of weights is stored as:

```text
[ integers, 4 bits each ]  +  [ one scale ]  (+ sometimes a zero point)
```

At run time the engine rebuilds `integer × scale` to get the approximate weights back. The Transformer never notices: it still receives numbers. They are just slightly approximate.

---

## Part 4 — Why one scale isn't enough: block-wise quantization

### The problem with a single range

Suppose we used **one scale for the entire model**. One layer might have weights such as `−100, 50, 75, −80`, and another might have `0.001, 0.003, −0.002`. They can't possibly share a scale. The scale for the first layer would round everything in the second to zero.

The fix is obvious: give each layer its own scale. But why stop there? A layer has millions of weights, and different parts of it behave differently.

### Outliers: one big weight ruins its neighbours

Here's the problem in miniature. Take our row, but replace `1.27` with a single large weight, `12.7`:

```text
original:   0.12   −0.83    0.41   12.70   −0.36    0.91

scale = 12.7 / 7 = 1.814        (the big weight sets the scale)

integers:      0       0       0       7       0       1
recovered:  0.00    0.00    0.00   12.70    0.00    1.81
```

Four of the six weights collapsed to zero. One outlier forced a coarse scale on all its neighbours, and nearly all their information was lost.

Real LLM weight matrices do contain occasional large values, so this is not hypothetical. The remedy is **block-wise quantization**:

```text
one scale for the whole layer:
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

one scale per small block (commonly 32 to 128 weights):
□□□□□□□□□□□□□□□□   □□□□□□□□□□□□□□□□   □□□□□□□□□□□□□□□□
 scale 1            scale 2            scale 3
```

An outlier now only damages the other weights in *its own small block*. Every block gets a scale suited to its own range, and accuracy improves dramatically.

### The cost of the extra bookkeeping

Block-wise quantization isn't free: every block stores its own scale. For example, if a block is 32 weights with one 16-bit scale, then:

```text
(32 weights × 4 bits  +  16 bits for the scale) / 32 weights  =  4.5 bits per weight
```

So a "4-bit" format actually costs about 4.5 bits per weight once the scales are counted. That's why a real 4-bit file is a little bigger than the ideal `params × 0.5 bytes`, and it's also why GGUF files are not just "compressed weights". They hold the integers **plus the metadata needed to rebuild each block** (Chapter 22).

### How coarse can the grouping be?

You'll meet several names for this choice:

| Granularity | One scale per… | Trade-off |
|---|---|---|
| Per-tensor | whole weight matrix | cheapest, least accurate |
| Per-channel | each row/column of the matrix | better, still cheap |
| Block-wise (group-wise) | small block of weights, e.g. 32 or 128 | best accuracy, small extra storage |

Modern low-bit formats use block-wise scales almost universally.

---

## Part 5 — Why the model survives

Here is the question that bothers almost everyone, and it is the right one.

> Suppose I quantize `W_Q`. Then `Q = X·W_Q` is slightly wrong. `K` is slightly wrong. `V` is slightly wrong. So the attention scores are slightly wrong. The FFN is slightly wrong. The LM head is slightly wrong. **Won't these tiny errors pile up through 32 or 80 Transformer layers and destroy the model?**

That is exactly what researchers expected. **They didn't.** Why not?

**1. Small perturbations stay small.** A weight changes from `0.1734` to `0.1732`. That slightly changes one neuron's output, which slightly changes the next layer. It doesn't flip anything dramatically. Networks are, in general, smooth: a small change in the input of a function usually gives a small change in the output.

**2. The errors are not all in the same direction.** As our worked example showed, rounding errors point both ways, so they partly cancel instead of adding up in one direction. And each output is the sum of *thousands* of products, so no single rounding mistake dominates.

**3. The network is redundant.** The knowledge isn't stored in one weight. It is spread across billions of them, and those weights work together, so a modest disturbance to each is averaged over many.

**4. Training was noisy anyway.** Every training step uses a random mini-batch (Chapter 11), and the weights began as random numbers (Chapter 9). A model that ends up good has learned a solution that tolerates a certain amount of jitter, rather than one that balances on a knife edge.

Think of GPS. Your true position is `19.0761° N`, and the GPS says `19.0760° N`. You'll still reach your destination, because the error is too small to matter. Or think of a human answering `2 + 2` with a little background noise: you can still answer.

The most important mental model:

> **Quantization is not deleting knowledge. It is storing the same knowledge less precisely.**

Like saving a photo as a high-quality JPEG instead of an uncompressed bitmap: the picture is the same, and some microscopic details disappear.

### But it is not free

This robustness has limits, and honesty about them matters:

- **8-bit is almost always effectively lossless.** 4-bit with a good method loses a small, usually hard-to-notice amount of quality. 3-bit and 2-bit lose clearly more.
- **Smaller models are less tolerant** than larger ones. A 7B model at 4-bit degrades more than a 70B model at 4-bit, because there is less redundancy to absorb the noise.
- **Some skills degrade first.** Long reasoning chains, arithmetic, and rare knowledge tend to suffer before casual chat does.
- **Naive rounding is much worse than clever methods** at the same bit-width. That's what Part 7 is about.

So the right summary isn't "quantization is free". It is "quantization gives up a *small, controllable* amount of quality for a *large* saving in memory and time".

---

## Part 6 — How the math runs, and why smaller is also faster

### Does the GPU convert everything back?

Once the weights are stored as `integer × scale`, the matrix multiplication still needs to happen. There are two broad ways to do it:

**Method 1: dequantize, then multiply.**

```text
INT4 weights  →  convert back to FP16  →  ordinary matrix multiply
```

Simple and flexible, and the conversion happens on the fly right before the multiply, so the big FP16 copy never has to sit in memory.

**Method 2: quantized (fused) kernels.**

```text
INT4 weights  →  optimized kernel that unpacks, scales and multiplies in one go  →  output
```

Modern inference libraries fuse dequantization into the matrix-multiply itself, or use hardware that multiplies low-precision numbers directly. This is a large part of why engines like llama.cpp are so fast on ordinary hardware.

Either way, the pipeline becomes:

```text
Compressed weight  →  recover approximate weight  →  matrix multiply
```

and the Transformer around it is unchanged.

### Why fewer bits also means faster generation

You'd expect quantization to save memory. The surprise is that it also makes generation **faster**. Recall from Chapter 12 that the *decode* phase (generating one token at a time) is **memory-bound**: for every new token, the GPU has to read essentially *all the weights* from memory, but does very little arithmetic with each one.

When the bottleneck is how fast bytes can be read, then:

```text
tokens per second  ≲  memory bandwidth  /  bytes of weights read per token
```

Take a 7B model on a GPU that can read about 1 TB per second:

```text
FP16  (14 GB):   1000 GB/s  /  14 GB   ≈  70 tokens/s   (theoretical ceiling)
INT4  (≈4 GB):   1000 GB/s  /   4 GB   ≈ 250 tokens/s   (theoretical ceiling)
```

Real engines don't reach these ceilings, but the ratio is the point. **Fewer bytes to read per token means more tokens per second**, so quantization makes the model both smaller *and* faster.

(Prefill, the compute-heavy phase, benefits less, because it is limited by arithmetic rather than by reading weights.)

---

## Part 7 — Smarter quantization: GPTQ, AWQ and mixed precision

So far we've treated every weight as equally important and rounded them all the same way. Modern methods ask a better question:

> **Which weights can tolerate approximation, and which should be preserved more carefully?**

That is the difference between **naive** and **production** quantization.

### Why importance differs

Consider four weights:

| Weight | If quantization changes it, the damage to the model is… |
|---|---|
| 0.91 | large ⭐⭐⭐⭐⭐ |
| 0.72 | considerable ⭐⭐⭐⭐ |
| 0.44 | modest ⭐⭐ |
| 0.01 | negligible ⭐ |

Rounding `0.91` to `0.70` could be disastrous. Rounding `0.01` to `0.00` is almost irrelevant. A good quantizer should spend its limited precision where it matters. To judge what matters, these methods use a **calibration dataset**.

### Calibration: a small test run, not training

A calibration set is a small sample of typical text (perhaps a few hundred examples: Wikipedia, books, code, questions). The quantization method runs the model on it and watches how the numbers behave. Crucially:

- It is **not training** and it is **not fine-tuning**.
- There is **no backpropagation**, no AdamW, no weight updating by gradient descent.
- The data is only used to *measure* what the model's layers do, so the quantizer can make better rounding decisions.

### GPTQ: compensate for the damage as you go

**GPTQ** works layer by layer. Instead of rounding all the weights independently, it quantizes the weights of a layer in sequence and, after rounding each one, **adjusts the not-yet-quantized weights to compensate for the error just introduced**, so that the layer's *output* on the calibration data stays as close as possible to the original. (It uses second-order information about how sensitive the layer's output is to each weight, which is what lets it judge importance.) The result is a much better 3-bit or 4-bit model than plain rounding could give.

### AWQ: protect the weights that meet big activations

**AWQ** (Activation-aware Weight Quantization) asks about importance from a different angle. Remember that a layer computes `Y = XW`, and there are two things here: `W`, the weights, and `X`, the *activations*, the vectors flowing through the network.

Consider two weights:

```text
Weight 0.8, multiplied by an activation of 0.00001   →  nearly irrelevant
Weight 0.2, multiplied by an activation of 500       →  a tiny error here is amplified 500×
```

The second weight matters far more, even though it is smaller, because the input it is multiplied by is large. AWQ uses the calibration data to find which weights are paired with large, important activations, and protects them by scaling them so they are represented more accurately. A small fraction of weights ends up being responsible for a large share of the quality.

### Comparing the two

| | **GPTQ** | **AWQ** |
|---|---|---|
| Asks | How much does this weight's error change the layer's output? | Which weights meet large, important activations? |
| Looks at | The effect of weight errors, using calibration data | Activation statistics from calibration data |
| Idea | Round, then compensate with the remaining weights | Scale up the important weights so they lose less |
| Retrains? | **No** | **No** |

Both aim for the same goal, **keeping quality high while shrinking memory**, and they simply estimate "importance" differently.

### Mixed precision: not every tensor needs the same bits

A related, simpler idea: some *tensors* are more sensitive than others. Modern formats often keep very sensitive tensors at higher precision (for instance, embeddings, the LM head, or parts of attention and FFN) and compress the less sensitive ones harder. That is why a "4-bit" model is often a mix of 4-bit and 5- or 6-bit pieces.

### Two other things worth knowing

**When quantization happens.** Everything above is **post-training quantization (PTQ)**: take a finished model and compress it, with no retraining. The alternative is **quantization-aware training (QAT)**, where the model is trained (or fine-tuned) while simulating low precision, so it learns to be robust to it. PTQ is far cheaper and is what most people use. QAT can do better at very low bit-widths.

**What gets quantized.** So far we've compressed **weights only**. You can also compress **activations**, so the multiplication itself runs in low precision. This is harder, because LLM activations contain rare but very large outlier values that don't compress well. Weights-only schemes (such as 4-bit weights with 16-bit activations, often written "W4A16") are the common practical choice, and 8-bit weights *and* activations ("W8A8") is used where hardware support is good.

---

## Part 8 — Reading the names

When you browse Hugging Face you constantly see names like:

```text
Q4_K_M.gguf     Q5_K_M.gguf     Q8_0.gguf     IQ4_XS.gguf
```

Now you can read them. Take them apart:

### `Q4`, `Q5`, `Q8`: the bit-width

```text
Q4  →  about 4 bits per weight
Q5  →  about 5 bits per weight
Q8  →  about 8 bits per weight
```

### `K`: the "k-quant" family

`K` marks a newer family of **block-wise** schemes from llama.cpp, with better scale metadata than the older formats. It stores weights in larger "super-blocks" that contain smaller sub-blocks, each with its own scale, so the scales themselves are stored compactly. Think of it as **a smarter block quantization method**. The older formats, with names ending in `_0` or `_1` (such as `Q8_0` and `Q4_0`), use one simple scale per block of 32 weights.

### `S`, `M`, `L`: the preset size

The final letter is a preset in a family that trades file size against quality:

```text
_S  (Small)   →   _M  (Medium)   →   _L  (Large)
smallest file                          slightly more accurate, slightly bigger
```

Larger presets keep more of the sensitive tensors at higher precision.

### Putting it together

```text
Q4_K_M  =  4-bit weights  +  k-quant block scheme  +  medium-quality preset
```

### `IQ`: importance-based formats

Names starting with `IQ` (such as `IQ4_XS`) are formats that use a calibration step, an *importance matrix*, to decide which weights deserve more precision. It's the same idea as Part 7, applied inside the GGUF world.

### What the sizes look like

For a model of about 8 billion parameters, such as Llama-3-8B, approximate file sizes are:

| Format | Approx. bits per weight | Approx. file size | Typical quality |
|---|---:|---:|---|
| FP16 / BF16 | 16 | ≈ 16 GB | the reference |
| `Q8_0` | 8.5 | ≈ 8.5 GB | almost indistinguishable |
| `Q5_K_M` | ≈ 5.7 | ≈ 5.7 GB | very close |
| `Q4_K_M` | ≈ 4.8 | ≈ 4.9 GB | small loss; a popular default |
| `Q3` / `Q2` variants | ≈ 3 or less | smaller still | quality drops noticeably |

(Sizes are approximate, and quality depends on the model and the task.) `Q4_K_M` is a common sweet spot, and the reason is just the trade-off above: it cuts the file to about a third of FP16 with only a small loss.

### Other formats you'll meet

Outside llama.cpp's GGUF world, you'll see:

- **GPTQ** and **AWQ** checkpoints: models saved with the GPTQ or AWQ methods from Part 7, commonly used with GPU servers.
- **bitsandbytes**: a library that quantizes on the fly when a model loads, including **NF4**, a 4-bit format designed for the bell-shaped distribution of weights. It's what QLoRA (Chapter 13) uses.
- **FP8 and FP4**: low-precision floating-point formats that modern GPUs support natively, used both for inference and, in some recent models, for training.

One correction to the toy picture we started with. We said quantization means "replace every weight with an integer". That is conceptually right and the right mental model, but in production each block stores the quantized values, one or more scales, and sometimes extra metadata such as zero points or specialized encodings. The dequantization math is a little richer than `integer × scale`. The idea doesn't change.

---

## Part 9 — The engineering trade-off

### Would you ship the quantized model?

Imagine two options for the same 7B model:

| | Scenario 1: FP16 | Scenario 2: INT4 |
|---|---|---|
| Model size | 14 GB | 3.5 to 4 GB |
| Needs a GPU of about | 24 GB | 8 GB |
| Speed (illustrative) | 15 tokens/s | 45 tokens/s |
| Quality (illustrative) | 100% | ≈ 99% |

(The numbers are illustrative of the *shape* of the trade, not benchmarks.)

Which would you deploy? Almost every production team picks INT4. Users notice faster responses, lower latency and lower cost. They almost never notice a 1% quality difference in everyday use.

This is a general engineering lesson:

> **Researchers optimize for maximum quality. Engineers optimize for the best trade-off.** Those are different goals.

That is why GPTQ, AWQ and GGUF became famous. They weren't trying to create a smarter model. They were asking *"can we keep about 99% of the intelligence while cutting memory by 75%?"* and the answer was yes.

(One honest caveat. If your task is quality-critical, such as complex reasoning, code or maths, measure the quantized model on *your* task before shipping. Pick the highest bit-width you can afford, and treat the quality loss as something to test, not assume.)

### Quantizing things other than the weights

Weights aren't the only thing that can be stored in fewer bits:

| What | Why it matters | Notes |
|---|---|---|
| **Weights** | The biggest fixed memory cost | The main topic of this chapter |
| **KV cache** | Grows with every token (Chapter 12) | Engines such as llama.cpp and vLLM can store keys and values in 8-bit or lower, so longer contexts or more users fit in memory |
| **Activations** | Make the arithmetic itself cheaper | Harder because of outliers; W8A8 is used where hardware allows |
| **Training numbers** | Faster, cheaper training | Training runs in BF16 or even FP8 (Chapter 21) |

### How quantization combines with what came before

- **QLoRA** (Chapter 13) is quantization plus LoRA: a 4-bit frozen base model with small trainable adapters on top.
- **LoRA adapters** are normally kept in higher precision, since they're tiny and they're the part being learned.
- **The KV cache** (Chapter 12) and quantized weights are independent: you can compress either, both, or neither.

### The pattern behind modern LLM engineering

Look at the breakthroughs we have met so far. Nobody invented a brand-new Transformer for any of them. Each one asked what the most expensive thing in the current system is, and whether it can be simplified without sacrificing much quality:

| Problem | The question asked | The idea |
|---|---|---|
| Training is expensive | Do we need to update every weight? | **LoRA** (Chapter 13) |
| Generation repeats work | Do we need to recompute every token? | **KV cache** (Chapter 12) |
| Memory is expensive | Do we need to store every weight at full precision? | **Quantization** (this chapter) |
| RLHF is complicated | Do we need a reward model and RL? | **DPO** (Chapter 13) |

Modern AI engineering is often about optimizing *around* the Transformer rather than replacing it.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "Quantization deletes some of the model's knowledge." | It stores the same numbers less precisely, like a JPEG of a photo. Nothing is removed, only rounded. |
| "A quantized model is a different model." | Same architecture, same tokenizer, same layers. Only the weight storage differs. |
| "You need to retrain a model to quantize it." | Post-training quantization needs no training, just a conversion (and sometimes a small calibration run). |
| "GPTQ and AWQ train the model on calibration data." | Calibration data is only used to *measure* sensitivity. There is no backpropagation or optimizer. |
| "INT4 means every weight is exactly 4 bits." | Each block also stores a scale (and sometimes more), so a "4-bit" format costs roughly 4.5 bits per weight. |
| "One scale for the whole model is enough." | One large outlier would ruin the precision of everything else. Scales are per block. |
| "Rounding errors must add up over 80 layers and ruin the model." | They are small, point in both directions, and are spread over huge numbers of redundant weights. In practice 8-bit is near-lossless and good 4-bit loses little. |
| "Quantization only saves memory." | It also speeds up generation, because decoding is limited by how many bytes it has to read per token. |
| "Lower bits is always better." | Below about 4 bits, quality drops noticeably, especially for small models and reasoning tasks. |
| "FP16 and BF16 are the same." | Both are 16 bits. FP16 has more precision but a small range; BF16 has FP32's range but less precision. |
| "Quantizing the weights shrinks the whole memory footprint by the same ratio." | The KV cache and activations are separate. A 4-bit model still needs memory for them. |
| "Q4_K_M is a magic code." | It reads `4-bit` + `k-quant block scheme` + `medium preset`. |

---

## Quick reference

```text
Model memory  =  parameters  ×  bytes per parameter
                 7B:   FP32 28 GB | FP16/BF16 14 GB | INT8 7 GB | INT4 3.5 GB   (+ overhead)

Floating point:  sign | exponent (range) | mantissa (precision)
                 FP32 = 1+8+23    FP16 = 1+5+10    BF16 = 1+8+7
Integers:        n bits → 2ⁿ levels  (INT8: 256, INT4: 16)

Quantize:    scale = max|w| / max_integer ;   int = round(w / scale)
Dequantize:  w ≈ int × scale             (a block stores ints + scale [+ zero point])
Block-wise:  one scale per small block (32–128) so one outlier can't ruin everything
             ≈ 4.5 bits per weight for a "4-bit" format

Why it works:  small, two-sided errors, spread over billions of redundant weights
Why faster:    decode is memory-bound → fewer bytes read per token → more tokens/s

GPTQ   round layer by layer, compensate error with the remaining weights
AWQ    protect the weights that meet large activations
PTQ    compress a finished model        QAT   train while simulating low precision

Names:  Q4_K_M = 4-bit + k-quant blocks + Medium preset
        Q8_0 ≈ near-lossless   Q5_K_M ≈ very close   Q4_K_M ≈ popular default

Unchanged by quantization:  architecture, tokenizer, attention, blocks, KV cache, LM head
```

---

## What's next

We have now covered the full life of a GPT-style model: how it is built (Chapters 2 to 8), trained (Chapters 9 to 11), run (Chapter 12), adapted (Chapter 13) and compressed (this chapter). But everything so far has been one particular design: the plain GPT decoder. Chapter 15 steps back to look at the **Transformer family**: encoder-only models like BERT, decoder-only models like GPT, and encoder–decoder models like T5. We'll see what each one is for, how they differ, and why decoder-only models ended up dominating modern LLMs. Chapters 16 and 17 then show how those decoder-only models have been improved, Chapters 18 and 19 extend the architecture itself (Mixture of Experts and multimodal models), and Chapter 20 turns to reasoning. Much later, Chapters 22 and 23 return to the engineering of serving models, starting with the file format, GGUF, that stores the quantized weights you just learned about.
