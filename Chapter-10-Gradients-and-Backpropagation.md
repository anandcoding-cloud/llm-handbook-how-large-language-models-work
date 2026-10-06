# Chapter 10 — Training II: Gradients and Backpropagation

## Introduction

Chapter 9 ended with a loss: one number that says how wrong the model's prediction was. This chapter takes the hardest step in training, which is turning that one number into a separate instruction for every weight in the model. It introduces gradients first, and then backpropagation, the method that computes all of them at once.

---

## Part 1 — The big question: one number, billions of weights

Now we've reached the first moment the model gets **feedback**. Everything before this was just computation. Here we can finally say: *"You were wrong, and here's by how much."* Suppose the loss is `2.73`.

That's a single number. The model has billions of parameters. How can one number tell every parameter exactly how to change? Should `W_Q` increase by `0.0003`? Should one value inside `W2` decrease by `0.00007`? How could the loss possibly know?

This is the hardest and most important question in deep learning, and the answer has a surprisingly gentle starting point: **the derivative.**

### What a derivative is: a question about sensitivity

A derivative answers one question:

> *If I change this one thing by a tiny amount, how much does the result change?*

Think of a salary that depends on years of experience. "If I work one more year, how much does my salary go up?" That's a derivative. Now replace experience with **one single weight** inside the model, and salary with **the loss**:

```text
"If I nudge this one weight up by a tiny amount, does the loss go up, go down, or stay the same?"
```

Suppose one weight is `W = 2` and the loss is `50`. Nudge it to `2.1` and the loss becomes `49`. Nudge it to `2.2` and it drops to `48`. The derivative is telling us: *increase this weight.* Now take a different weight, `W = 8`, with the same loss of `50`. Nudge it up slightly and the loss rises to `60`. The derivative says: *decrease this weight.* That's the whole idea. For every weight, we ask "if I change you slightly, does the loss improve or get worse, and by how much?"

The collection of these answers has a name. The derivative of the loss with respect to one weight is called that weight's **gradient**, written:

```text
∂Loss / ∂W      read as:   "how much does the loss change when W changes a tiny bit?"
```

### But we can't try this one weight at a time

Here is the catch. With 10 billion parameters, testing each weight individually would mean: nudge weight 1, run the entire model, measure the loss, nudge weight 2, run the entire model again... ten billion times, for a *single* training step. That would take years.

The genius of backpropagation is that it computes **all** the gradients at once, using one forward pass and one backward pass. That is the subject of the next part.

---

## Part 2 — Backpropagation

Backpropagation isn't another neural network, and it isn't magic. It's simply an extremely efficient way of answering one question for every parameter at the same time:

> **"How sensitive is the loss to this parameter?"**

Let's build it from the smallest possible example, then scale up.

### The world's smallest neural network

One input, one weight, one loss:

```text
x = 3  ──►  × w = 2  ──►  Prediction y = 6  ──►  Loss = (y − 8)² = 4
```

We want the target to be `8`, but the model predicted `6`, so the loss is `(6 − 8)² = 4`. We can already see by hand that increasing `w` helps: with `w = 2.1` the prediction is `6.3` and the loss falls to `2.89`. But the model can't test every weight by hand, so we need mathematics.

### Don't jump straight from the loss to the weight

Notice that the weight doesn't affect the loss directly. It affects the **prediction**, and the prediction affects the loss:

```text
Weight  →  Prediction  →  Loss
```

So we split one hard question into two easy ones. This is the **chain rule**, one of the most famous ideas in calculus:

```text
∂Loss/∂w   =   ∂Loss/∂Prediction   ×   ∂Prediction/∂w
```

In plain English: *"how does the loss respond to the prediction, multiplied by how the prediction responds to the weight."*

**Question 1: if the prediction changes, how does the loss change?** The loss is `(y − 8)²`, whose derivative is `2(y − 8)`. At `y = 6`:

```text
∂Loss/∂Prediction = 2(6 − 8) = −4
```

The meaning: *if the prediction increases a tiny bit, the loss decreases.*

**Question 2: if the weight changes, how does the prediction change?** The prediction is `3 × w`, so if the weight rises by 1, the prediction rises by 3:

```text
∂Prediction/∂w = 3
```

**Multiply them:**

```text
∂Loss/∂w = (−4) × 3 = −12
```

That `−12` is the gradient. It does **not** mean "decrease the weight by 12". It means *"if you increase this weight slightly, the loss drops quickly."* The **sign** tells you the direction (negative gradient: increase the weight), and the **size** tells you how sensitive the loss is to that weight.

Look at what actually happened in the backward direction:

```text
Forward:    Weight  →  Prediction  →  Loss
Backward:   Loss  →  gradient at Prediction  →  gradient at Weight
```

Each step only did one tiny job. The loss node computed `∂Loss/∂Prediction`. The multiply node computed `∂Prediction/∂w`. The chain rule simply multiplied them.

### How this scales to GPT

Replace that tiny graph with GPT's:

```text
Embedding → Attention → W_O → Residual → LayerNorm → FFN → Residual → LayerNorm → LM Head → Softmax → Loss
```

It's just a much bigger graph, and the chain rule still works. **Every operation only needs to know one thing: "if my output changes a little, how does my input change?"** Nobody has to understand the whole Transformer. Each operation knows only its own local derivative, and the chain rule stitches them together, like workers on an assembly line passing a message backward:

```text
Loss  →  Softmax says "here's my derivative"
      →  LM Head says "multiply by mine"
      →  FFN says "multiply by mine"
      →  Attention says "multiply by mine"
      →  Embedding says "now I know my gradient"
```

This is exactly what `loss.backward()` does in PyTorch. There is no AI inside it. It walks backward through the computation graph asking each operation for its local derivative and applies the chain rule. It's just calculus.

### Where does the backward pass start in GPT?

It starts at the loss, and the very first gradient has a beautifully simple form. For softmax followed by cross-entropy, the gradient on each logit is just **(predicted probability − truth)**. Back to our example, with the correct answer `tired`:

```text
Word       Predicted p   Truth   Gradient on its logit (p − truth)
tired         0.30         1          −0.70    ← push this score UP
hungry        0.55         0          +0.55    ← push this score DOWN (it's the biggest mistake)
asleep        0.10         0          +0.10    ← push down a little
running       0.05         0          +0.05    ← push down a little
```

The signal says exactly what you'd hope: raise the correct word's score, lower the others, and lower the most-overconfident wrong answer the most. From here, that signal travels backward through the whole network.

### Which operations actually learn?

A common confusion is thinking backpropagation "updates every layer." It doesn't, and the distinction is worth getting exactly right. Many operations in a Transformer, like softmax, dot products, addition, GELU and concatenation, have **no parameters**: nothing to adjust. Yet the gradient still has to pass *through* them to reach the layers that do.

Think of plumbing. Water flows from tank A through pipes and a valve to tank B. Only the valve has a knob you can adjust. The pipes have no knobs, but the water still has to flow through them to reach the valve. Or think of a relay race where only some runners are employees who can receive training: the coach sends a message backward, volunteers simply pass it along, and employees update their training.

```text
Gradient arrives at an operation
        │
        ├── Does it have parameters?
        │       ├── Yes → compute the gradient for those parameters (to store),
        │       │         AND pass a gradient backward
        │       └── No  → just pass a gradient backward
        ▼
Previous operation
```

So **every operation participates, but only operations with trainable parameters ever get updated.** (This is also why PyTorch records every operation during the forward pass: not because every operation learns, but because every operation must know how to send gradients to its inputs on the way back.)

### A worked example: one linear layer

Every learned matrix in a Transformer is a linear layer, `Y = X · W`, so this is the single most useful example to see once. We'll use the smallest possible version: an input with two numbers, and one weight for each.

**The forward pass.** The input is `x₁ = 2` and `x₂ = 3`. The weights are `w₁ = 4` and `w₂ = 5`. The layer multiplies each input by its weight and adds the results:

```text
x₁ = 2 ──× w₁ (=4)──►  8  ──┐
                             ├──► add ──► Y = 23 ──► Loss = (23 − 20)² = 9
x₂ = 3 ──× w₂ (=5)──► 15  ──┘                         (target is 20)
```

In one line, `Y = 2·w₁ + 3·w₂ = 8 + 15 = 23`. The target was `20`, so the model overshot, and the loss is `(23 − 20)² = 9`.

**The backward pass.** We walk the same picture from right to left. Each operation applies one small rule, using only its own local derivative, exactly as described earlier in this part.

*Step 1: the loss.* If `Y` goes up by a little, how much does the loss change? The derivative of `(Y − 20)²` is `2(Y − 20)`, so at `Y = 23` the gradient arriving at `Y` is `2 × 3 = 6`. It's positive, which says "`Y` is too high, so it should come down."

*Step 2: the add.* Addition has the simplest rule of all: **it hands the same gradient to each thing that was added.** Both branches receive `6`.

*Step 3: the multiplications.* This is the rule worth remembering: **for a multiplication, each input's gradient is the incoming gradient times the *other* input.**

```text
For the weights:                         For the inputs:
 gradient for w₁ = 6 × x₁ = 6 × 2 = 12    gradient for x₁ = 6 × w₁ = 6 × 4 = 24
 gradient for w₂ = 6 × x₂ = 6 × 3 = 18    gradient for x₂ = 6 × w₂ = 6 × 5 = 30
```

Why "the other input"? Because in `x₁ · w₁`, if `w₁` rises by 1, the product rises by `x₁`, and if `x₁` rises by 1, the product rises by `w₁`. Then the chain rule multiplies by the `6` that arrived from above.

Putting the whole backward pass on the same picture:

```text
Gradient at Y = 6
        │
        ▼
      add ─────────► 6 goes to both branches
        │
   ┌────┴────┐
   ▼         ▼
 × (x₁,w₁)  × (x₂,w₂)
  │    │      │    │
  ▼    ▼      ▼    ▼
 w₁:12  x₁:24 w₂:18 x₂:30
```

**Checking that it's right.** We don't have to take these numbers on trust. Nudge `w₁` from `4` to `4.01`: then `Y = 2 × 4.01 + 15 = 23.02`, and the loss becomes `(3.02)² = 9.1204`. It went up by about `0.12`, which is `12 × 0.01`. The gradient `12` predicted exactly that. Nudge `w₂` from `5` to `5.01` instead: `Y = 23.03`, loss `9.1809`, up by about `0.18 = 18 × 0.01`. Also as predicted.

**Using the gradients.** Both weight gradients are positive, meaning *increasing these weights makes the loss worse*, so they should go **down**. A gradient-descent step (Chapter 11) with a learning rate of `0.01`:

```text
w₁ = 4 − 0.01 × 12 = 3.88
w₂ = 5 − 0.01 × 18 = 4.82

New Y = 2 × 3.88 + 3 × 4.82 = 22.22      New loss = (22.22 − 20)² = 4.93   (it was 9)
```

One step took the loss from `9` to about `4.9`.

Notice that **one loss value of 9 turned into a separate gradient for each parameter** (`12` and `18`), plus gradients for the input (`24` and `30`). These two kinds of gradient have completely different purposes:

```text
          Gradient from above
                  │
                  ▼
             Matrix multiply
              /           \
             ▼             ▼
   Gradient for X      Gradient for W
   (pass backward,     (stored for the optimizer
    to the layer        to update the weights)
    before)
```

- **Parameter gradients** (for `W`) are what the optimizer uses to actually change the model.
- **Input gradients** (for `X`) are never used to change `X`. They exist only so the *previous* layer can compute *its* gradients.

The same pattern happens for `Q = Embedding × W_Q`: one gradient continues back toward the embedding, and one is stored for `W_Q`.

### How one loss reaches every parameter

Run attention backward, and the gradient **branches** at every step:

```text
                   Loss
                    │
                   W_O ──► Grad W_O
                    │
              Weighted sum ──────► V ──► W_V ──► Grad W_V
                    │
                 Softmax
                    │
               Dot product
                 /     \
               Q         K
               │         │
              W_Q       W_K
               │         │
          Grad W_Q   Grad W_K
```

The loss starts as one number, but every operation asks *"how much of this error belongs to each of my inputs?"*, so it keeps splitting until every learnable matrix has its own gradient. In a full block, the residual connections (Chapter 7) split gradients, LayerNorm passes them along, the FFN produces gradients for `W1` and `W2`, attention produces gradients for `W_Q`, `W_K`, `W_V` and `W_O`, and the embedding table gets its own. Our entire list of learned parameters comes alive at once.

One more precision: it's tempting to say "the loss flows backward", but what actually flows backward is the **gradient of the loss**, not the loss value `2.73` itself.

And remember: backpropagation **only produces gradients**. It never changes a single weight. That's the next step's job.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "The model figures out whether to raise or lower a weight by trial and error." | The gradient tells it mathematically, for every parameter at once, in one backward pass. Trying weights one at a time would take years per step. |
| "Backpropagation updates every layer." | Every operation passes gradients backward, but only operations with parameters produce gradients that get used for updates. Backprop itself never changes any weight. |
| "The loss value flows backward through the network." | What flows backward is the *gradient of* the loss, computed step by step with the chain rule. |

---

## Quick reference

```text
Gradient:        ∂Loss/∂w — sensitivity of the loss to each parameter
Backprop:        chain rule through the graph; every op passes gradients backward,
                 only ops with parameters get updated
```

---

## What's next

Backpropagation produces a gradient for every parameter, but it never changes a single weight. Chapter 11 covers the step that does: gradient descent and its modern replacement AdamW, plus the mini-batches and epochs that make training on huge datasets practical.
