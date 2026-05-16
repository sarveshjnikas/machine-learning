# Autograd & Backpropagation
> Notes from Andrej Karpathy's *micrograd* — for revision.

---

## 1. The Setup

A neural network is a function that maps inputs to predictions:

$$\hat{y} = f(x, w)$$

where $x$ is the input and $w$ are the learnable weights.

We measure how wrong the prediction is using a **loss function**:

$$L = \mathcal{L}(\hat{y},\, y)$$

**Goal:** find weights $w$ that minimize $L$.

---

## 2. Gradient — The Core Idea

The gradient tells you how sensitive the output is to a small change in the input:

$$\frac{\partial L}{\partial w} \approx \frac{L(w + \epsilon) - L(w)}{\epsilon} \quad \text{as } \epsilon \to 0$$

- Positive gradient → increasing $w$ increases $L$ → decrease $w$.
- Negative gradient → increasing $w$ decreases $L$ → increase $w$.

---

## 3. Computation Graph

Every mathematical expression can be decomposed into a **directed acyclic graph** where each node is a primitive operation (+, ×, tanh, etc.) and edges carry values (forward) and gradients (backward).

### Example

$$c = a + b, \qquad L = c \cdot f$$

```
a ──┐
    ├──[+]── c ──[×]── L
b ──┘        f ──┘
```

**Forward pass** (compute values, left → right):

| Node | Value |
|------|-------|
| $a$  | 2     |
| $b$  | 3     |
| $f$  | 4     |
| $c = a + b$ | 5 |
| $L = c \times f$ | 20 |

---

## 4. Backpropagation

Backprop is the algorithm that efficiently computes $\partial L / \partial w$ for **all** weights in one backward pass using the **chain rule**.

### Chain Rule

If $L$ depends on $b$, and $b$ depends on $a$:

$$\frac{dL}{da} = \frac{dL}{db} \cdot \frac{db}{da}$$

More generally, for a node $z$ that feeds into downstream nodes $q_1, q_2, \ldots$:

$$\frac{\partial L}{\partial z} = \sum_i \frac{\partial L}{\partial q_i} \cdot \frac{\partial q_i}{\partial z}$$

### Backward Pass (example continued)

Start at the output and work backwards:

$$\frac{dL}{dL} = 1$$

$$\frac{dL}{dc} = f = 4, \qquad \frac{dL}{df} = c = 5$$

$$\frac{dL}{da} = \frac{dL}{dc} \cdot \frac{dc}{da} = 4 \cdot 1 = 4$$

$$\frac{dL}{db} = \frac{dL}{dc} \cdot \frac{dc}{db} = 4 \cdot 1 = 4$$

---

## 5. Local Derivative Rules

Each node only needs to know its **local derivative**. The chain rule does the rest.

### Addition: $z = x + y$

$$\frac{\partial z}{\partial x} = 1, \qquad \frac{\partial z}{\partial y} = 1$$

> Addition is a **gradient distributor** — it passes the upstream gradient unchanged to both inputs.

### Multiplication: $z = x \cdot y$

$$\frac{\partial z}{\partial x} = y, \qquad \frac{\partial z}{\partial y} = x$$

> Multiplication is a **gradient switcher** — each input's gradient is the other input's value, scaled by upstream gradient.

### General pattern

Each node computes:

$$\text{grad\_input} = \text{grad\_output} \times \text{local\_derivative}$$

---

## 6. Autograd

Autograd (as in PyTorch) automates the entire backward pass.

### How it works

1. **Forward pass:** as operations execute, a computation graph is dynamically built. Each tensor tracks its creator operation.
2. **Intermediate values** are stored (needed for backward derivatives, e.g., $\partial(xy)/\partial x = y$).
3. **Backward pass:** calling `loss.backward()` traverses the graph in reverse topological order, applying the chain rule at each node.

```python
# PyTorch example
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x       # builds graph
y.backward()
print(x.grad)             # dy/dx = 2x + 3 = 7
```

---

## 7. Weight Update

Once we have the gradients, we update weights to reduce the loss:

$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$

where $\eta$ (eta) is the **learning rate**.

---

## 8. Learning Rate

The learning rate $\eta$ controls the step size of each update.

| Learning Rate | Effect |
|---|---|
| Too small | Slow convergence, many iterations needed |
| Too large | Overshooting minima, oscillation, instability |
| Way too large | Exploding loss (see below) |

### Exploding Loss — The Chain

$$\text{large } w \;\Rightarrow\; \text{large activations} \;\Rightarrow\; \text{huge } L \;\Rightarrow\; \text{huge } \nabla_w L \;\Rightarrow\; \text{even larger } w$$

This is a positive feedback loop — a death spiral. Use gradient clipping or a smaller $\eta$ to prevent it.

---

## 9. Gradient Accumulation in PyTorch

**Critical behavior:** PyTorch **accumulates** gradients by default.

```python
loss.backward()
# internally: param.grad += new_gradient   ← NOT param.grad = new_gradient
```

### Why accumulation exists (intentional use)

Simulate a larger effective batch size when GPU memory is limited:

```python
for i, (x, y) in enumerate(loader):
    loss = criterion(model(x), y) / accumulation_steps
    loss.backward()                          # gradients accumulate
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Why it's dangerous (unintentional use)

If you forget `zero_grad()`, gradients from previous iterations bleed into the current one:

| Iteration | Gradient | `param.grad` without zero |
|---|---|---|
| 1 | 5 | 5 |
| 2 | 3 | **8** ← wrong |

The model updates on stale, accumulated gradients → incorrect training.

---

## 10. Correct Training Loop

```python
for x, y in dataloader:
    optimizer.zero_grad()          # 1. clear gradients from previous step

    y_hat = model(x)               # 2. forward pass
    loss = criterion(y_hat, y)     # 3. compute loss

    loss.backward()                # 4. backward pass — compute all gradients
    optimizer.step()               # 5. update weights: w = w - lr * grad
```

### What each line does

| Step | Code | Math |
|---|---|---|
| Zero grads | `optimizer.zero_grad()` | $\nabla \leftarrow 0$ |
| Forward | `y_hat = model(x)` | $\hat{y} = f(x, w)$ |
| Loss | `loss = criterion(y_hat, y)` | $L = \mathcal{L}(\hat{y}, y)$ |
| Backward | `loss.backward()` | $\nabla_w L$ via chain rule |
| Update | `optimizer.step()` | $w \leftarrow w - \eta \nabla_w L$ |

---

## Quick Reference

$$\boxed{\frac{dL}{dw} = \frac{dL}{d\text{out}} \cdot \frac{d\text{out}}{dw}}$$

$$\boxed{w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}}$$

| Operation | Forward | Backward (grad to $x$) |
|---|---|---|
| $z = x + y$ | $z = x + y$ | $\delta_x = \delta_z$ |
| $z = x \cdot y$ | $z = xy$ | $\delta_x = \delta_z \cdot y$ |
| $z = x^n$ | $z = x^n$ | $\delta_x = \delta_z \cdot nx^{n-1}$ |
| $z = \tanh(x)$ | $z = \tanh(x)$ | $\delta_x = \delta_z \cdot (1 - z^2)$ |
| $z = \max(0, x)$ (ReLU) | $z = \text{ReLU}(x)$ | $\delta_x = \delta_z \cdot \mathbf{1}[x > 0]$ |