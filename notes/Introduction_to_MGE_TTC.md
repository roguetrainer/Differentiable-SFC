# Introduction to Maslov-Gibbs Einsum (MGE-TTC)

The **Maslov-Gibbs Einsum (MGE)**, or **Thermodynamic Tensor Contraction (TTC)**, is a mathematical framework that bridges two worlds: the discrete world of "Hard Logic" (If/Then, Min/Max) and the continuous world of "Neural Learning" (Gradients, Optimization).

## 1. The Core Concept: Thawing the Logic

In classical economic models (like Stella or standard I-O), decisions are often "frozen" in discrete steps:

* **A Bottleneck:** "The output is exactly the **MIN** of the inputs."
* **A Policy Trigger:** "**IF** unemployment > 5%, **THEN** spend $10B."

Even in Standard Semi-ring models (like SFC), these "Hard Triggers" create mathematical "cliffs." If you are at 4.9% unemployment, the math sees no reason to act. If you hit 5.1%, it snaps instantly. Because these logic gates are flat (horizontal) or vertical, they have no **gradient**. A computer cannot "feel" its way toward a better solution because the terrain has no slope.

**MGE-TTC** uses a technique called **Dequantization** to turn these "cliffs" into "slopes" by treating logic as a thermodynamic state.

## 2. Two Families of Models: Standard vs. Tropical

Within the MGE framework, we distinguish between two fundamental types of economic logic based on their underlying "Semi-ring" algebra.

### A. The Standard Semi-ring (Accounting/Volume Models)

These models operate on the arithmetic of **addition and multiplication** (ℝ, +, ×).

* **Focus:** They track **Volumes** and monetary flows.
* **The "Frozen" Problem:** While the arithmetic is continuous, the *decisions* (like tax thresholds) are often discrete steps that block backpropagation.

### B. The Tropical Semi-ring (Structural/Bottleneck Models)

These models operate on the "Min-Plus" algebra (ℝ ∪ {∞}, min, +).

* **Focus:** They track **Constraints** and "weakest-link" dependencies.
* **The "Frozen" Problem:** The `min` operator is inherently non-differentiable at the point where the bottleneck shifts from one component to another.

## 3. Why Interpolate? The Need for the MGE

You might ask: *If my SFC model already uses standard arithmetic, why do I need the MGE?*

The answer lies in **Structural Hybridity**. Real-world systems (like the Canadian economy) are not purely one or the other.

1. **The Gradient Signal:** We use the MGE to turn "Hard" standard-logic triggers into "Soft" differentiable reaction functions. This creates the gradient necessary for backpropagation.
2. **Phase Transitions:** We interpolate between Standard and Tropical logic because climate-economic systems undergo **Phase Transitions**. In normal times, an economy is additive (Standard); during a supply chain collapse or a "Minsky Moment" of debt, it becomes governed by bottlenecks (Tropical).
3. **The Unified Operator:** The MGE allows a single model to "morph" its logic. By varying β, we can see how an additive monetary flow (Standard) suddenly hits a physical resource limit (Tropical), creating the "Obsidian Snap."

## 4. The Role of Beta (β): The Computational Thermostat

The variable β is the "Inverse Temperature" (1/T) of the model's logic. By adjusting β, we control how "solid" or "liquid" the logic behaves.

* **Low β (High Temperature / "Liquid Logic"):** The model is fuzzy. A "Min" operator doesn't just pick the smallest number; it takes a weighted average. A "Hard Trigger" becomes a smooth ramp.
  * *Benefit:* Gradients flow everywhere. The optimizer can "see" a tipping point from miles away and start steering early.
* **High β (Low Temperature / "Frozen Logic"):** As β increases, the curves sharpen. The "fuzzy" average settles back into a strict "Min" or a hard "If/Then" rule.
  * *Benefit:* This represents the real-world "Hard Logic" of accounting and physical constraints.

## 5. The MGE Formula: Soft-Min via Sigmoid

The core mathematical operation of MGE-TTC is the **soft minimum** function, which approximates discrete logic using a sigmoid (temperature-controlled).

### Discrete Hard Trigger (Non-Differentiable)

```python
if (x > threshold):
    output = 1
else:
    output = 0
```

**Problem:** Zero gradient almost everywhere; infinite gradient at kink. Backpropagation fails.

### Differentiable Soft-Min via Sigmoid (MGE Formula)

The **Maslov-Gibbs approximation** replaces discrete logic with a sigmoid:

**Universal TTC Operator (Paper 31):**

$$Y_j = \frac{1}{\zeta} \log \left( \sum_i \exp(\zeta \cdot (W_{ij} + X_i)) \right)$$

Where:

* **ζ = β + iγ** = Complex tropical number (β: inverse temperature, γ: topological phase)
* **W, X** = Input weights and values
* **Log-Sum-Exp** = Unique operator bridging arithmetic (+,×) and tropical (max,+) semirings

**Application: Default Probability (Financial Crisis Example):**

```
P_d(x, β) = σ(β · (x - threshold))
          = 1 / (1 + exp(-β · (x - threshold)))
```

Where:

* **x** = trigger variable (e.g., debt-service ratio, unemployment rate)
* **threshold** = critical value where system transitions
* **β** = inverse temperature (controls sharpness)
* **σ(·)** = sigmoid function (special case of TTC)

### General MGE Soft-Min Formula (Tropical Semi-ring)

For multiple competing constraints (bottlenecks), the soft minimum is:

```
soft_min_β(x₁, x₂, ..., xₙ) = -1/β · log(∑ᵢ exp(-β · xᵢ))
```

This is the **LogSumExp trick** from machine learning. As β → ∞, it converges to the true minimum:

```
lim (β→∞) soft_min_β(x₁, x₂, ..., xₙ) = min(x₁, x₂, ..., xₙ)
```

### β-Annealing Schedule

Optimization typically uses a schedule where β increases over epochs:

```
β(epoch) = β₀ · (1 + epoch / τ)^α
```

* **Early epochs:** Low β (e.g., β=1): Fuzzy, smooth gradients
* **Late epochs:** High β (e.g., β=100): Sharp, discrete-like behavior
* **Effect:** Start exploring broadly, then refine to actual critical points

### Key Properties of MGE

| Property | Hard Logic | MGE-TTC Sigmoid |
| --- | --- | --- |
| **Differentiable** | ✗ | ✓ |
| **Gradient Signal** | None (flat/cliff) | Smooth everywhere |
| **Limit β→∞** | N/A | Recovers true logic |
| **Allows Backprop** | ✗ | ✓ |
| **Annealing** | N/A | β(epoch) schedule |

### Implementation Example

```python
import torch
import torch.nn as nn

def sigmoid_trigger(x, threshold, beta=10.0):
    """
    Soft-threshold policy trigger (MGE formula).

    Args:
        x: Trigger variable (e.g., unemployment rate)
        threshold: Critical value
        beta: Inverse temperature (sharpness)

    Returns:
        Differentiable probability in [0, 1]
    """
    return torch.sigmoid(beta * (x - threshold))

def soft_min(values, beta=10.0):
    """
    Soft minimum via LogSumExp (tropical semi-ring).

    Args:
        values: Tensor of competing constraints
        beta: Inverse temperature

    Returns:
        Soft minimum (differentiable bottleneck)
    """
    return -torch.logsumexp(-beta * values, dim=-1) / beta

def mge_annealing_schedule(epoch, beta0=1.0, final_beta=100.0, total_epochs=150):
    """
    β-annealing: Start fuzzy, gradually sharpen.
    """
    progress = min(epoch / total_epochs, 1.0)
    return beta0 + (final_beta - beta0) * progress
```

### Connection to Physical Theory

The formula comes from **statistical mechanics**:

* In statistical mechanics, β = 1/(k_B·T) where k_B is Boltzmann's constant
* At high temperature (low β): System explores many states (high entropy)
* At low temperature (high β): System settles into ground state (low entropy)
* **Economic parallel:** High-β systems exhibit sharp phase transitions (like debt collapse)

## 5. Summary

MGE-TTC turns a **search** problem (finding a needle in a haystack of discrete rules) into a **navigation** problem (walking down a hill toward the best possible outcome). The interpolation between semi-rings via β is what allows us to model a world that is both an accounting ledger and a physical machine.

---

**Source:** Framework documentation for Differentiable SFC Experiments
**Context:** Foundational theory for X7-X8 models and the GEMMES notebook
