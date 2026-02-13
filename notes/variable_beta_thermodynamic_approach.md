# Variable β (Inverse Temperature) in Climate-SFC Models

**Date:** 2026-02-12
**Status:** Research Note - Theoretical Foundation
**Keywords:** Thermodynamic Tensor Contraction, Simulated Annealing, Accounting Integrity, Optimization Resilience

## Executive Summary

Variable β (inverse temperature, β = 1/T) is not merely a computational convenience—it is the **mathematical bridge** between structural accounting integrity and computational tractability in climate-SFC models.

By treating β as a learnable or controllable parameter during optimization, we can:

1. **Escape oscillatory behavior** ("chattering") by smoothing binary triggers into differentiable sigmoids
2. **Navigate non-convex policy spaces** through annealing (simulated annealing approach)
3. **Maintain accounting rigor** while enabling gradient-based optimization
4. **Solve the "constant productivity" problem** by treating technical coefficients as differentiable manifolds

This note explains why variable β is essential for transforming Stella's discrete, brittle logic into resilient, optimizable economic models.

---

## 1. The Chattering Problem: Why Stella Oscillates

### Stella's Discrete Architecture

Your Stella model exhibits documented **oscillatory behavior**. This is not a bug; it's a feature of discrete-time simulation with binary triggers.

**The Loop:**
1. Model steps forward: Year t → Year t+1
2. Unemployment check: Is U > U*? (Heaviside function: yes/no)
3. If yes → Trigger massive correction (e.g., government spending spike)
4. Overcorrection → Next step: U < U*
5. Trigger reverses → Opposite correction
6. System rings like a bell, unable to settle

**Mathematical Root:**
The system is trying to find an equilibrium, but the discrete stepping and binary logic create **discontinuous jumps** in the phase space.

### Visualization

```
Unemployment (U)
     ↑
 U*  |     ╱╲        ╱╲        ╱╲        ╱╲
     |    ╱  ╲      ╱  ╲      ╱  ╲      ╱  ╲
     |   ╱    ╲    ╱    ╲    ╱    ╲    ╱    ╲
     |  ╱      ╲  ╱      ╲  ╱      ╲  ╱      ╲
     |╱________╲╱________╲╱________╲╱________╲
     └──────────────────────────────────────→ Time

     This is "Chattering": System overshoots equilibrium
     repeatedly because triggers are too sharp.
```

### Why Traditional Damping Fails

In Stella, you manually add "smoothing constants" to damp oscillations:
- Increase time delays
- Add averaging/smoothing blocks
- Adjust lookup table shapes

**Problem:** These are ad-hoc patches that don't scale. Every new scenario requires re-tuning. The model becomes fragile.

---

## 2. The β Solution: Smoothing Binary Logic

### From Heaviside to Sigmoid

A Heaviside (binary) trigger looks like:

$$H(x) = \begin{cases} 0 & \text{if } x < 0 \\ 1 & \text{if } x \geq 0 \end{cases}$$

This is **non-differentiable** (discontinuity at x=0).

We replace it with a **sigmoid** (controlled by β):

$$\sigma_\beta(x) = \frac{1}{1 + e^{-\beta x}}$$

**Key Properties:**
- $\beta = 0$: Completely flat (no trigger effect)
- $\beta = 1$: Smooth S-curve (gradient everywhere)
- $\beta = 10$: Steeper S-curve (approaching hard switch)
- $\beta \to \infty$: Approaches true Heaviside

### The Derivative: Where Gradients Flow

The sigmoid has a smooth derivative:

$$\frac{d\sigma_\beta}{dx} = \beta \cdot \sigma_\beta(x) \cdot (1 - \sigma_\beta(x))$$

This is **always finite** and **always non-zero** (except at extremes).

**Consequence:** Gradients can flow through the trigger. The optimizer can "feel" how changes in policy affect the trigger, and adjust accordingly.

### Applied to Unemployment

Replace:
```stella
IF Unemployment > Target
  THEN Government_Spending = Emergency_Level
  ELSE Government_Spending = Normal_Level
```

With:
```pytorch
trigger = sigmoid(beta * (Unemployment - Target))
Gov_Spending = Normal_Level + trigger * (Emergency_Level - Normal_Level)
```

**Result:** Government spending transitions **smoothly** from normal to emergency levels, preventing sudden shocks that cause oscillations.

---

## 3. Escaping Non-Convexity: Simulated Annealing

### The Landscape Problem

Climate-SFC models are **highly non-convex**. The policy space contains many local minima:

- **Debt-trap steady states:** High unemployment, stable but suboptimal
- **Climate-collapse regimes:** Low emissions but economic ruin
- **Growth-at-any-cost paths:** High GDP, catastrophic climate damage
- **True Pareto frontier:** Optimal balance (hard to find)

Traditional gradient descent gets stuck in the first local minimum it encounters.

### Simulated Annealing via Variable β

The idea: Use β as a **temperature control** to jump over local minima.

**Phase 1: High Temperature (Low β)**
- β = 0.1
- Sigmoid triggers are very fuzzy (nearly flat)
- The loss landscape is **smooth and convex-ish** (gradient is gentle everywhere)
- Optimizer can explore broadly
- May not be precise about accounting boundaries

**Phase 2: Cooling (Increase β gradually)**
- β = 1 → 5 → 10 → 50
- Sigmoid triggers gradually sharpen
- Loss landscape becomes more complex as true constraints emerge
- Optimizer follows gradients into deeper valleys
- Discovers structure that wasn't visible at high temperature

**Phase 3: Low Temperature (High β)**
- β → ∞
- Sigmoid becomes nearly Heaviside (true accounting logic)
- Loss landscape is exact (reflects true model)
- Optimizer is in precise location respecting accounting integrity
- Policy is actionable (no fuzzy boundaries)

### Analogy: Physical Annealing

In metallurgy, you heat a metal (high temperature, atoms move freely) and cool it slowly. The atoms settle into a **low-energy configuration**. Rapid cooling creates defects; slow cooling creates optimal structure.

Economics is similar: Start fuzzy (explore), cool gradually (sharpen), end precise (act).

---

## 4. The "Obsidian Snap": Bridging Learning and Action

### The Metaphor (from Paper 31)

The "Obsidian Snap" is the moment when abstract exploration becomes concrete reality. In climate-SFC models:

**Fuzzy Phase (Low β):**
- Model is learning what policies could work
- Gradients guide exploration
- Bank solvency is a "soft constraint" (can be ~0.8 before "defaulting")
- Allows the optimizer to explore trade-offs

**Hard Phase (High β):**
- Model becomes a stress-test tool
- Binary logic is restored (bank either solvent or insolvent)
- Gradients are sharp (telling you exactly when default occurs)
- Policy is actionable (no ambiguity about financial system)

### Example: Bank Solvency

At low β, you might have:
- Default Probability = sigmoid(β * (Debt/Equity - threshold))
- At β=0.1: Bank is "40% likely to default" (fuzzy state)

At high β:
- Default Probability = sigmoid(β * (Debt/Equity - threshold))
- At β=50: Bank is either "99% likely to default" or "0.1% likely" (nearly binary)

The optimizer discovers the policy that keeps banks in the sharp "not in default" regime, respecting real accounting boundaries.

---

## 5. Solving "Constant Productivity"

### The Problem in LowGrow

Your Stella metadata notes that "Prop of GDP from labour productivity" is **constant**. This suggests the model cannot find an alternative sectoral allocation that achieves climate targets without breaking.

**Why?** Because the technical coefficients matrix A is **fixed**. The model has no way to express "what if we reallocated labor from brown to green sectors?"

### Treating A as a Differentiable Manifold

With variable β, we can:

1. **Thaw** A (set β low, make A soft/learnable)
2. **Let gradients explore** sectoral reallocations
3. **Search for efficiency gains** that hit climate targets
4. **Freeze** the new allocation (increase β until A becomes nearly fixed)

**Example:**
$$A_{\text{damaged}}(T, \beta) = A_{\text{baseline}} \otimes \text{SoftMax}_\beta(D(T))$$

Where:
- Low β: A is highly malleable, can shift between sectors freely
- High β: A locks into discovered sectoral structure

The optimizer finds: "To hit net-zero, we need 40% of labor in green sectors, 30% in maintenance, 20% in traditional, 10% in support."

Then, with high β, this structure becomes **locked** (near-constant), respecting the reality that sectoral transitions are slow.

### Information Flow

$$\text{Gradient of Climate Objective} \to \text{Optimal Sectoral A} \to \text{Structural Constraint}$$

At low β, this chain is smooth (learning phase). At high β, it's rigid (stability phase).

---

## 6. Multi-Objective Pareto Frontier Navigation

### The Non-Convex Frontier

Climate-SFC models optimize three competing objectives:
- Minimize unemployment (maximize output)
- Minimize emissions
- Minimize financial instability

These objectives are **Pareto-incompatible**: You can't achieve all three perfectly simultaneously.

### How Variable β Helps

**Low β (Exploration):**
- Can "see" all three objectives simultaneously
- Gradient points toward general "good direction"
- Escapes suboptimal local minima (debt-trap regimes)

**High β (Refinement):**
- Objectives become more precisely defined
- Sharp gradients show exact trade-off surfaces
- Optimizer settles at Pareto-optimal point respecting accounting

**Sweeping β:**
By running multiple optimizations at different β values, you trace out the **full Pareto frontier**:
- High emission target at low β → Gradient points toward climate action
- Higher β reveals: "To achieve net-zero, unemployment must rise 1.5%"
- Even higher β shows: "At this unemployment, banks become insolvent; need larger green stimulus"

The frontier emerges as β increases.

---

## 7. Implementation Strategy: The β Schedule

### Conceptual Algorithm

```
Initialize: policy_params ← random
            β ← β_initial (low, e.g., 0.1)

For epoch = 1 to N:
    1. Forward pass:
       - Compute model with current β
       - Triggers are fuzzy, accounting soft

    2. Compute loss L(policy_params, β)

    3. Backward pass:
       - gradients = ∇L w.r.t. policy_params
       - Update: policy_params ← policy_params - η * gradients

    4. Anneal temperature:
       - β ← β * (1 + cooling_rate)
       - As β increases, model "hardens"

Return: policy_params* (discovered optimal policy)
```

### Practical β Schedule

**Inverse Temperature Schedule:**

$$\beta(t) = \beta_0 \cdot \exp\left(\frac{t}{t_{\text{anneal}}}\right)$$

Where:
- $\beta_0 = 0.1$ (start fuzzy)
- $t_{\text{anneal}} = 100$ (annealing timescale)
- At t=100: β ≈ 2.7 (moderately hard)
- At t=200: β ≈ 7.4 (quite hard)
- At t=400: β ≈ 55 (nearly rigid)

**Alternative: Piece-wise Schedule**

```
Epochs 0-100:   β = 0.5  (Exploration phase)
Epochs 100-200: β = 2.0  (Transition phase)
Epochs 200-300: β = 10   (Refinement phase)
Epochs 300+:    β = 100  (Final hardening phase)
```

---

## 8. Practical Benefits in LowGrow Reconstruction

### Before (Pure Stella)

```
+---------+
| Stella  |  ← Black box
+---------+
    ↓
 Output: Single scenario
 Problem: Oscillations, manual tuning
 Action: Add more damping constants
 Result: Fragile, unpredictable
```

### After (Differentiable with Variable β)

```
+---------------------------+
| PyTorch SFC with β        |  ← White box, differentiable
+---------------------------+
    ↓
 Output: Full Pareto frontier
 Process: Automated annealing
 Action: Gradient descent finds policy
 Result: Robust, interpretable, actionable
```

### Specific Advantages for LowGrow

1. **Eliminate oscillations:** Start with β=0.1, let optimizer smooth out triggers
2. **Find productivity allocation:** Optimize sectoral A with low β, freeze at high β
3. **Multi-objective governance:** Sweep β to show unemployment-emissions trade-off
4. **Validate accounting:** High β ensures final policy respects balance sheets
5. **Real-time policy cockpit:** Adjust β in real-time to balance exploration vs. precision

---

## 9. Comparison: Standard vs. Variable β Approaches

| Aspect | Standard Differentiable | Variable β Approach |
|--------|------------------------|-------------------|
| **Starting Point** | Fixed β (preset fuzziness) | β = 0.1 (very fuzzy) |
| **Optimization** | Direct gradient descent | Annealing schedule |
| **Local Minima** | Often trapped | Escape via high-T exploration |
| **Accounting Precision** | Approximate always | Exact at high β |
| **Policy Actionability** | Fuzzy (30% chance default?) | Clear (default or not) |
| **Frontier Exploration** | Single point | Full Pareto frontier |
| **Computational Cost** | One optimization run | Multiple runs (with different β) |
| **Interpretability** | "Why this policy?" (hard) | "What trade-offs exist?" (clear) |

---

## 10. Why This Isn't "Hammer Looking for Nail"

### The Structural Need for β

Variable β addresses a **fundamental gap** in Stella's architecture:

**Stella's Design:** Binary logic + discrete stepping = discontinuous system
**Problem:** Gradient-based optimization requires smooth functions
**Solution Options:**
1. Smooth everything permanently (lose accounting precision)
2. Use only approximate methods (lose accuracy)
3. **Use variable β (best of both worlds)**

### Evidence This is Essential

1. **Empirical:** Your Stella model shows documented oscillatory behavior → needs damping
2. **Theoretical:** Non-convex policy spaces require annealing → need temperature control
3. **Practical:** LowGrow's "constant productivity" suggests fixed A matrix → needs softening
4. **Mathematical:** Sigmoid is differentiable approximation to Heaviside → naturally use β

Variable β is not optional; it's the **minimal sufficient framework** for turning discrete SFC into differentiable economics.

---

## 11. Implementation Roadmap

### Phase 1: Single-Temperature Baseline (X4, Current)
- [ ] Implement Green-SFC with fixed β = 20.0
- [ ] Verify smooth optimization (no oscillations)
- [ ] Validate accounting integrity at high β

### Phase 2: Annealing Algorithm
- [ ] Add β as dynamic parameter
- [ ] Implement annealing schedule
- [ ] Test on simplified 10-year scenario
- [ ] Visualize loss landscape as β increases

### Phase 3: Pareto Frontier Generation
- [ ] Multiple runs with different β trajectories
- [ ] Sweep weights (λ₁, λ₂, λ₃) to find frontier
- [ ] Produce 2D/3D frontier plots (unemployment vs. emissions vs. financial stability)

### Phase 4: LowGrow Reconstruction
- [ ] Use Stella parser to extract equations
- [ ] Implement in PyTorch with variable β
- [ ] Reproduce published LowGrow results
- [ ] Show oscillations disappear with annealing

### Phase 5: Policy Cockpit
- [ ] Real-time β adjustment
- [ ] Live frontier visualization
- [ ] Integration with national accounts data

---

## 12. Open Research Questions

1. **Optimal Annealing Schedule:** What β(t) schedule is most efficient? Can we learn it adaptively?

2. **Convergence Guarantees:** Under what conditions does the algorithm converge to true Pareto frontier?

3. **Structural Change:** Can variable β handle discontinuous shifts (e.g., sudden carbon tax policy)?

4. **Computational Scaling:** How does computational cost grow with model size (sectors, time horizon, state variables)?

5. **Uncertainty Quantification:** How do we report confidence intervals on policy recommendations when β annealing introduces non-uniqueness?

---

## 13. Conclusion: From Brittle to Resilient

**The Core Insight:**

By using variable β as a **thermodynamic control**, we transform Stella's discrete, brittle system into a differentiable, resilient manifold that can be **learned, optimized, and verified**.

This is the missing piece that allows:
- Stella's rigorous accounting → PyTorch's gradient power
- Discrete logic → continuous optimization
- Single scenario → full Pareto frontier
- Manual tuning → automated discovery
- Fragile model → actionable policy tool

**In Short:** Variable β is not a hammer looking for a nail. It's the **essential bridge** between the discrete world of financial accounting and the continuous world of optimization.

---

## References

**Thermodynamic Methods:**
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671-680.
- Frisch, U. (1968). Probability and combinatorics: A probabilistic view of optimization in statistical mechanics.

**Differentiable Economics:**
- Godley, W., & Lavoie, M. (2012). *Monetary Economics: An Integrated Approach*.
- Paszke, A., et al. (2019). PyTorch: An Imperative Style High-Performance Deep Learning Library.

**Climate-Economic Models:**
- Stern, N. (2006). *The Economics of Climate Change: The Stern Review*.
- Recalibrating Climate Risk (2024). Working paper on tipping point dynamics.

**Tropical Geometry (Soft-Min):**
- Maclagan, D., & Sturmfels, B. (2015). *Introduction to tropical geometry*.
- Gondran, M., & Minoux, M. (2008). *Graphs, dioids, and semirings*.

---

## Related Notes/Experiments

- **notes/differentiable_green_sfc_architecture.md** — Three-layer architecture (uses β implicitly)
- **notes/differentiable_define_model.md** — DEFINE integration (soft-defaults via β)
- **X4 (green_sfc.py)** — Current implementation with fixed β
- **lib/semiring_engines.py** — TropicalSemiringSupplyChain uses β for soft-min

---

**Next Step:** Implement X5 with dynamic β schedule, demonstrating Pareto frontier generation for LowGrow reconstruction.
