# Differentiable DEFINE: From Simulation to Optimization

**Date:** 2026-02-11
**Status:** Research Note - Strategic Implementation Blueprint
**Related:** Dafermos, Nikolaidi, & Galanis (DEFINE model), P2 (Framework), Differentiable Green-SFC Architecture

## Executive Summary

The DEFINE model (Dafermos, Nikolaidi, & Galanis) is a sophisticated SFC framework that explicitly links the physical carbon cycle with the financial debt-service cycle. However, it's implemented as a traditional discrete simulation (Gauss-Seidel iteration), creating analytical bottlenecks:

- **Oscillations:** Arise from discrete stepping and binary default logic
- **Non-differentiability:** Black-box solver prevents sensitivity analysis
- **Manual calibration:** Parameters tuned by hand, not learned from data
- **Limited optimization:** No automatic discovery of policy paths

This note outlines a **Differentiable DEFINE** architecture that transforms it from a "what-if" simulation engine into a "control-and-calibrate" optimization platform, solving the oscillation and productivity problems that plague your Stella implementation.

## 1. The SFC Core: From Gauss-Seidel to Matrix Gradients

### Traditional DEFINE Solving

The DEFINE model solves the SFC system iteratively:

$$x^{(k+1)} = A x^{(k)} + d$$

using Gauss-Seidel or similar fixed-point algorithms until convergence.

**Properties:**
- Opaque: Convergence is "internal" to the solver
- Non-differentiable: No access to gradients
- Single-valued: Finds one steady state (or oscillation)
- Slow: Requires iteration to convergence

### The Differentiable Shift

Rewrite as a linear system:

$$(I - A) x = d$$

**Solve via:** `torch.linalg.solve(I - A, d)`

**Properties:**
- Transparent: Matrix operations are explicit
- Fully differentiable: PyTorch tracks all gradients
- Analytical: Direct solution (no iteration)
- Fast: O(n³) single solve vs. iterative convergence

### What This Enables

$$\frac{\partial x}{\partial A} = \frac{\partial (I-A)^{-1}d}{\partial A}$$

We can now compute:
- How output responds to changes in behavioral coefficients
- How a policy change in year 1 cascades to year 30
- The exact "economic gradient" of any parameter

**Key Insight:** Backpropagation Through Time (BPTT) flows financial instability from future defaults back to inform current policy.

---

## 2. Differentiable Financial Fragility: Soft-Thresholds

### The Problem: Binary Default Logic

In traditional DEFINE, firms default when:

$$\text{dsr}_t = \frac{\text{Interest Payments}}{{\text{Operating Cashflow}}} > \text{threshold}$$

This is a **binary snap**: either solvent or in default.

**Consequence:** Creates discontinuities in the gradient landscape, leading to:
- Sharp oscillations (system jumps between equilibria)
- Chattering behavior (oscillations around threshold)
- Non-differentiability (zero gradient almost everywhere, infinite at the kink)

This is exactly the "oscillation problem" you encounter in your Stella model.

### The Differentiable Fix: Soft-Min via Sigmoid

Replace the binary default with a **probability of default**:

$$P_d(t) = \sigma(\beta \cdot (\text{dsr}_t - \text{threshold}))$$

Where $\sigma(z) = \frac{1}{1+e^{-z}}$ is the sigmoid function.

**Temperature Control:**
- High $\beta$ (T→0): Sharp threshold, approaches true default condition
  - Correctly identifies collapse point
  - High-curvature gradient landscape

- Low $\beta$ (T→∞): Fuzzy threshold, smooth transition
  - Allows easy gradient flow
  - Identifies gradual instability

### Maslov-Gibbs Connection

This soft-min is the **Maslov-Gibbs (thermodynamic) approximation** to the true default rule:

$$P_d(\beta) = \text{SoftMax}_\beta(\text{dsr}, \text{threshold})$$

As $\beta \to \infty$, this converges to the hard binary logic.

**The Insight:** By making $\beta$ a learnable or annealing parameter, we can:

1. **Explore** (high T, low $\beta$): Discover feasible policy configurations
2. **Sharpen** (cooling, increase $\beta$): Identify collapse points and critical transitions
3. **Anneal**: Navigate from fuzzy to hard logic, finding the transition point

### Implementation in PyTorch

```python
def soft_default_probability(dsr, threshold, beta=10.0):
    """
    Differentiable default probability using sigmoid.

    dsr: Debt-Service Ratio (firm-level)
    threshold: Default threshold (e.g., 1.0)
    beta: Inverse temperature controlling softness
    """
    return torch.sigmoid(beta * (dsr - threshold))

def firm_solvency(operating_cashflow, default_prob):
    """
    Solvency is (1 - P_d).
    This flows back through BPTT to affect cashflow.
    """
    return operating_cashflow * (1.0 - default_prob)
```

---

## 3. Endogenizing the Physical-Financial Link

### The DEFINE Feedback Loop

DEFINE's power comes from explicitly modeling:

**Climate → Physical Damage → Firm Profitability → Financial Fragility → Macro Instability**

Traditional DEFINE treats this as:
1. Climate scenario (exogenous, prescribed path)
2. Calculate damage
3. Reduce firm output/profitability
4. Monitor defaults

**Problem:** The coupling is one-way (climate → finance). The economy's response doesn't feed back to climate (policy adaptation, structural change).

### Differentiable Bidirectional Coupling

#### Standard Semi-ring (Volume Effects)

Model how temperature affects sectoral efficiency:

$$A_{\text{damaged}}(T) = A_{\text{baseline}} \cdot (1 + \gamma \cdot D(T))$$

Where $D(T) = 1 - \frac{1}{1 + e^{s(T-T_{\text{thresh}})}}$ is the sigmoid damage function.

**Effect:** Rising temperature increases the technical coefficient matrix, meaning "more inputs needed per unit output."

#### Tropical Semi-ring (Structure Effects)

The Energy Transition creates supply-chain bottlenecks:

- Green capital availability is limited (construction rates, supply constraints)
- Brown capital must be retired (stranded assets)
- Structural mismatch creates production inefficiencies

Model as tropical min-plus:

$$\text{Output}_{\text{available}} = \min(\text{Output}_{\text{demand}}, \text{Green\_Capital}_{\text{stock}})$$

**Effect:** Even if firms want to invest in green capital, the actual transition is bottleneck-constrained.

#### The Sensitivity Gradient

We can now compute:

$$\frac{\partial \text{Bank Solvency}}{\partial \text{Carbon Emissions}} = \frac{\partial \text{Bank Solvency}}{\partial \text{Default}} \times \frac{\partial \text{Default}}{\partial \text{Firm Profitability}} \times \frac{\partial \text{Profitability}}{\partial \text{Damage}} \times \frac{\partial \text{Damage}}{\partial T} \times \frac{\partial T}{\partial E}$$

This entire chain is computed automatically by backpropagation.

**Interpretation:** The exact sensitivity of global banking system solvency to emissions trajectory.

---

## 4. Automated Calibration via Inverse Modeling

### Traditional DEFINE Calibration

DEFINE has dozens of parameters:
- $\alpha_1$: Propensity to consume (income)
- $\alpha_2$: Propensity to consume (wealth)
- $\delta$: Depreciation rate
- $\rho_d$: Debt sensitivity
- $\mu_e$: Emission intensity by sector
- ... many more

**Current Practice:** Manually set to match economic history or "reasonable" values. Then run the model and observe behavior.

**Problem:** No systematic way to ensure consistency with actual data. Calibration is art, not science.

### Differentiable Approach: Learning from Data

Treat parameters as **learnable weights**. Define a loss function:

$$L_{\text{calib}} = \sum_t \| \text{Model}_t(\theta) - \text{Data}_t \|^2$$

Where:
- $\text{Data}_t$ = actual national accounts, financial data, emissions data
- $\text{Model}_t(\theta)$ = model predictions for parameters $\theta$

**Run optimization:**

$$\theta^* = \arg\min_\theta L_{\text{calib}}$$

**Result:** The model "learns" parameter values that are consistent with historical data.

### What Data to Use?

For Canadian/Global DEFINE:
- National accounts (SNA): GDP, sectoral outputs, consumption, investment
- Balance sheets: Firm debt, equity, asset stocks
- Flows: Interest payments, corporate cashflows
- Climate: Emissions by sector, fuel use

All of this can be organized into a loss function and used to calibrate $\theta$ via Adam or SGD.

---

## 5. The "Policy Cockpit" Loss Function: Sustainability Frontier

### Multi-Objective Optimization

Policy-makers face competing objectives. The Differentiable DEFINE can optimize all simultaneously using a weighted loss function:

$$L_{\text{policy}}(\theta_{\text{pol}}, T) = \lambda_1 \cdot L_{\text{employment}} + \lambda_2 \cdot L_{\text{emissions}} + \lambda_3 \cdot L_{\text{stability}} + \lambda_4 \cdot L_{\text{debt}}$$

Where:

**$L_{\text{employment}}$** (Unemployment Gap)
$$L_{\text{empl}} = \sum_t (U_t - U_{\text{target}})^2$$

**$L_{\text{emissions}}$** (Climate Mitigation)
$$L_{\text{emiss}} = \sum_t (E_t - E_{\text{target}})^2$$

**$L_{\text{stability}}$** (Macro Stability)
$$L_{\text{stab}} = \sum_t [\text{Var}(Y_t) + \text{Var}(\pi_t) + \text{Var}(r_t)]$$

**$L_{\text{debt}}$** (Financial Sustainability)
$$L_{\text{debt}} = \sum_t (\text{Debt}_t / \text{GDP}_t - \text{Target})^2 + \text{Var}(P_d)$$

### The Policy Parameters $\theta_{\text{pol}}$

The optimizer adjusts:
- **Tax rates:** $\tau_t^{\text{income}}, \tau_t^{\text{corporate}}, \tau_t^{\text{carbon}}$
- **Government spending:** $G_t, G_t^{\text{green}}$ (green investment allocation)
- **Monetary policy:** Interest rate rule $r_t = r_0 + \beta_\pi(\pi_t - \pi^*) + \beta_y(Y_t - Y^*)$
- **Structural policy:** Green capital subsidy, brown capital tax, etc.

### Solution Interpretation

The optimized policy $\theta_{\text{pol}}^*$ represents the **Pareto frontier** of feasible policies:

- **If $\lambda_1$ is high:** Employment-heavy solution (may sacrifice emissions targets)
- **If $\lambda_2$ is high:** Emissions-heavy solution (may sacrifice employment)
- **If $\lambda_3$ is high:** Stability-heavy solution (may require counter-cyclical fiscal policy)

By sweeping $\lambda$ weights, we generate the full policy frontier, showing trade-offs.

### Computing Policy Gradients

Once optimized, we can ask:

$$\frac{\partial L_{\text{total}}}{\partial \tau_t^{\text{carbon}}}$$

"What's the total impact (across all objectives) of a 1% increase in carbon tax in year t?"

This is computed via BPTT across the entire 50+ year horizon.

---

## 6. Why This Solves Your Stella Oscillations

### The Root Cause

Your Stella model exhibits oscillations because:

1. **Discrete stepping:** Each year is a hard timestep; transitions are abrupt
2. **Binary thresholds:** If/then logic creates kinks in the solution space
3. **Feedback delays:** Stock-flow lags cause overshooting and undershoot
4. **No damping:** Manual damping constants don't generalize

**Result:** System rings like a bell, unable to settle to equilibrium.

### The Differentiable Solution

| Problem | Traditional | Differentiable |
|---------|-------------|----------------|
| **Oscillations** | Manual damping constants | Variable $\beta$ soft-min; gradients naturally find sweet spot |
| **Binary Logic** | Hard if/then default | Sigmoid soft-default; smooth gradient everywhere |
| **Productivity** | Fixed technical coefficients | $A(T)$ endogenous to climate; gradients drive structural change |
| **Policy Discovery** | Trial-and-error trigger values | Automatic gradient descent finds optimal policies |
| **Calibration** | Manual parameter tuning | Learn from data via inverse modeling |
| **Multi-objective** | Analyst judgment | Loss function optimization with weighted trade-offs |

### Key Mechanism: Variable Temperature ($\beta$)

The oscillation problem is fundamentally about the model's **phase space**:

- At low $\beta$ (high T): The model is "fuzzy"—many policies seem viable, gradients are gentle
- At high $\beta$ (low T): The model is "sharp"—discontinuities matter, gradients are steep

By annealing $\beta$ during optimization:
1. **Start hot:** Explore the broad policy space
2. **Cool gradually:** Sharpen toward the true equilibrium
3. **End cold:** Settle at the precise collapse point

This is how physics simulations identify phase transitions. Applied to economics, it identifies systemic tipping points.

---

## 7. Implementation Roadmap

### Phase 1: Prototype (Weeks 1-4)

- [ ] Implement basic DEFINE core as `StandardSemiringIO` in lib/
- [ ] Add simple default rule with sigmoid soft-threshold
- [ ] Replicate DEFINE results for baseline Canadian scenario
- [ ] Verify: No oscillations in differentiable version

### Phase 2: Climate Integration (Weeks 5-8)

- [ ] Add Layer 1 (climate damage function)
- [ ] Implement temperature-dependent $A(T)$
- [ ] Add tropical supply chain constraints
- [ ] Replicate DEFINE's "debt-deflation spiral" results

### Phase 3: Optimization (Weeks 9-12)

- [ ] Implement multi-objective loss function
- [ ] Add policy parameter learning (tax, spending rules)
- [ ] Test policy optimization: find optimal carbon tax path
- [ ] Benchmark against manual DEFINE calibration

### Phase 4: Validation (Weeks 13-16)

- [ ] Calibrate to actual Canadian national accounts
- [ ] Test against historical shocks (financial crisis, climate events)
- [ ] Sensitivity analysis: parameter uncertainty
- [ ] Deliver policy recommendations

---

## 8. Open Technical Questions

### 1. Uniqueness of Solutions

The policy landscape is non-convex. Multiple local minima likely exist.

**Question:** Can we guarantee finding the global optimum?

**Approaches:**
- Ensemble optimization (multiple random initializations)
- Simulated annealing ($\beta$ scheduling)
- Convex relaxation (loosen some constraints, solve, then tighten)

### 2. Stability Under Structural Break

DEFINE's parameters may shift if the economy undergoes structural change (green transition, technological leap).

**Question:** How stable are learned parameters under out-of-sample climate scenarios?

**Approach:** Cross-validation—calibrate on 1990-2000, test on 2000-2010, etc.

### 3. Real-Time Optimization

Can we solve the optimization problem fast enough for real policy decisions?

**Current estimate:** Likely requires GPU; expect minutes to hours for full 50-year horizon.

**Question:** Can we achieve real-time ("seconds") solutions?

**Approaches:**
- Dimensional reduction (fewer sectors, aggregated model)
- Surrogate models (train neural net on solutions, query instead)
- Approximation methods (first-order gradient methods vs. second-order)

### 4. Interpretability

When the optimizer discovers novel policies, how do we explain them to policymakers who expect intuition?

**Example:** The model might recommend a carbon tax path that zigzags (rising, dipping, rising again) in unintuitive ways.

**Question:** Can we add regularization to prefer "smooth" policy paths?

**Approach:** Add penalty term: $\lambda_{\text{smooth}} \sum_t (\Delta \tau_t)^2$

### 5. Ethical Trade-off Weights

The $\lambda$ weights embody normative choices (how much to value employment vs. climate).

**Question:** Who decides the weights?

**Answer:** This is a political decision, not a technical one. The model's role is to make trade-offs **transparent**: "To hit 1.5°C, unemployment must rise 2%; to avoid that, delay net-zero to 2070."

---

## 9. Comparison to Existing Tools

| Tool | Type | Solver | Differentiable | Multi-Objective |
|------|------|--------|----------------|-----------------|
| DEFINE (traditional) | SFC simulation | Gauss-Seidel | No | Manual judgment |
| Stella/Vensim | System dynamics | Euler/RK4 | No | Limited |
| **Differentiable DEFINE** | SFC optimization | `torch.linalg.solve` | **Yes** | **Yes** |
| CGE models | General equilibrium | Complementarity | Sometimes | Limited |
| Neural networks | Black-box | Gradient descent | Yes | Yes, but opaque |

Our approach: **Differentiable + transparent economics + automated optimization**

---

## 10. Expected Outcomes

### For DEFINE Users

1. **No more oscillations:** The model settles smoothly to equilibrium
2. **Automatic calibration:** Learn parameters from data, not manual guessing
3. **Policy discovery:** Find optimal carbon tax, green investment, etc. automatically
4. **Sensitivity reports:** Instant Jacobians showing what matters most
5. **Real-time cockpit:** Visualize trade-off frontier as new data/scenarios arrive

### For Climate-Economics

1. **Closing the loop:** Physical risk (climate) feeds back through financial system exactly
2. **Tipping point identification:** Automatic detection of systemic collapse thresholds
3. **Multi-objective governance:** Balance climate, employment, stability simultaneously
4. **Scalability:** From Canadian economy to global multi-region model

### For Differentiable Economics (Field)

1. **Proof of concept:** SFC models *can* be fully differentiable
2. **Benchmark:** Standard against which other tools can compare
3. **Methodological contribution:** Semi-ring algebra bridges discrete and continuous modeling

---

## References

**Original DEFINE Model:**
- Dafermos, Y., Nikolaidi, M., & Galanis, G. (2018). "A stock-flow-consistent macroeconomic model with environmental dimension." *Ecological Economics*, 131, 207-219.

**SFC Foundations:**
- Godley, W., & Lavoie, M. (2012). *Monetary Economics: An Integrated Approach to Credit, Money, Income, Production and Wealth*. Palgrave Macmillan.

**Climate-Economics:**
- Stern, N. (2006). *The Economics of Climate Change: The Stern Review*. UK Treasury.
- Recalibrating Climate Risk (2024). Working paper on tipping point dynamics.

**Differentiable Methods:**
- Baydin, A. G., et al. (2018). Automatic differentiation in machine learning: a survey. *JMLR*, 18.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686-707.

**Tropical Algebra:**
- Maclagan, D., & Sturmfels, B. (2015). *Introduction to tropical geometry*. AMS.
- Gondran, M., & Minoux, M. (2008). *Graphs, dioids, and semirings: New models and algorithms*. Springer.

---

## Related Work in This Repository

- **P1:** Differentiable SFC Models (methodology)
- **P2:** Unified Framework for Differentiable Economics (context)
- **X1:** SIM model optimization (simplest case)
- **X2:** Climate-damaged IO models (climate coupling)
- **X3:** Tropical supply chains (bottleneck logic)
- **notes/differentiable_green_sfc_architecture.md:** Three-layer architecture blueprint

---

**Next Steps:**

1. Implement DEFINE core in `lib/differentiable_define.py`
2. Add soft-default logic and verify no oscillations
3. Test multi-objective optimization on simplified 10-year scenario
4. Benchmark against published DEFINE results
5. Extend to full 50-year climate-policy optimization
