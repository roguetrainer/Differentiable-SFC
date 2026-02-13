# Differentiable Green-SFC Architecture (SIM-Climate)

**Date:** 2026-02-11
**Status:** Research Note
**Related:** P2 (Differentiable Economics Framework), X1 (SIM model), X2 (Climate IO)

## Overview

This note outlines a unified differentiable architecture bridging macroeconomic SFC models with climate physics and supply chain constraints. The architecture integrates three algebraic layers (physical, financial, structural) using thermodynamic tensor methods to solve the "sustainability frontier" problem.

## Problem Statement

Traditional SFC models (e.g., LowGrow) struggle with:
1. **Oscillation problem:** Behavioral parameters create damped/undamped cycles
2. **Productivity constraints:** Supply-side bottlenecks not captured in standard SFC
3. **Climate coupling:** Temperature effects typically treated as exogenous shocks
4. **Multi-objective optimization:** Balancing employment, emissions, and stability requires manual policy tuning

**Solution:** Differentiable Green-SFC treats these as a unified computational graph with automatic gradient-based policy discovery.

## Multi-Layer Architecture

### Layer 1: Physical (Climate-Economic Coupling)

**Input:** Carbon emission stock $E_t$
**Process:**
- Cumulative emissions → atmospheric CO₂
- CO₂ → temperature increase $T_t$
- Temperature → sectoral damage fraction $D_t$

**Mathematical:**
$$T_t = f(E_t) = \text{TCRE} \times \int_0^t E_s \, ds$$
$$D_t = 1 - \frac{1}{1 + e^{s(T_t - T_{\text{thresh}})}}$$

Where:
- TCRE = Transient Climate Response to Emissions
- $s$ = sensitivity (steepness of tipping point)
- $T_{\text{thresh}}$ = temperature threshold (2°C in our baseline)

**Output:** Damage fraction $D_t$ propagates to Layer 2

### Layer 2: Financial (SFC Transaction-Flow Matrix)

**Input:** Damage fraction $D_t$ from Layer 1, Policy parameters $\theta$
**Process:**
- Standard SFC identities (income, consumption, investment)
- Climate-damaged technical coefficients: $A_{\text{damaged}}(D) = A_{\text{baseline}} \cdot (1 + \gamma D)$
- Leontief solution: $x_t = (I - A_{\text{damaged}})^{-1} d_t$

**Key Variables:**
- $Y_t$ = GDP
- $C_t$ = Consumption
- $I_t$ = Investment
- $B_t$ = Government bonds (debt)
- $E_t$ = Carbon emissions (endogenous to production)

**Output:** Economic aggregates and bond stock

### Layer 3: Structural (Supply Chain / Production Bottlenecks)

**Input:** Sectoral outputs from Layer 2, supply shocks
**Process:**
- Tropical (Min-Plus) semi-ring logic
- Bill-of-Materials (BOM) constraints
- Capacity utilization: $x_{\text{realized}} = \min(x_{\text{demand}}, x_{\text{supply}})$

**Role:**
- Prevents unrealistic output under climate stress
- Identifies critical sectoral bottlenecks
- Models infrastructure resilience/redundancy

**Output:** Final feasible sectoral outputs, constrained by topology

## The Thermodynamic Bridge (Maslov-Gibbs Einsum)

### Temperature-Dependent Coupling

The innovation is treating economic efficiency as a **thermodynamic variable** that depends on model temperature $\beta = 1/T$:

$$A_{\text{damaged}}(T, \beta) = A_{\text{baseline}} \otimes \text{SoftMax}_\beta(D(T))$$

Where:
- $T$ = Climate temperature (exogenous, from Layer 1)
- $\beta$ = Model temperature (inverse, learnable)
- $\otimes$ = Thermodynamic tensor contraction (Einsum)
- $\text{SoftMax}_\beta$ = Soft approximation with temperature dependence

### Physical Interpretation

As climate warms ($T$ rises):
- Sectoral efficiency decreases (damage $D$ increases)
- Model entropy increases (economy becomes "fuzzy"/uncertain)
- Gradient landscape flattens (multiple policies seem viable)
- Beyond tipping point: sharp phase transition (collapse)

By making $\beta$ learnable (or annealing), we can:
1. **Start hot** ($\beta$ small): Explore many feasible policy configurations
2. **Cool down** ($\beta$ large): Sharpen to the optimal policy path
3. **Detect phase transitions:** Where the system becomes unstable

## The Control Knob Objective: Sustainability Frontier

We define a multi-objective loss function capturing competing economic and environmental goals:

$$L(\theta, T) = \lambda_1 (U_t - U^*)^2 + \lambda_2 (E_t - E_{\text{target}})^2 + \lambda_3 \text{Var}(B_t) + \lambda_4 \text{Volatility}(Y_t)$$

Where:
- $(U_t - U^*)^2$ = Unemployment gap (maintain full employment)
- $(E_t - E_{\text{target}})^2$ = Emissions gap (achieve net-zero)
- $\text{Var}(B_t)$ = Debt variability (maintain stability)
- $\text{Volatility}(Y_t)$ = GDP volatility (smooth growth)
- $\lambda_i$ = Lagrange multipliers (trade-off weights)

**The policy parameters $\theta$ include:**
- Tax rates $\tau_t$
- Government spending rules $G_t(\cdot)$
- Green investment allocation $I_{\text{green}}$
- Interest rate rules $r_t(Y, \pi)$
- Carbon price/tax $p_{\text{carbon}}$

## Algorithm: Differentiable Policy Optimization

```
Input: Initial economic state, climate trajectory T(t), model architecture
Output: Optimal policy parameters θ*

1. Initialize θ (policy parameters) randomly or from prior
2. For each epoch:
   3. Forward pass (Layer 1 → Layer 2 → Layer 3):
      - Compute damage D(t) from climate trajectory
      - Solve SFC dynamics with damaged A(D)
      - Apply tropical constraints
   4. Compute loss L(θ, T) over entire time horizon
   5. Backward pass (BPTT):
      - dL/dθ via automatic differentiation
      - Gradients flow through all three layers
   6. Update θ ← θ - η ∇L
   7. Optional: Anneal β (cool system to sharpen solutions)

Return: θ* that minimizes L
```

## Expected Solution Structure

When optimized, the model discovers:

1. **Optimal Tax Path:** A tax rate trajectory that smooths consumption and stabilizes debt
2. **Green Investment Allocation:** How much to invest in emissions reduction vs. traditional capital
3. **Interest Rate Rule:** Central bank guidance that prevents both oscillation and collapse
4. **Carbon Price Trajectory:** Steadily rising price that incentivizes structural change

**Key Result:** The oscillation problem is solved automatically—the gradient descent naturally finds the "sweet spot" that damps cycles without over-suppressing activity.

## Distinction from Traditional LowGrow

| Aspect | Traditional LowGrow | Differentiable Green-SFC |
|--------|-------------------|------------------------|
| Climate coupling | Exogenous shocks | Endogenous, layer-wise |
| Oscillations | Manual damping constants | Gradient-discovered optimum |
| Policy | Manual trigger values | Learned via BPTT |
| Multi-objective | Analyst judgment | Loss function optimization |
| Bottlenecks | Not modeled | Layer 3 (tropical semi-ring) |
| Computational cost | Single simulation | One forward + one backward pass |

## Implementation Roadmap

### Phase 1: Minimal Proof-of-Concept
- [ ] Extend X1 (SIM model) to include carbon emissions
- [ ] Add Layer 1 (climate damage function)
- [ ] Optimize single policy parameter (tax rate)

### Phase 2: Full Integration
- [ ] Implement Layers 1, 2, 3 together
- [ ] Add green investment as control variable
- [ ] Implement multi-objective loss function

### Phase 3: Thermodynamic Methods
- [ ] Add learnable/annealing $\beta$ parameter
- [ ] Implement MGE tensor contraction
- [ ] Phase transition detection

### Phase 4: Empirical Calibration
- [ ] Calibrate to actual national accounts data
- [ ] Test against historical climate-economic shocks
- [ ] Validate policy recommendations

## Key Insights

1. **Unified Framework:** By treating climate, finance, and topology as layers in a computational graph, we move beyond ad-hoc coupling.

2. **Gradient Coherence:** Gradients flow from climate outcomes back to year-1 policies via BPTT. A policy that avoids tipping points in year 40 influences the optimal year-1 action.

3. **Temperature as Bridge:** The thermodynamic temperature parameter $\beta$ provides a unified way to interpolate between:
   - Hard constraints (tropical bottlenecks)
   - Soft approximations (smooth optimization)
   - Probabilistic behavior (entropy-based decisions)

4. **Automatic Discovery:** Instead of manually tuning LowGrow's "magic constants," the optimizer discovers the optimal policy path automatically.

5. **Scalability:** This architecture scales to multi-country, multi-sector models with thousands of variables—gradient-based optimization becomes feasible where manual analysis is impossible.

## Open Questions

1. **Uniqueness:** Are there multiple local minima in the policy space? How sensitive are solutions to initialization?

2. **Interpretability:** When the optimizer discovers novel policy rules, how do we explain them to policymakers?

3. **Robustness:** How stable are solutions when climate parameters change (e.g., new TCRE estimates)?

4. **Tractability:** Can we solve this in real-time (seconds/minutes) or only offline (hours/days)?

5. **Ethical:** If the model discovers that some carbon cuts require unemployment, who decides the trade-off weights $\lambda_i$?

## References

- Godley & Lavoie (2012): SFC modeling foundations
- Stern (2006): Climate economics
- Raissi et al. (2019): Physics-informed neural networks (differentiable physics)
- Baydin et al. (2018): Automatic differentiation survey
- Recalibrating Climate Risk (2024): Tipping point dynamics
- Paszke et al. (2019): PyTorch framework

## Related Experiments/Papers

- **X1:** Differentiable SIM (baseline optimization)
- **X2:** Climate-damaged IO (layer-wise structure)
- **X3:** Tropical supply chains (bottleneck logic)
- **P1:** Differentiable SFC models (methodology)
- **P2:** Unified framework across models (context)

---

**Next Steps:**
1. Implement Layer 1 (climate coupling) as extension to X1
2. Test on simplified 50-year horizon
3. Benchmark against manual LowGrow calibration
4. Expand to multi-sector, multi-country version
