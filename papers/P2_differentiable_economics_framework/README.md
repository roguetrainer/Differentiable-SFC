# P2: Differentiable Economics as a Unified Framework

## Overview

This paper advocates for a paradigm shift in economic modeling: from discrete-time simulations to **differentiable programming**. By representing economic identities as computational graphs, researchers can leverage automatic differentiation to optimize policy, calibrate models, and perform sensitivity analysis—transforming economics from an observational science to an engineering discipline.

## Title

In Praise of Differentiable Economic Models: A Unified Framework for Optimization, Sensitivity, and Control in Macro-Financial Systems

## Key Contributions

1. **Unified Differentiable Framework:** Demonstrates how to transform diverse economic models (SFC, IO, ABM, CGE) into differentiable programs

2. **Five Case Studies:**
   - Stock-Flow Consistent (SFC) models: Automatic discovery of stabilizing policies
   - Supply Chain Networks: Instant bottleneck identification via gradients
   - Financial Contagion (DebtRank): Optimal capital buffer design
   - Input-Output Models: Learning technical coefficients from trade data
   - Agent-Based Models: Training agent populations toward social objectives

3. **The "Holy Trinity" of Advantages:**
   - Analytic Sensitivity (Jacobian of the economy)
   - Optimal Control via gradient descent
   - Automated Calibration against historical data

4. **Thermodynamic Extensions:** Introduction of the Maslov-Gibbs Einsum (MGE) framework with variable temperature ($\beta$) for navigating non-convex policy spaces

5. **Policy Infrastructure Vision:** A call for central banks to maintain differentiable codebases and "Policy Cockpits" for real-time economic governance

## Files

- `draft.md` - Complete paper with all sections, case studies, and technical discussion
- `README.md` - This file

## Structure

1. **Introduction:** The "run-and-see" bottleneck and thesis for differentiability
2. **Differentiable Paradigm:** AD vs. numerical derivatives, computational graphs, BPTT
3. **Five Case Studies:** SFC, supply chains, DebtRank, IO, ABMs
4. **Holy Trinity:** Analytic sensitivity, optimal control, automated calibration
5. **Obstacles & Critiques:** Non-convexity, discrete events, Lucas critique
6. **Future Horizon:** Thermodynamic tensor methods and variable temperature
7. **Conclusion:** Call for differentiable national accounts and policy cockpits

## Key Insight

Traditional economic models are "closed" systems—you run them and observe the output. Differentiable economic models are "open" systems—you define objectives, compute gradients toward them, and let optimization discover the policies that work.

This is the conceptual leap from 20th-century econometrics to 21st-century economic engineering.

## Status

- Draft in progress
- Integrates concepts from X1 experiment and extends to broader modeling paradigms
- Includes forward-looking discussion of thermodynamic approaches

## Target Audience

- Macroeconomists and policy practitioners
- Complexity scientists and systems modelers
- Machine learning researchers interested in applications to economics
- Central banks and statistical agencies exploring computational methods

## Related Work

- **Closely related:** Paper P1 (Differentiable Stock-Flow Consistent Models)
- **Experimental basis:** X1 experiment (Differentiable SIM model in PyTorch)
- **Future:** Additional experiments implementing case studies 2-5

## Next Steps

1. Implement differentiable supply chain models (X2)
2. Develop relaxed DebtRank framework (X3)
3. Build inverse IO learning system (X4)
4. Create agent-population trainer (X5)
5. Explore MGE/thermodynamic methods (X6+)
