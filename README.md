# Differentiable Stock-Flow Consistent (SFC) Models

A research project exploring the application of automatic differentiation and differentiable programming to macroeconomic modeling. By representing economic systems as computational graphs, we enable gradient-based optimization, sensitivity analysis, and automated calibration—transforming economics from an observational science into an engineering discipline.

## Project Overview

Traditional economic models—whether Stock-Flow Consistent (SFC), Input-Output (IO), or Agent-Based (ABM)—are implemented as discrete-time simulations. This "run-and-see" approach makes them difficult to optimize, calibrate, and analyze for sensitivities.

This project advocates for a **Differentiable Programming (DP)** paradigm:
- Represent economic identities as computational graphs (PyTorch, JAX, TensorFlow)
- Leverage automatic differentiation to compute exact gradients across entire time horizons
- Use gradient-based optimization to discover stabilizing policies, climate resilience strategies, and sectoral structures
- Enable instantaneous sensitivity analysis: how does GDP respond to a 1% tax change in year 1?

### The "Holy Trinity" of Advantages

1. **Analytic Sensitivity:** Compute the full Jacobian of the economy (how every outcome responds to every parameter)
2. **Optimal Control:** Define loss functions representing economic health and use gradients to minimize them
3. **Automated Calibration:** Treat historical data as training sets and automatically learn behavioral parameters

## Repository Structure

```
Differentiable-SFC/
├── papers/                  # Research papers
│   ├── P1_differentiable_sfc/          # Bridging SD & Deep Learning
│   ├── P2_differentiable_economics_framework/  # Unified Framework
│   └── README.md           # Papers index
├── experiments/             # Proof-of-concept implementations (X1-X8)
│   ├── X1_differentiable_sim_pytorch/
│   ├── X2_differentiable_io_leontief/
│   ├── X3_tropical_supply_chain/
│   ├── X4_green_sfc_climate/
│   ├── X5_stimulus_trigger_annealing/
│   ├── X6_lowgrow_differentiable_conversion/
│   ├── X7_goodwin_volterra_control/
│   ├── X8_giraud_collapse_model/
│   └── README.md           # Experiments index
├── notebooks/               # Jupyter notebooks
│   └── GEMMES_X7_X8_differential_control.ipynb
├── notes/                   # Research notes and theoretical foundations
│   ├── Introduction_to_MGE_TTC.md
│   ├── variable_beta_thermodynamic_approach.md
│   ├── differentiable_green_sfc_architecture.md
│   ├── differentiable_define_model.md
│   └── README.md           # Notes index
├── _private/                # Private notes (symlinked to OneDrive)
└── README.md               # This file
```

## Papers

### [P1: Differentiable Stock-Flow Consistent Models](papers/P1_differentiable_sfc/)

**Title:** Bridging System Dynamics and Deep Learning: Macroeconomic Policy Design via Differentiable Stock-Flow Consistent (SFC) Models

Introduces a framework for implementing SFC models as differentiable programs. Demonstrates automatic differentiation for sensitivity analysis, gradient-based policy optimization, and automated calibration using PyTorch, JAX, and TensorFlow.

**Key Topics:**
- Computational graphs for macroeconomic systems
- Backpropagation Through Time (BPTT)
- Multi-objective loss functions for economic stability
- Proof-of-concept SIM model

### [P2: Differentiable Economics as a Unified Framework](papers/P2_differentiable_economics_framework/)

**Title:** In Praise of Differentiable Economic Models: A Unified Framework for Optimization, Sensitivity, and Control in Macro-Financial Systems

Advocates for differentiable programming across diverse economic models. Includes five case studies (SFC, supply chains, financial contagion, input-output, agent-based models) and introduces thermodynamic tensor methods for navigating non-convex policy spaces.

**Key Topics:**
- The "simulation bottleneck" and why differentiability matters
- Five case studies demonstrating differentiable approaches
- The "Holy Trinity": analytic sensitivity, optimal control, automated calibration
- Thermodynamic tensor methods and variable temperature ($\beta$) approaches
- Vision for differentiable national accounts and "policy cockpits"

## Experiments (X1-X8): Progressive Complexity

The experiments demonstrate a **progressive arc** from simple macroeconomic models to coupled climate-economic collapse dynamics, all implemented as differentiable programs.

### [X1: Differentiable SIM Model (PyTorch)](experiments/X1_differentiable_sim_pytorch/)

Classic Service-Induced Macroeconomic (SIM) model as a differentiable program. Demonstrates policy optimization using backpropagation to find tax rates that minimize volatility.

**Key Features:** Learnable behavioral parameters, multi-objective loss, policy discovery via gradients

### [X2: Differentiable Input-Output (Leontief) Models](experiments/X2_differentiable_io_leontief/)

Two implementations: inverse I-O calibration and climate-integrated Leontief with damage feedback and tipping points.

**Key Features:** Technical coefficient learning, climate damage functions, stress testing

### [X3: Tropical Supply Chain (Min-Plus Algebra)](experiments/X3_tropical_supply_chain/)

Differentiable bottleneck modeling using tropical (min-plus) semiring. Optimizes buffer allocations to minimize supply disruption under production constraints.

**Key Features:** Soft-min via LogSumExp, supply chain resilience, learnable buffers

### [X4: Green-SFC Climate Integration](experiments/X4_green_sfc_climate/)

Three-layer SFC model: physical (climate), structural (productivity), financial (debt). Climate damage reduces capital efficiency; optimizer discovers green investment policies.

**Key Features:** Climate-economic coupling, endogenous productivity, multi-objective optimization

### [X5: Stimulus Trigger with Variable β-Annealing](experiments/X5_stimulus_trigger_annealing/)

Introduces **Maslov-Gibbs Einsum (MGE)**: soft sigmoid triggers controlled by β (inverse temperature) that gradually sharpen during optimization.

**Key Features:** β-annealing schedule, differentiable policy triggers, escape from discrete IF/THEN logic

### [X6: LowGrow-SFC Differentiable Conversion](experiments/X6_lowgrow_differentiable_conversion/)

Extracts 1524 variables from real Stella model (LowGrow), converts to PyTorch, implements differentiable SFC with 158 stocks, 224 flows, and 1342 auxiliaries.

**Key Features:** Real model extraction, large-scale differentiable macroeconomic system, automated variable parsing

### [X7: Goodwin-Volterra Cycle - Optimal Control](experiments/X7_goodwin_volterra_control/)

**Pedagogical foundation:** Simplest continuous limit cycle in economics (2 variables: employment vs. wage share). Demonstrates that gradient descent discovers optimal policy damping (78.9% variance reduction).

**Key Features:** Lotka-Volterra predator-prey dynamics, policy dampening via learned parameter, phase portrait visualization

**Why it matters:** Bridge between simple oscillations and complex collapse dynamics; shows how differentiability enables "taming the cycle."

### [X8: Giraud Collapse Model - Phase Transitions](experiments/X8_giraud_collapse_model/)

**Advanced capstone:** Five-variable GEMMES model with three coupled feedback loops (climate-economic coupling, Keen-Minsky investment, debt-solvency crisis). Uses β-annealing to navigate collapse basins.

**Key Features:** Multiple feedback loops, "Obsidian Snap" phase transition, "safety corridor" discovery (optimal investment rate at edge of collapse)

**Why it matters:** Demonstrates that stable growth sits on a knife's edge; differentiability enables discovery of policies navigating toward sustainability.

### Quick Run All Experiments

```bash
# Run experiments in order (X1-X8)
for dir in experiments/X{1..8}_*/; do
  cd "$dir"
  python *.py
  cd ../..
done
```

## Quick Start

### Prerequisites

```bash
pip install torch numpy matplotlib
```

Optional (for advanced experiments):
```bash
pip install jax tensorflow pandas
```

### Run Individual Experiments

Start with X1 (simplest) and progress to X8 (most complex):

#### X1: SIM Model

```bash
cd experiments/X1_differentiable_sim_pytorch
python sim_model.py
```

#### X2: IO with Climate

```bash
cd experiments/X2_differentiable_io_leontief
python io_model_climate.py
```

#### X3: Supply Chain (Tropical Algebra)

```bash
cd experiments/X3_tropical_supply_chain
python supply_chain.py
```

#### X7: Goodwin-Volterra (Limit Cycles)

```bash
cd experiments/X7_goodwin_volterra_control
python goodwin_model.py
```

#### X8: Giraud Collapse (Phase Transitions)

```bash
cd experiments/X8_giraud_collapse_model
python giraud_model.py
```

### Run the GEMMES Notebook

Comprehensive walkthrough of X7 and X8 with MGE/TTC framework explanation:

```bash
jupyter notebook notebooks/GEMMES_X7_X8_differential_control.ipynb
```

## Key Concepts

### Automatic Differentiation (AD)

Instead of computing derivatives manually, AD tracks the gradient of every operation through a computation. For a simulation with 1000 time steps and 50 variables, AD computes the full gradient in **one backward pass**—approximately the cost of running the simulation once.

### Computational Graphs

Economic models are represented as directed acyclic graphs (DAGs):
- **Nodes** = variables (GDP, consumption, debt, etc.)
- **Edges** = dependencies (behavioral rules, accounting identities)
- **Time** = unrolled into multiple graph layers

This makes the entire system differentiable.

### Backpropagation Through Time (BPTT)

Information about future outcomes (e.g., a debt crisis in year 40) flows backward to optimize decisions in year 1. This is how we compute:

$$\frac{\partial L_T}{\partial \theta_1}$$

Where $L_T$ is a loss function at time $T$ and $\theta_1$ is a policy parameter at time 1.

### Policy Optimization via Gradients

Define a loss function $L$ representing economic "health":

$$L = \text{unemployment gap} + \text{volatility} + \text{debt sustainability}$$

Then simply:

$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla_{\theta} L$$

This discovers optimal policy automatically—no manual tuning required.

## Vision: Differentiable National Accounts

We propose that central banks and statistical agencies maintain **differentiable codebases** for national economic models. Benefits:

1. **Real-Time Policy Analysis:** See the gradient landscape of policy decisions instantly
2. **Multi-Objective Optimization:** Balance unemployment, inflation, climate emissions, and stability simultaneously
3. **Continuous Calibration:** As new data arrives, models automatically update their parameters
4. **Transparency:** Policymakers see exactly how policies couple to outcomes via explicit gradients

### The "Policy Cockpit"

Imagine a 21st-century economic control center where policymakers visualize:
- **Current state** of the economy (unemployment, inflation, debt, emissions)
- **Gradient landscape** showing which policy levers have highest impact
- **Risk surface** revealing how the economy behaves under different scenarios
- **Optimal policy path** navigating toward multiple objectives simultaneously

This is the future of evidence-based macroeconomic governance.

## Contributing

This is an active research project. To contribute:

1. Create a new experiment folder `XN_<descriptive_name>` with:
   - `code.py` or similar implementation
   - `README.md` documenting the approach and results

2. Create a new paper folder `PN_<descriptive_name>` with:
   - `draft.md` containing the full paper
   - `README.md` with abstract and overview

3. Update the corresponding folder's `README.md` to list your contribution

## Literature & References

### Foundational References

- Godley, W., & Lavoie, M. (2012). *Monetary Economics: An Integrated Approach to Credit, Money, Income, Production and Wealth*. Palgrave Macmillan.
- Leontief, W. (1936). "Quantitative Input-Output Relations in the Economic System of the United States." *Review of Economics and Statistics*, 18(3).

### Automatic Differentiation

- Griewank, A., & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*. SIAM.
- Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS 32*.

### Differentiable Programming in Science

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686-707.
- Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., & Lezcano-Casado, C. (2018). JAX: Composable transformations of Python+NumPy programs. *arXiv preprint arXiv:1811.02990*.

### Climate & Economics

- Stern, N. (2006). *The Economics of Climate Change: The Stern Review*. UK Treasury.
- Recalibrating Climate Risk. (Recent working paper on tipping point dynamics)

## License

This project is open for academic and research use. See specific paper/experiment READMEs for detailed licensing information.

## Contact

For questions or discussions about differentiable economic modeling, please open an issue or submit a pull request.

---

**Last Updated:** February 2026

**Status:** Active Development

The field of differentiable economics is emerging. We invite researchers, economists, and practitioners to contribute ideas, implementations, and critiques.
