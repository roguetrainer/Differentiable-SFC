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
├── experiments/             # Proof-of-concept implementations
│   ├── X1_differentiable_sim_pytorch/  # SIM model optimization
│   ├── X2_differentiable_io_leontief/  # IO models + climate
│   └── ...
├── notes/                   # Research notes and drafts
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

## Experiments

### [X1: Differentiable SIM Model (PyTorch)](experiments/X1_differentiable_sim_pytorch/)

Implements the classic Service-Induced Macroeconomic (SIM) model as a differentiable program in PyTorch. Demonstrates policy optimization: using backpropagation to find the tax rate that minimizes economic volatility and unemployment.

**Key Features:**
- `SIM_Model` class with learnable behavioral parameters
- Multi-objective loss function (GDP targeting + volatility minimization + financial stability)
- Adam optimizer for policy discovery
- Automatic solution to the "oscillation problem"

**Usage:**
```bash
cd experiments/X1_differentiable_sim_pytorch
python sim_model.py
```

### [X2: Differentiable Input-Output (Leontief) Models](experiments/X2_differentiable_io_leontief/)

Transforms static Input-Output tables into dynamic, learnable systems. Two implementations:

**io_model.py:** Solves the "Inverse I-O Problem"—given observed sectoral outputs and final demand, learns the technical coefficients matrix through gradient descent.

**io_model_climate.py:** Extends the Leontief model with climate damage feedback. Models how non-linear temperature effects reduce sectoral efficiency, leading to economic collapse at tipping points. Computes $\frac{\partial \text{GDP}}{\partial T}$ exactly.

**Key Features:**
- Differentiable Leontief solution: $x = (I - A)^{-1} d$
- Automatic calibration of technical coefficients from data
- Climate damage function with tipping point dynamics
- Stress testing across temperature scenarios

**Usage:**
```bash
cd experiments/X2_differentiable_io_leontief
python io_model.py              # Basic IO calibration
python io_model_climate.py      # Climate damage analysis
```

## Quick Start

### Prerequisites

```bash
pip install torch numpy matplotlib
```

Optional (for advanced experiments):
```bash
pip install jax tensorflow
```

### Run the First Experiment

```bash
cd experiments/X1_differentiable_sim_pytorch
python sim_model.py
```

This will:
1. Initialize a SIM model with behavioral parameters
2. Run 200 epochs of policy optimization
3. Display the optimized tax rate and stability metrics
4. Plot GDP and wealth trajectories

### Run the IO Experiment

```bash
cd experiments/X2_differentiable_io_leontief
python io_model.py              # Calibration example
python io_model_climate.py      # Climate risk analysis
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
