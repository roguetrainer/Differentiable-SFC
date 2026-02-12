# Papers

This folder contains drafts and notes for research papers related to the Differentiable SFC project.

## Papers

### P1: Differentiable Stock-Flow Consistent Models

**Title:** Bridging System Dynamics and Deep Learning: Macroeconomic Policy Design via Differentiable Stock-Flow Consistent (SFC) Models

**Abstract:** Introduces a framework for implementing SFC models as differentiable programs using JAX, PyTorch, and TensorFlow. Enables automatic differentiation for sensitivity analysis, gradient-based policy optimization, and automated calibration.

**Key Topics:**
- Computational graphs for macroeconomic systems
- Automatic differentiation (AD) and Backpropagation Through Time
- Multi-objective loss functions for economic stability
- Proof-of-concept SIM model implementation

**Location:** `P1_differentiable_sfc/`

---

### P2: Differentiable Economics as a Unified Framework

**Title:** In Praise of Differentiable Economic Models: A Unified Framework for Optimization, Sensitivity, and Control in Macro-Financial Systems

**Abstract:** Advocates for a paradigm shift from discrete-time simulations to differentiable programming across diverse economic models. Demonstrates how automatic differentiation enables policy optimization, sensitivity analysis, and automated calibration. Includes five case studies (SFC, supply chains, financial contagion, input-output, agent-based models) and introduces thermodynamic tensor methods for navigating non-convex policy spaces.

**Key Topics:**
- The simulation bottleneck and limits of "run-and-see" analysis
- Automatic differentiation vs. numerical sensitivity
- Five case studies: SFC, supply chains, DebtRank, IO models, ABMs
- The "Holy Trinity": analytic sensitivity, optimal control, automated calibration
- Thermodynamic tensor methods and variable temperature ($\beta$) approaches
- Vision for differentiable national accounts and policy cockpits

**Location:** `P2_differentiable_economics_framework/`

---

## Adding New Papers

When adding a new paper:
1. Create a folder named `PX_<short_name>` where X is the paper number
2. Include:
   - `draft.md` - Full paper draft
   - `README.md` - Overview and key information
   - Additional notes or code files as needed
3. Update this README with the new paper information
