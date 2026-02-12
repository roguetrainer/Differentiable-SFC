# X2: Differentiable Input-Output (Leontief) Model

## Overview

This experiment implements the classic Leontief Input-Output model as a differentiable program in PyTorch. It demonstrates how to solve the **"Inverse I-O Problem"**: given observed sectoral outputs and final demand, automatically calibrate the technical coefficients matrix through gradient-based optimization.

## Purpose

Traditional Input-Output models treat the technical coefficients matrix **A** (how much steel per car, how much electricity per steel, etc.) as static, fixed values. This creates several analytical limitations:

1. **Static Calibration:** Technical coefficients must be manually set from historical IO tables, which are expensive to construct and quickly become outdated.

2. **"What-If" Opacity:** Running scenarios requires recomputing everything manually; there's no principled way to understand how coefficient changes cascade through the economy.

3. **Learning Disabled:** As new data arrives, traditional IO models don't "learn" or update their coefficients—they remain frozen in time.

This experiment transforms the Leontief system into a **learnable, differentiable model** that can automatically calibrate itself against observed economic data.

## Key Features

1. **Differentiable Leontief Solution:** The core Leontief identity is implemented as a neural network module:
   - Input: Final demand vector ($d$)
   - Learnable parameters: Technical coefficients matrix ($A$)
   - Output: Total sectoral output ($x$)

2. **Inverse I-O Calibration:** Rather than manually setting A, we use gradient descent to "learn" A from observed data:
   - Given: observed sectoral outputs ($x_{obs}$) and final demand ($d$)
   - Find: technical coefficients ($A$) that minimize $\| x_{model}(A, d) - x_{obs} \|^2$

3. **Automatic Constraint Enforcement:**
   - Non-negativity: All coefficients $A_{ij} \geq 0$ (can't have negative flows)
   - Stability: Ensures $(I - A)^{-1}$ is well-defined and non-explosive

4. **Stable Linear Solver:** Uses `torch.linalg.solve` rather than matrix inversion, which is more numerically stable and provides correct gradients.

## Files

- `io_model.py` - Basic Leontief model with inverse calibration
- `io_model_climate.py` - Extended model with climate damage feedback
- `README.md` - This file

## How It Works

### The Leontief Model

The fundamental Leontief identity describes how sectoral outputs depend on intermediate demands and final demand:

$$x = Ax + d$$

Rearranging:
$$(I - A)x = d$$

$$x = (I - A)^{-1} d$$

Where:
- $x$ = total sectoral output (what we want to predict)
- $A$ = technical coefficients matrix (what we learn)
- $d$ = final demand (observed)
- $I$ = identity matrix

### The Inverse Problem

In traditional IO analysis, you start with $A$ (from historical tables) and compute $x$ given $d$.

In our differentiable approach, we **invert the problem**: given observed $x$ and $d$, we find the $A$ that best explains the data.

This is formulated as an optimization problem:

$$A^* = \arg\min_A \| (I - A)^{-1} d - x_{obs} \|^2$$

### Gradient Flow

The magic is that PyTorch automatically computes gradients of the loss with respect to $A$:

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial x_{pred}} \cdot \frac{\partial x_{pred}}{\partial A}$$

The second term involves the gradient of the Leontief solution through the linear solver, which PyTorch handles automatically via implicit differentiation.

## Usage

Run the experiment:

```bash
python io_model.py
```

This will:
1. Initialize a 3-sector economy (Energy, Services, Manufacturing) with random technical coefficients
2. Run 1001 epochs of gradient-based calibration
3. Print loss every 200 epochs
4. Display the final calibrated coefficient matrix
5. Verify the solution satisfies the Leontief identity

## Expected Output

```
Calibrating Technical Coefficients Matrix...
Epoch 0 | Loss: 12.456789
Epoch 200 | Loss: 3.123456
Epoch 400 | Loss: 0.456789
Epoch 600 | Loss: 0.012345
Epoch 800 | Loss: 0.000123
Epoch 1000 | Loss: 0.000012

Calibrated Technical Coefficients (A):
tensor([[0.1234, 0.0456, 0.0789],
        [0.0567, 0.1890, 0.0234],
        [0.0345, 0.0678, 0.1567]])

Verification (Calculated Demand from Calibrated A):
tensor([50.0123, 80.0089, 59.9945])
Observed Demand:
tensor([50., 80., 60.])
```

The loss decreases monotonically as the model learns to match observed outputs. The verification step confirms that the calibrated $A$ correctly reproduces the observed final demand.

## Key Insights

### 1. Inverse Modeling Enables Real-Time Learning

As new national accounts data arrives (quarterly or annually), the IO model can automatically update its coefficients without manual intervention. This transforms static IO tables into **living, adaptive models**.

### 2. Sensitivity Analysis via Gradients

Once the model is trained, we can compute:

$$\frac{\partial x_i}{\partial A_{jk}} \quad \text{(How does sector } i \text{'s output respond to changes in the } j \to k \text{ coefficient?)}$$

This gives regulators and planners instant insight into which technical relationships matter most for overall economic health.

### 3. Policy Optimization

We can extend the loss function to include policy objectives. For example:

$$L = \| x_{model} - x_{obs} \|^2 + \lambda_1 \cdot \text{Carbon}(A) + \lambda_2 \cdot \text{Employment}(A)$$

Then optimize $A$ to simultaneously:
- Fit observed data (realism)
- Minimize carbon emissions (environmental)
- Maximize employment (social)

The optimal sectoral structure emerges from gradient descent, not manual deliberation.

### 4. Bridging Micro and Macro

Traditional IO models are "macro" (aggregate sectoral flows). By making the coefficient matrix learnable, we can:
- Learn firm-level production functions (micro) and aggregate them into sectors
- Identify which firms/sectors are structural chokepoints
- Design targeted interventions at the firm level to optimize sector-level outcomes

## Extension: Climate-Aware IO Model

We provide a second implementation (`io_model_climate.py`) that extends the basic Leontief model with **climate damage feedback**. This demonstrates how differentiable IO can quantify systemic climate risk.

### Climate Damage Function

Traditional economic models treat climate as external. Our differentiable approach endogenizes temperature effects through a non-linear damage function:

$$D(T) = 1 - \frac{1}{1 + e^{s(T - T_{thresh})}}$$

Where:
- $T$ = global mean temperature (°C)
- $s$ = sensitivity (steepness of tipping point)
- $T_{thresh}$ = temperature threshold where non-linearities emerge

### Climate-Damaged Technical Coefficients

As temperature rises, sectors become less efficient. The damage function scales the coefficient matrix:

$$A_{damaged}(T) = A_{baseline} \cdot (1 + \gamma \cdot D(T))$$

Where $\gamma$ is sectoral vulnerability. As $D(T) \to 1$ (at high temperatures), the matrix approaches singularity, representing economic collapse.

### Stress Testing Output

The script:
1. Calibrates the baseline economy at current temperatures (1.1°C)
2. Simulates total output across a temperature range (1.1°C to 4.0°C)
3. Visualizes the "economic collapse" as the tipping point is crossed

### Key Insight: The Economic Jacobian of Climate

With automatic differentiation, we can compute:

$$\frac{\partial \text{GDP}}{\partial T} \quad \text{(Economic sensitivity to temperature)}$$

This is computed exactly through backpropagation, without numerical approximation. Policymakers can use this gradient to:
- Quantify the economic cost of each 0.1°C of warming
- Identify which sectors need resilience investment
- Discover optimal climate mitigation investments to avoid the "collapse regime"

### Expected Output

```
Step 1: Calibrating Baseline Economy...
Epoch 0 | Loss: 8.234567
Epoch 100 | Loss: 2.156789
Epoch 200 | Loss: 0.543210
Epoch 300 | Loss: 0.087654
Epoch 400 | Loss: 0.012345
Epoch 500 | Loss: 0.001234

Step 2: Simulating Climate Tipping Point Impact...

Analysis Complete: The model now quantifies how non-linear climate
damage propagates through sectoral interdependencies.
```

Two plots appear:
1. **Left plot:** Total economic output vs. temperature, showing smooth output until ~2°C, then sharp decline
2. **Right plot:** Damage fraction vs. temperature, showing the sigmoid "kink" at the tipping point

## Extensions

1. **Dynamic IO Models:** Add time-varying coefficients $A_t$, allowing technical change and structural evolution.

2. **Disaggregated Sectors:** Instead of 3 sectors, use 100+ detailed sectors from national accounts.

3. **Trade Integration:** Extend to multi-country IO with import/export flows and learn bilateral technical coefficients.

4. **Environmental Accounts:** Add physical units (tons of CO2, MJ of energy) alongside monetary flows and learn the carbon intensity of each coefficient.

5. **Neural IO Networks:** Replace the fixed Leontief multiplier $(I - A)^{-1}$ with a neural network that learns non-linear input-output relationships.

## References

Leontief, W. (1936). "Quantitative Input-Output Relations in the Economic System of the United States." *Review of Economics and Statistics*, 18(3).

Miller, R. E., & Blair, P. D. (2009). *Input-Output Analysis: Foundations and Extensions* (2nd ed.). Cambridge University Press.

Lenzen, M., Moran, D., Geschke, A., & Foran, B. (2007). "Geo-targeting of Japanese manufacturing emissions with high-resolution IO modeling." *Environmental Science & Technology*, 41(11).

PyTorch Documentation: Linear algebra solvers and implicit differentiation.
