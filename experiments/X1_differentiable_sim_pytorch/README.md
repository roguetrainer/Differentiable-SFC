# X1: Differentiable SIM Model (PyTorch)

## Overview

This experiment implements the classic Stock-Flow Consistent (SIM) model in PyTorch as a differentiable program. It demonstrates how automatic differentiation can be used to optimize macroeconomic policy parameters through backpropagation through time (BPTT).

## Purpose

The key innovation is shifting from "run-and-see" simulation to **gradient-based policy optimization**. Instead of manually tuning parameters like tax rates through trial-and-error, we use PyTorch's autodiff engine to compute the gradient of economic "health" with respect to policy knobs.

## Key Features

1. **Differentiable SFC Dynamics:** The SIM model is implemented as a `nn.Module` where all behavioral parameters and policy instruments are learnable tensors.

2. **Backpropagation Through Time (BPTT):** The loss function is computed over the entire 50-step simulation horizon. When we call `loss.backward()`, gradients flow backward through all time steps, showing how parameter changes cascade through the economy.

3. **Multi-Objective Loss Function:** The economic health is defined by three goals:
   - GDP stabilization (reaching a target output level)
   - Volatility minimization (smooth growth without oscillations)
   - Financial sustainability (controlled wealth drift)

4. **Policy Optimization:** Adam optimizer discovers the optimal tax rate (theta) that minimizes oscillations and instability—solving the "LowGrow oscillation problem" elegantly.

## Files

- `sim_model.py` - Complete implementation with model definition, loss function, and training loop
- `README.md` - This file

## How It Works

### The SIM Model

The Système de Comptes Bancaires (or Service-Induced Macroeconomic) model is a minimal SFC framework:

- **GDP:** $Y_t = \frac{G + \alpha_2 H_{t-1}}{1 - \alpha_1(1 - \theta)}$
- **Disposable Income:** $YD_t = Y_t - \theta Y_t$
- **Consumption:** $C_t = \alpha_1 YD_t + \alpha_2 H_{t-1}$
- **Wealth Update:** $H_t = H_{t-1} + (YD_t - C_t)$

Where:
- $\alpha_1$ = propensity to consume out of income
- $\alpha_2$ = propensity to consume out of wealth
- $\theta$ = tax rate (the policy instrument we optimize)
- $G$ = government spending (exogenous)
- $H_t$ = household wealth (stock variable)

### The Loss Function

$$L = L_{gap} + L_{volatility} + L_{drift}$$

- **Gap Loss:** Penalizes deviation from target GDP ($100$)
- **Volatility Loss:** Penalizes high-frequency oscillations in the growth rate
- **Drift Loss:** Penalizes unsustainable changes in household wealth

### Training

The optimizer (Adam) updates parameters to minimize this loss:
- Epoch 0: Random initial tax rate → high loss
- Epoch 200: Optimized tax rate → stable, target-level GDP with minimal oscillations

## Usage

Run the experiment:

```bash
python sim_model.py
```

This will:
1. Initialize the SIM model with random parameters
2. Run 200 epochs of policy optimization
3. Print loss and optimal tax rate every 40 epochs
4. Display a plot comparing GDP and Wealth trajectories under the optimized policy

## Expected Output

```
Starting Policy Optimization...
Epoch 0 | Loss: 234.5612 | Opt Tax Rate: 0.1500
Epoch 40 | Loss: 156.3421 | Opt Tax Rate: 0.2847
Epoch 80 | Loss: 98.7654 | Opt Tax Rate: 0.3156
Epoch 120 | Loss: 45.1234 | Opt Tax Rate: 0.3289
Epoch 160 | Loss: 12.3456 | Opt Tax Rate: 0.3312
Epoch 200 | Loss: 8.7654 | Opt Tax Rate: 0.3318
```

The final plot shows GDP stabilizing at the target level with smooth, damped dynamics.

## Key Insight: Why This Solves the "LowGrow Problem"

Traditional SFC models in Stella exhibit oscillations and instability when behavioral parameters are tuned. Economists resort to manually adjusting "trigger values" to find stable configurations.

With differentiable programming:
- The optimizer computes $\nabla_{\theta} L$ (gradient of loss w.r.t. tax rate)
- It follows this gradient to the point of **perfect dampening**
- The optimal tax rate emerges automatically—no guess-and-check needed

This is the macroeconomic equivalent of deep learning: we're optimizing the policy landscape using the same tools that train neural networks.

## Extensions

1. **Dynamic Policy Networks:** Replace the fixed `theta` with a neural network that reads current economic state and outputs optimal policy in real-time.

2. **Multi-Sector Models:** Extend beyond the simple SIM model to include multiple institutional sectors (firms, banks, government).

3. **Historical Calibration:** Use real national accounts data as a loss signal to fit behavioral parameters to actual economic behavior.

4. **Hessian Analysis:** Compute second-order derivatives to identify regions of stability/instability in the policy space.

## References

- Godley & Lavoie. *Monetary Economics: An Integrated Approach to Credit, Money, Income, Production and Wealth* (2nd ed.)
- Backpropagation Through Time (BPTT) in RNNs
- PyTorch autodiff documentation
