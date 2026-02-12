# P1: Differentiable Stock-Flow Consistent Models

## Overview

This paper introduces a novel framework for implementing Stock-Flow Consistent (SFC) models as differentiable programs using modern computational engines (JAX, PyTorch, TensorFlow).

## Title

Bridging System Dynamics and Deep Learning: Macroeconomic Policy Design via Differentiable Stock-Flow Consistent (SFC) Models

## Key Contributions

1. **Differentiable SFC Implementation:** Treating macroeconomic systems as computational graphs with automatic differentiation (AD) support
2. **Instant Sensitivity Analysis:** Computing exact analytical gradients across entire time horizons
3. **Gradient-based Policy Optimization:** Discovering optimal policy parameters that minimize systemic volatility
4. **Automated Calibration:** High-performance parameter fitting against historical datasets

## Files

- `draft.md` - Full paper draft with abstract, methodology, and technical appendix

## Status

- Draft in progress
- Proof-of-concept: SIM model implementation
- Future: Multi-sector model scaling

## References

- Godley & Lavoie on SFC modeling
- Automatic differentiation in machine learning
- Backpropagation Through Time (BPTT)
