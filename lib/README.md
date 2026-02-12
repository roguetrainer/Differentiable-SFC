# Differentiable Economics Library

Core computational engines for differentiable economic modeling using semi-ring algebra.

## Overview

This library provides foundational building blocks for representing economic systems as differentiable computational graphs. Rather than implementing economics as discrete simulations, we use algebraic structures that naturally capture different economic logics.

## Two Algebraic Worlds

### Standard Semi-ring: (+, *) — The "Volume" Logic

**Mathematical Structure:** Standard real numbers with addition and multiplication

**Economics:** How much is produced? Additive composition of sectoral outputs.

**Applications:**
- Stock-Flow Consistent (SFC) models
- Input-Output (Leontief) analysis
- National accounts (GDP accounting)
- Monetary flows and financing

**Key Properties:**
- Identity: 0 (additive), 1 (multiplicative)
- Natural composition: If you have two factories, you sum outputs
- Differentiability: Straightforward via polynomial gradients
- Singularity: Matrix (I-A) becomes singular when sectoral efficiency exceeds 100%

**Implementation:** `StandardSemiringIO`

### Tropical Semi-ring: (min, +) — The "Topology" Logic

**Mathematical Structure:** Real numbers with min (tropical addition) and + (tropical multiplication)

**Economics:** What can be produced? Bottleneck constraints from network topology.

**Applications:**
- Supply chain networks and critical paths
- Financial contagion (DebtRank)
- Infrastructure resilience
- Systemic risk identification

**Key Properties:**
- Identity: +∞ (tropical addition), 0 (tropical multiplication)
- Natural composition: Output limited by minimum component availability
- Differentiability: Traditionally non-differentiable (kinks)
- Solution: Log-Sum-Exp "softening" with temperature parameter

**Implementation:** `TropicalSemiringSupplyChain`

## The Temperature Bridge

The crucial innovation is the **temperature parameter** (β = 1/T) that connects these two algebraic worlds:

- **High β (T→0):** Pure tropical logic
  - Hard bottleneck constraints (min operator)
  - Identifies critical failure points
  - Zero entropy in decision-making

- **Low β (T→∞):** Smooth approximation
  - Fuzzy logic (soft-min via Log-Sum-Exp)
  - Gradients flow easily through bottlenecks
  - Maximum entropy/flexibility

- **Variable β:** Simulated Annealing
  - Start with high T (fuzzy exploration)
  - Gradually cool (β→∞) to identify precise failure points
  - Detect phase transitions in the economic system

## Core Classes

### `StandardSemiringIO(num_sectors)`

Differentiable Input-Output model with climate damage feedback.

**Key Methods:**
- `get_A(temperature)` — Technical coefficients with climate damage
- `get_damage_fraction(temperature)` — Tipping point sigmoid
- `forward(final_demand, temperature)` — Leontief solution

**Key Insight:** As temperature rises, technical coefficients inflate (efficiency declines), eventually causing matrix singularity (economic collapse).

### `TropicalSemiringSupplyChain(adjacency_matrix, beta)`

Differentiable supply chain with bottleneck logic.

**Key Methods:**
- `soft_min(x, dim)` — Log-Sum-Exp softened tropical min
- `set_temperature(T)` — Dynamic cooling for annealing
- `forward(exogenous_shocks)` — Propagate shocks through network

**Key Insight:** Failures propagate downstream via tropical min operator; buffers/redundancy can absorb shocks.

### `HybridEconomicModel(num_sectors, supply_chain_bom)`

Combines standard (volume) and tropical (topology) logics.

**Process:**
1. Standard semiring: compute unconstrained demand satisfaction
2. Tropical semiring: constrain by supply chain bottlenecks
3. Result: realistic output accounting for both market forces and physical limits

## Utility Functions

### `compute_jacobian(model, input_tensor)`

Compute full sensitivity matrix: how outputs respond to all inputs.

Returns the "Economic Jacobian" — the directional derivatives of the economy with respect to all parameters.

**Use Case:** "How does GDP respond to changes in each technical coefficient?"

### `analyze_sensitivity(model, base_input, parameter_name, perturbation)`

Elasticity analysis: percentage change in output per percentage change in parameter.

**Use Case:** "What's the economic cost of a 1% tax increase?"

## Design Philosophy

1. **Algebraic Clarity:** Models are organized by their underlying mathematical structure, not by economic sector or time horizon.

2. **Differentiability First:** Every component is designed to support automatic differentiation and gradient flow.

3. **Temperature Universality:** The β parameter provides a unified way to interpolate between deterministic and stochastic behavior.

4. **Composability:** Standard and tropical models can be combined (via `HybridEconomicModel`) for richer dynamics.

5. **Scalability:** Code uses PyTorch's efficient tensor operations and GPU support.

## Usage Example

```python
import torch
from lib.semiring_engines import StandardSemiringIO, analyze_sensitivity

# Create a 5-sector economy
model = StandardSemiringIO(num_sectors=5)

# Define final demand
demand = torch.tensor([100.0, 150.0, 120.0, 90.0, 110.0])

# Compute output at current temperature (1.1°C)
output = model(demand, temperature=torch.tensor(1.1))

# Sensitivity analysis: economic cost of sectoral efficiency loss
sensitivity = analyze_sensitivity(model, demand, "tp_sensitivity", perturbation=0.01)
print(f"1% increase in tipping point sensitivity reduces output by {sensitivity['elasticity']}%")

# Stress test: how much warming until collapse?
temps = torch.linspace(1.1, 4.0, 30)
outputs = [model(demand, temperature=t).sum().item() for t in temps]
```

## Extending the Library

To add new semi-ring models:

1. Inherit from `nn.Module`
2. Implement `forward()` with differentiable operations
3. Define the algebraic identity and composition operator
4. Document which economic logics the model captures

## References

**Semi-ring Algebra:**
- Droste, M., & Kuich, W. (2009). "Semirings and formal power series." In *Handbook of weighted automata*.
- Gondran, M., & Minoux, M. (2008). *Graphs, dioids, and semirings: New models and algorithms*. Springer.

**Tropical Geometry in Economics:**
- Maclagan, D., & Sturmfels, B. (2015). *Introduction to tropical geometry*. American Mathematical Society.
- Recent applications to supply chain optimization and systemic risk.

**Differentiable Programming:**
- Paszke, A., et al. (2019). PyTorch: An Imperative Style High-Performance Deep Learning Library.
- Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning.

**Economic Applications:**
- Godley, W., & Lavoie, M. (2012). *Monetary Economics: An Integrated Approach*. Palgrave Macmillan.
- Leontief, W. (1936). "Quantitative Input-Output Relations in the Economic System of the United States."
- Recalibrating Climate Risk. (2024). Working paper on tipping point dynamics.
