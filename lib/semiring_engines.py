"""
Differentiable Economics Engine: Semi-ring Algebraic Foundations

This module implements differentiable economic models using two algebraic structures:

1. StandardSemiringIO: Standard arithmetic (+, *) for accounting identities
   - Application: SFC models, Input-Output analysis, monetary flows
   - Logic: Output is the SUM of intermediate and final demands
   - Differentiability: Straightforward (linear/polynomial)

2. TropicalSemiringSupplyChain: Tropical (Min-Plus) algebra (min, +) for bottlenecks
   - Application: Supply chains, DebtRank, critical path analysis
   - Logic: Output is the MINIMUM of component availability
   - Differentiability: Via Log-Sum-Exp softening (temperature-dependent)

The Temperature Parameter (β):
- β = 1/T (inverse temperature)
- High β (T→0): Pure Tropical logic (hard min bottlenecks)
- Low β (T→∞): Smooth Standard-like approximation
- Variable β: Enables simulated annealing through algebraic phases

This bridges deterministic and stochastic economics through a unified framework.
"""

import torch
import torch.nn as nn


# ============================================================================
# STANDARD SEMI-RING: (+, *) - THE "VOLUME" LOGIC
# ============================================================================

class StandardSemiringIO(nn.Module):
    """
    Differentiable I-O Model using Standard Semi-ring Arithmetic.

    The standard semi-ring describes the "What": how much is produced.
    Uses additive composition (geometric series expansion of Leontief multiplier).

    Args:
        num_sectors (int): Number of economic sectors
    """

    def __init__(self, num_sectors):
        super(StandardSemiringIO, self).__init__()
        self.num_sectors = num_sectors

        # Technical coefficients matrix (learnable)
        # Represents baseline efficiency before climate/structural damage
        self.A_raw = nn.Parameter(torch.randn(num_sectors, num_sectors) * 0.01)

        # Climate damage parameters (from "Recalibrating Climate Risk" framework)
        # Sensitivity: steepness of tipping point sigmoid
        self.tp_sensitivity = nn.Parameter(torch.tensor(2.5))
        # Threshold: temperature (°C) where non-linearities accelerate
        self.tp_threshold = nn.Parameter(torch.tensor(2.0))
        # Gamma: sectoral vulnerability multiplier
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def get_damage_fraction(self, temperature):
        """
        Tipping Point Damage Function: D(T) = 1 - 1 / (1 + exp(s(T - T_thresh)))

        Models non-linear economic collapse as temperature rises.
        Represents phase transition in sectoral efficiency.

        Args:
            temperature (torch.Tensor): Global mean temperature (°C)

        Returns:
            torch.Tensor: Damage fraction in [0, 1]
        """
        return 1.0 - (1.0 / (1.0 + torch.exp(self.tp_sensitivity * (temperature - self.tp_threshold))))

    def get_A(self, temperature=None):
        """
        Get technical coefficients matrix, optionally damaged by climate effects.

        A_baseline ∈ [0, 0.5] ensures stability (eigenvalues < 1).
        A_damaged = A_baseline * (1 + γ * D(T)) inflates coefficients with damage.

        As matrix approaches singularity (row/col sums → 1), economy collapses.

        Args:
            temperature (torch.Tensor, optional): Temperature for climate damage

        Returns:
            torch.Tensor: Technical coefficients matrix [num_sectors × num_sectors]
        """
        A_baseline = torch.sigmoid(self.A_raw) * 0.5

        if temperature is not None:
            damage = self.get_damage_fraction(temperature)
            # Damage acts as efficiency multiplier: more inputs needed per output
            A_damaged = A_baseline * (1.0 + self.gamma * damage)
            return A_damaged

        return A_baseline

    def forward(self, final_demand, temperature=None):
        """
        Leontief Solution: x = (I - A)^-1 * d

        Standard semi-ring composition uses matrix inversion (geometric series).
        Gradients flow through the linear solver (torch.linalg.solve).

        Args:
            final_demand (torch.Tensor): Final demand vector [num_sectors]
            temperature (torch.Tensor, optional): Climate forcing

        Returns:
            torch.Tensor: Total sectoral output [num_sectors]
        """
        A = self.get_A(temperature)
        I = torch.eye(self.num_sectors, device=A.device)

        # torch.linalg.solve is differentiable and numerically stable
        # Solves (I - A) x = d more stably than torch.inverse
        total_output = torch.linalg.solve(I - A, final_demand)
        return total_output


# ============================================================================
# TROPICAL SEMI-RING: (min, +) - THE "TOPOLOGY" LOGIC
# ============================================================================

class TropicalSemiringSupplyChain(nn.Module):
    """
    Differentiable Supply Chain Model using Tropical (Min-Plus) Algebra.

    The tropical semi-ring describes the "Can": structural limits on production.
    Uses bottleneck composition (min-plus algebra from network theory).

    The tropical min operator is traditionally non-differentiable (kinks),
    but we use Log-Sum-Exp to create a smooth approximation with temperature parameter.

    Args:
        adjacency_matrix (torch.Tensor): Bill-of-Materials (BOM) matrix [n_nodes × n_nodes]
                                        non-zero A[i,j] means node i requires node j
        beta (float): Inverse temperature (1/T) for controlling hardness of min
                     High beta (T→0): Pure tropical min (hard bottleneck)
                     Low beta (T→∞): Soft approximation (more forgiving)
    """

    def __init__(self, adjacency_matrix, beta=10.0):
        super(TropicalSemiringSupplyChain, self).__init__()
        self.num_nodes = adjacency_matrix.shape[0]
        self.register_buffer('BOM', adjacency_matrix)
        self.beta = beta

        # Learnable buffer parameters (resilience knobs)
        # Controls inventory/redundancy at each node to mitigate bottlenecks
        self.buffers = nn.Parameter(torch.zeros(1, self.num_nodes))

    def set_temperature(self, T):
        """
        Dynamically adjust temperature for annealing.

        As system cools (T→0, β→∞), tropical bottleneck logic hardens.
        This enables phase transitions and identification of critical paths.

        Args:
            T (float): Temperature. β = 1/T
        """
        self.beta = 1.0 / max(T, 1e-6)

    def soft_min(self, x, dim=-1):
        """
        Differentiable "Dequantization" of the Tropical Min Operator.

        Uses Log-Sum-Exp (LSE) trick to compute smooth approximation to min:
        soft_min(x) ≈ -log(sum(exp(-β*x))) / β

        As β→∞, this converges to the true min function.
        Gradients flow through via automatic differentiation.

        Args:
            x (torch.Tensor): Input tensor
            dim (int): Dimension along which to take soft min

        Returns:
            torch.Tensor: Soft min values
        """
        return -(1.0 / self.beta) * torch.logsumexp(-self.beta * x, dim=dim)

    def forward(self, exogenous_shocks):
        """
        Tropical supply chain propagation: Output = min of all required components.

        Process:
        1. Each node has initial capacity (shock + buffer)
        2. Iterate through supply chain depth
        3. At each iteration, output constrained by BOM dependencies
        4. Return final node outputs (representing delivered goods)

        The "Tropical Collapse": If any upstream component fails (shock < threshold),
        the min operator propagates the failure downstream, halting all dependent production.

        Args:
            exogenous_shocks (torch.Tensor): Supply shocks at each node [1 × num_nodes]

        Returns:
            torch.Tensor: Final node outputs [1 × num_nodes]
        """
        # Initial capacity: shocks limited by buffers
        # Buffers act as inventory/redundancy to smooth shocks
        effective_capacity = torch.clamp(
            exogenous_shocks + torch.sigmoid(self.buffers),
            0.0, 1.0
        )

        node_outputs = effective_capacity.clone()

        # Iterate through supply chain depth (3 iterations for typical networks)
        for _ in range(3):
            # Build input requirement matrix
            # inputs[i,j] = output[j] * BOM[i,j] (node i's requirement from node j)
            inputs = node_outputs.unsqueeze(0) * self.BOM.unsqueeze(1)

            # Mask non-dependencies: where BOM[i,j]=0, set to "tropical infinity" (1.0)
            # This means: if there's no dependency, it doesn't constrain output
            inputs = torch.where(
                self.BOM.unsqueeze(1) > 0,
                inputs,
                torch.tensor(1.0, device=inputs.device)
            )

            # Apply tropical min (softened)
            # For each node, output is limited by the minimum of its input requirements
            bottleneck = self.soft_min(inputs, dim=-1)

            # Update outputs: constrained by bottleneck but capped at initial capacity
            node_outputs = effective_capacity * bottleneck

        return node_outputs


# ============================================================================
# MULTI-SEMIRING ORCHESTRATOR
# ============================================================================

class HybridEconomicModel(nn.Module):
    """
    Combines Standard and Tropical semi-rings to model both volume and topology.

    Standard SFC/IO component: "How much is produced?" (additive logic)
    Tropical supply component: "Can it be produced?" (bottleneck logic)

    The hybrid model first computes standard (unconstrained) demand,
    then constrains it through tropical supply chain bottlenecks.

    Args:
        num_sectors (int): Number of sectors
        supply_chain_bom (torch.Tensor): Bill-of-materials for supply bottlenecks
    """

    def __init__(self, num_sectors, supply_chain_bom=None, beta=10.0):
        super(HybridEconomicModel, self).__init__()
        self.standard_io = StandardSemiringIO(num_sectors)

        if supply_chain_bom is not None:
            self.tropical_chain = TropicalSemiringSupplyChain(supply_chain_bom, beta=beta)
        else:
            self.tropical_chain = None

    def forward(self, final_demand, temperature=None, supply_shocks=None):
        """
        Hybrid computation: Standard output constrained by Tropical bottlenecks.

        Args:
            final_demand (torch.Tensor): Final demand [num_sectors]
            temperature (torch.Tensor, optional): Climate damage parameter
            supply_shocks (torch.Tensor, optional): Supply chain shocks

        Returns:
            torch.Tensor: Final economic output after all constraints
        """
        # Step 1: Standard semi-ring (unconstrained demand satisfaction)
        output_demand = self.standard_io(final_demand, temperature=temperature)

        # Step 2: Tropical semi-ring (supply chain constraints)
        if self.tropical_chain is not None and supply_shocks is not None:
            # Normalize output to [0,1] for compatibility with supply chain
            output_normalized = output_demand / (output_demand.max() + 1e-6)
            output_constrained = self.tropical_chain(output_normalized.unsqueeze(0))
            # Scale back to original magnitude
            return output_constrained.squeeze(0) * output_demand.max()

        return output_demand


# ============================================================================
# UTILITY FUNCTIONS FOR GRADIENT ANALYSIS
# ============================================================================

def compute_jacobian(model, input_tensor, param_name=None):
    """
    Compute full Jacobian matrix: how outputs respond to inputs/parameters.

    J[i,j] = ∂output[i] / ∂input[j]

    Args:
        model (nn.Module): Differentiable model
        input_tensor (torch.Tensor): Input to model
        param_name (str, optional): Specific parameter to differentiate w.r.t.

    Returns:
        torch.Tensor: Jacobian matrix
    """
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    output_size = output.numel()
    input_size = input_tensor.numel()

    jacobian = torch.zeros(output_size, input_size)

    for i in range(output_size):
        model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()

        output[i].backward(retain_graph=True)
        jacobian[i] = input_tensor.grad.view(-1).detach().clone()

    input_tensor.requires_grad_(False)
    return jacobian


def analyze_sensitivity(model, base_input, parameter_name, perturbation=0.01):
    """
    Sensitivity analysis: How much does output change for 1% parameter change?

    Useful for policy evaluation: "What's the economic cost of a 1% tax increase?"

    Args:
        model (nn.Module): Differentiable model
        base_input (torch.Tensor): Base input vector
        parameter_name (str): Name of parameter to perturb
        perturbation (float): Fractional perturbation (default 1%)

    Returns:
        dict: Sensitivity metrics
    """
    with torch.no_grad():
        base_output = model(base_input)

    for name, param in model.named_parameters():
        if parameter_name in name:
            original_value = param.data.clone()

            # Perturb parameter
            param.data = original_value * (1.0 + perturbation)
            with torch.no_grad():
                perturbed_output = model(base_input)

            # Restore
            param.data = original_value

            # Compute elasticity: % change in output / % change in parameter
            elasticity = (perturbed_output - base_output) / base_output / perturbation

            return {
                "parameter": parameter_name,
                "base_output": base_output.detach().numpy(),
                "perturbed_output": perturbed_output.detach().numpy(),
                "elasticity": elasticity.detach().numpy(),
            }

    return None
