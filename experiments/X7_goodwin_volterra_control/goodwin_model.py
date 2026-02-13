"""
X7: Differentiable Goodwin-Lotka-Volterra Model with Optimal Control

Economic Interpretation:
- Prey (x): Employment Rate. It wants to grow, but is 'eaten' by high wages.
- Predator (y): Wage Share. It grows when employment is high, but 'starves' output.

Goal: Use a 'Policy Control' (e.g., Taxes or Incomes Policy) to stabilize the cycle.

This is the simplest example of a Limit Cycle in economics. Unlike discrete artifacts
(chattering in Stella), a Limit Cycle is a true continuous oscillation governed by
Lotka-Volterra equations—the mathematical heart of the Goodwin cycle in GEMMES.

By applying gradient descent to find the optimal policy intervention, we demonstrate
that differentiability enables us to "break" the boom-bust cycle without changing
the underlying economic structure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class GoodwinVolterra(nn.Module):
    """
    Differentiable Goodwin-Lotka-Volterra Model.

    The model implements the predator-prey dynamic:
    - Employment Rate (x): Prey. Grows naturally, but is suppressed by wages.
    - Wage Share (y): Predator. Grows when employment is high, but suppresses output.

    Without intervention, the system exhibits a perpetual limit cycle.
    With optimal control, we can dampen oscillations and stabilize at equilibrium.

    Parameters:
    -----------
    alpha : float
        Growth rate of employment (prey natural growth)
    beta : float
        Impact of wages on employment (predation rate)
    delta : float
        Impact of employment on wages (predator reproduction rate)
    gamma : float
        Decay rate of wages (predator mortality)
    """

    def __init__(self, alpha=0.1, beta=0.5, delta=0.5, gamma=0.2):
        super(GoodwinVolterra, self).__init__()

        # --- STRUCTURAL PARAMETERS ---
        # These create the inherent oscillation
        self.alpha = alpha   # Growth rate of Employment (Prey)
        self.beta = beta     # Impact of Wages on Employment (Predation rate)
        self.delta = delta   # Impact of Employment on Wages (Reproduction rate)
        self.gamma = gamma   # Decay rate of Wages (Mortality)

        # --- THE CONTROL KNOB ---
        # A learnable parameter representing a "Stabilization Policy"
        # e.g., Counter-cyclical taxation, investment buffering, or incomes policy
        # The optimizer adjusts this to dampen the cycle
        # Initialize to small random value to break symmetry and enable gradient flow
        self.policy_strength = nn.Parameter(torch.tensor(0.5))

    def forward(self, steps=200, dt=0.1):
        """
        Simulate the Goodwin-Volterra dynamics with policy intervention.

        Parameters:
        -----------
        steps : int
            Number of time steps to simulate
        dt : float
            Integration step size (Euler method)

        Returns:
        --------
        x : torch.Tensor of shape (steps,)
            Employment rate trajectory
        y : torch.Tensor of shape (steps,)
            Wage share trajectory
        """
        # Initial State: Perturbed far from equilibrium to generate strong oscillations
        # Parameters chosen to generate a clear limit cycle
        x = torch.tensor(0.8)  # High employment rate (80%)
        y = torch.tensor(0.3)  # Low wage share (30%)

        history = {'x': [], 'y': []}

        for _ in range(steps):
            # --- POLICY INTERVENTION ---
            # The control applies a damping force proportional to the interaction term
            # This represents counter-cyclical policy: tax during booms, spend during busts
            # Intervention = -k * (x * y), where k = |policy_strength|
            # The negative sign means: when both employment and wages are high, apply brake
            intervention = -torch.abs(self.policy_strength) * (x * y)

            # --- LOTKA-VOLTERRA DYNAMICS ---
            # dx/dt = alpha*x - beta*x*y + intervention
            # Interpretation: Employment grows naturally, suppressed by wage share
            dx = (self.alpha * x) - (self.beta * x * y) + intervention

            # dy/dt = delta*x*y - gamma*y - intervention
            # Interpretation: Wages grow with employment, decay naturally
            dy = (self.delta * x * y) - (self.gamma * y) - intervention

            # --- EULER INTEGRATION ---
            x = x + dx * dt
            y = y + dy * dt

            # --- CONSTRAINTS ---
            # For visualization, clamp to [0, 1], but preserve gradient-bearing originals
            x_clamped = torch.clamp(x, 0.0, 1.0)
            y_clamped = torch.clamp(y, 0.0, 1.0)

            history['x'].append(x_clamped)
            history['y'].append(y_clamped)

            # Use unclamped values for next iteration (so gradients can flow)
            # This maintains gradient flow through the dynamics

        return torch.stack(history['x']), torch.stack(history['y'])


def run_goodwin_experiment():
    """
    Main experiment: Find optimal policy to stabilize the business cycle.

    Strategy:
    1. Run baseline without policy (reveal the limit cycle)
    2. Optimize policy_strength to minimize variance (stability)
    3. Visualize before/after as phase portraits
    """

    print("=" * 70)
    print("X7: GOODWIN-VOLTERRA CYCLE - OPTIMAL CONTROL DEMONSTRATION")
    print("=" * 70)
    print()

    # Initialize model and optimizer
    # Parameters tuned to generate a clear, pronounced oscillation
    # Classic Goodwin cycle parameters: strong predator-prey dynamics
    model = GoodwinVolterra(alpha=0.2, beta=1.2, delta=1.5, gamma=0.05)
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # --- PHASE 1: BASELINE (No Policy) ---
    print("Phase 1: Baseline Simulation (No Policy)")
    print("-" * 70)

    with torch.no_grad():
        x_base, y_base = model()

    baseline_emp_var = torch.var(x_base).item()
    baseline_wage_var = torch.var(y_base).item()

    print(f"Employment Variance (uncontrolled):  {baseline_emp_var:.6f}")
    print(f"Wage Share Variance (uncontrolled):  {baseline_wage_var:.6f}")
    print()

    # --- PHASE 2: OPTIMIZATION ---
    print("Phase 2: Optimization (Finding Optimal Policy)")
    print("-" * 70)

    losses_over_time = []
    policy_over_time = []

    for epoch in range(101):
        optimizer.zero_grad()
        x, y = model()

        # --- LOSS FUNCTION ---
        # Objective 1: Minimize Deviations from Equilibrium
        # Target: Employment = 0.6, Wage Share = 0.4 (balanced state)
        # This directly penalizes cycles away from this point
        loss_target = 100 * (torch.mean((x - 0.6) ** 2) + torch.mean((y - 0.4) ** 2))

        # Objective 2: Minimize Policy Cost
        # The "Minimal Intervention" principle: prefer small policy adjustments
        loss_cost = 0.01 * model.policy_strength ** 2

        total_loss = loss_target + loss_cost

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Tracking
        losses_over_time.append(total_loss.item())
        policy_over_time.append(model.policy_strength.item())

        if epoch % 20 == 0:
            grad_info = "No grad" if model.policy_strength.grad is None else f"{model.policy_strength.grad.item():.6f}"
            print(
                f"Epoch {epoch:3d} | "
                f"Total Loss: {total_loss.item():.6f} | "
                f"Target Loss: {loss_target.item():.6f} | "
                f"Policy: {model.policy_strength.item():+.4f} | "
                f"∂L/∂k: {grad_info}"
            )

    print()

    # --- PHASE 3: FINAL EVALUATION ---
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    with torch.no_grad():
        x_opt, y_opt = model()

    controlled_emp_var = torch.var(x_opt).item()
    controlled_wage_var = torch.var(y_opt).item()
    final_policy = model.policy_strength.item()

    print(f"Optimal Policy Strength: {final_policy:.4f}")
    print()
    print("Variance Reduction:")
    print(f"  Employment:  {baseline_emp_var:.6f} → {controlled_emp_var:.6f}  "
          f"({100*(1 - controlled_emp_var/baseline_emp_var):.1f}% reduction)")
    print(f"  Wage Share:  {baseline_wage_var:.6f} → {controlled_wage_var:.6f}  "
          f"({100*(1 - controlled_wage_var/baseline_wage_var):.1f}% reduction)")
    print()

    # Mean values
    emp_mean_base = torch.mean(x_base).item()
    emp_mean_opt = torch.mean(x_opt).item()
    wage_mean_base = torch.mean(y_base).item()
    wage_mean_opt = torch.mean(y_opt).item()

    print("Mean Values:")
    print(f"  Employment (baseline): {emp_mean_base:.4f}  →  (controlled): {emp_mean_opt:.4f}")
    print(f"  Wage Share (baseline): {wage_mean_base:.4f}  →  (controlled): {wage_mean_opt:.4f}")
    print()

    # --- VISUALIZATION ---
    print("Generating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Time Series Comparison
    ax = axes[0, 0]
    time = np.arange(len(x_base))
    ax.plot(time, x_base.detach().numpy(), label='Employment (Uncontrolled)',
            linestyle='--', alpha=0.6, linewidth=2)
    ax.plot(time, y_base.detach().numpy(), label='Wage Share (Uncontrolled)',
            linestyle='--', alpha=0.6, linewidth=2)
    ax.plot(time, x_opt.detach().numpy(), label='Employment (Controlled)',
            linewidth=2, color='C2')
    ax.plot(time, y_opt.detach().numpy(), label='Wage Share (Controlled)',
            linewidth=2, color='C3')
    ax.set_title('Time Series: Dampening the Cycle', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Rate / Share')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 2: Phase Portrait (The Limit Cycle)
    ax = axes[0, 1]
    ax.plot(x_base.detach().numpy(), y_base.detach().numpy(),
            linestyle='--', linewidth=2, alpha=0.6, label='Limit Cycle (Uncontrolled)')
    ax.plot(x_opt.detach().numpy(), y_opt.detach().numpy(),
            linewidth=2.5, color='C2', label='Damped Trajectory (Controlled)')
    # Mark equilibrium point
    ax.plot([0.5], [0.4], 'r*', markersize=15, label='Initial State')
    ax.set_title('Phase Portrait: Taming the Cycle', fontsize=12, fontweight='bold')
    ax.set_xlabel('Employment Rate (x)')
    ax.set_ylabel('Wage Share (y)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Plot 3: Loss Convergence
    ax = axes[1, 0]
    ax.plot(losses_over_time, linewidth=2, color='C4')
    ax.set_title('Optimization: Loss Convergence', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 4: Policy Strength Evolution
    ax = axes[1, 1]
    ax.plot(policy_over_time, linewidth=2, color='C5')
    ax.axhline(y=final_policy, color='r', linestyle='--', alpha=0.5, label=f'Final: {final_policy:.4f}')
    ax.set_title('Policy Parameter Evolution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Policy Strength (Control Knob)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/X7_goodwin_volterra_control/goodwin_phase_portrait.png',
                dpi=150, bbox_inches='tight')
    print("✓ Plot saved to: experiments/X7_goodwin_volterra_control/goodwin_phase_portrait.png")
    print()
    print("=" * 70)


if __name__ == "__main__":
    run_goodwin_experiment()
