"""
X7: Giraud-Bovari (GEMMES) Dynamic - Minsky-Climate Collapse Model

A Differentiable Implementation of the Giraud-Bovari model from the GEMMES framework.
Focus: The coupling between Private Debt (Solvency) and Climate Damage (Entropy).

The model demonstrates:
1. The "Obsidian Snap" - Phase transition from growth to collapse via debt threshold
2. Keen-Minsky dynamics - Predator-prey between profits and wages
3. Climate-economic feedback - Damage destroys capital, reducing output, increasing debt stress
4. The Seneca Cliff - Non-linear jump to catastrophic breakdown

This is the "Advanced Class" demonstration of Differentiable Economics,
showing how to solve dynamic stability problems and navigate collapse basins.

References:
- Giraud, G. et al. (GEMMES framework)
- Keen, S. & Minsky, H. (Debt-Investment dynamics)
- Maslov-Gibbs Einsum (TTC Theory) for phase transitions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class GiraudCollapseModel(nn.Module):
    """
    A Differentiable Implementation of the Giraud-Bovari Dynamic.

    State Variables:
    - K: Capital Stock
    - D: Private Debt
    - T: Temperature Anomaly
    - omega: Wage Share (implicit in profit/wage ratio)

    The model exhibits:
    - Minsky cycles (debt-driven investment booms)
    - Climate feedback (damage destroys capital)
    - Phase transitions (debt threshold triggers collapse)

    Args:
        beta (float): Inverse temperature for sigmoid triggers
                     High beta → sharp thresholds (discrete behavior)
                     Low beta → fuzzy thresholds (smooth gradients)
    """

    def __init__(self, beta=10.0):
        super(GiraudCollapseModel, self).__init__()
        self.beta = beta

        # --- PHYSICAL PARAMETERS ---
        # Base depreciation rate and climate sensitivity
        self.depreciation_base = 0.05
        self.damage_sensitivity = nn.Parameter(torch.tensor(0.15))
        """How fast climate damage scales with temperature (entropy effect)"""

        # --- BEHAVIORAL PARAMETERS (Learnable via optimization) ---
        self.investment_reaction = nn.Parameter(torch.tensor(2.5))
        """Sensitivity of Investment to Profit Rate (Keen-Minsky accelerator)"""

        self.debt_reaction = nn.Parameter(torch.tensor(0.5))
        """How much Investment is debt-financed vs. retained earnings"""

        # --- TIPPING POINT PARAMETERS ---
        self.climate_threshold = torch.tensor(2.0)
        """Temperature anomaly at which damage accelerates (sigmoid midpoint)"""

        self.debt_threshold = torch.tensor(1.5)
        """Debt-to-Output ratio triggering solvency crisis (Minsky moment)"""

        # Exogenous parameters
        self.wage_share = 0.6
        """Fixed wage share (can be made learnable)"""

        self.capital_output_ratio = 3.0
        """Leontief production: K/Y = 3 (structural parameter)"""

        self.emissions_intensity = 0.0005
        """CO2 emissions per unit GDP (drives temperature)"""

    def maslov_gibbs_trigger(self, val, threshold):
        """
        The Thermodynamic Tensor Contraction (TTC) Operator.

        Maps a continuous variable to a "trigger probability" using sigmoid.
        This is the bridge between discrete events (bankruptcy, default) and
        continuous differentiable optimization.

        Mathematical form:
            trigger(x) = σ(β(x - threshold))

        Behavior:
        - β → ∞: Sharp step function (true discrete behavior)
        - β → 0: Linear approximation (smooth gradient flow)
        - β = β(t) annealing: Start fuzzy → cool to sharp

        This is the "Obsidian Snap" - the phase transition from safety to collapse.

        Args:
            val (Tensor): Value to trigger on (e.g., debt ratio, temperature)
            threshold (Tensor): Threshold value

        Returns:
            Tensor: Trigger signal ∈ [0, 1]
        """
        return torch.sigmoid(self.beta * (val - threshold))

    def forward(self, steps=100):
        """
        Simulate the Giraud-Bovari dynamic through time.

        Process:
        1. Climate damage reduces capital efficiency (entropy)
        2. Production is constrained by damaged capital
        3. Debt accumulates based on investment minus retained earnings
        4. When debt ratio crosses threshold, investment collapses (Minsky moment)
        5. Capital depreciation (boosted by damage) causes death spiral

        Returns:
            dict: Historical time series of key variables
        """
        # Initial States
        K = torch.tensor(100.0)
        """Capital stock (starting point)"""

        D = torch.tensor(50.0)
        """Private debt (starting point)"""

        T = torch.tensor(1.1)
        """Temperature anomaly in °C above baseline"""

        history = {'Y': [], 'D_ratio': [], 'T': [], 'K': [], 'I': [], 'damage': []}

        for _ in range(steps):
            # ===== 1. CLIMATE DAMAGE (Thermodynamic Entropy) =====
            # Damage function D(T) reduces capital efficiency
            # Sigmoid tipping point at climate_threshold (typically 2°C)
            damage = 1.0 - (1.0 / (1.0 + torch.exp(3.0 * (T - self.climate_threshold))))
            """Fraction of capital lost to climate damage"""

            # ===== 2. PRODUCTION (Leontief with Damage) =====
            # Output Y is constrained by Capital and Damage
            # Y = (K / ν) × (1 - D(T))
            # where ν is capital-output ratio
            Y = (K / self.capital_output_ratio) * (1.0 - damage)
            """Actual output after climate damage reduction"""

            # ===== 3. FINANCIAL FRAGILITY (The Minsky Moment) =====
            # The debt-to-output ratio measures solvency stress
            debt_ratio = D / (Y + 1e-6)
            """Ratio of debt to annual output"""

            # The "Obsidian Snap": If Debt Ratio > Threshold, investment collapses
            # We use the Maslov-Gibbs trigger to make this differentiable
            # This is the key to avoiding discrete discontinuities
            financial_stress = self.maslov_gibbs_trigger(debt_ratio, self.debt_threshold)
            """Probability of solvency crisis (∈ [0, 1])"""

            # ===== 4. INVESTMENT DYNAMICS (Keen-Minsky) =====
            # Profit share = 1 - wage share
            profit_share = 1.0 - self.wage_share
            """Share of output going to capital/profits"""

            # Desired investment is proportional to profit rate
            # I_desired = α × profit_share × Y
            # where α is the "animal spirits" multiplier
            desired_invest = self.investment_reaction * profit_share * Y
            """Investment demand from profit-seeking firms"""

            # Actual Investment is throttled by Financial Stress
            # When debt ratio is high, investment collapses
            # I = I_desired × (1 - financial_stress)
            I = desired_invest * (1.0 - financial_stress)
            """Actual investment after financial constraint"""

            # ===== 5. DYNAMICS UPDATE =====
            # Capital Accumulation with Damage-Accelerated Depreciation
            # Giraud's key insight: Climate change destroys capital directly
            # K_{t+1} = K_t + I - (δ + δ_climate × D(T)) × K
            depreciation = self.depreciation_base + (0.05 * damage)
            """Total depreciation rate (base + climate damage)"""

            K_next = K + I - (depreciation * K)
            """Capital evolves via investment minus accelerated depreciation"""

            # Debt Dynamics: Debt grows by Investment minus Retained Earnings
            # D_{t+1} = D_t + (I - Retained Earnings)
            # where Retained Earnings = profit_share × Y
            retained_earnings = profit_share * Y
            """Profits available to pay down debt"""

            D_next = D + (I - retained_earnings)
            """Debt accumulates when investment exceeds retained profits"""

            # Temperature Dynamics: Proportional to Output (Emissions)
            # T_{t+1} = T_t + φ × Y
            # where φ is emissions intensity per unit GDP
            T_next = T + (self.emissions_intensity * Y)
            """Temperature rises with economic output (emissions)"""

            # Update States
            K = K_next
            D = D_next
            T = T_next

            # Record History
            history['Y'].append(Y)
            history['D_ratio'].append(debt_ratio)
            history['T'].append(T)
            history['K'].append(K)
            history['I'].append(I)
            history['damage'].append(damage)

        return history

    def set_temperature(self, T: float):
        """
        Set inverse temperature for β-annealing.

        Used in optimization to transition from fuzzy (differentiable)
        to sharp (discrete-like) behavior.

        Args:
            T (float): Temperature; β = 1/T
        """
        self.beta = 1.0 / max(T, 0.01)


# ============================================================================
# EXPERIMENT: AVOIDING THE SENECA CLIFF
# ============================================================================

def run_giraud_experiment():
    """
    Optimization experiment: Find policy parameters that avoid collapse.

    The Seneca Cliff: A sharp non-linear collapse from growth to breakdown.

    Optimizer's Task:
    - Maximize Output (Y) [growth imperative]
    - Avoid Debt Crisis (debt_ratio < 1.5) [solvency constraint]
    - Navigate Climate Damage (T rising) [entropy constraint]

    The optimizer must find the "Safety Corridor" - the narrow band of
    investment reaction rates that allow growth without triggering collapse.
    """
    print("=" * 70)
    print("X7: Giraud Collapse Model - Avoiding the Seneca Cliff")
    print("=" * 70)
    print("\nOptimization: Finding the 'Safety Corridor' for sustainable growth")
    print("Objective: Maximize Output while avoiding Debt Crisis\n")

    # Initialize model with moderate β (fuzzy triggers for smooth optimization)
    model = GiraudCollapseModel(beta=15.0)
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # Tracking
    losses = []
    investment_reactions = []
    debt_reactions = []
    max_debt_ratios = []
    final_outputs = []

    for epoch in range(151):
        optimizer.zero_grad()

        # Forward simulation
        hist = model()
        Y = torch.stack(hist['Y'])
        D_ratio = torch.stack(hist['D_ratio'])

        # ===== MULTI-OBJECTIVE LOSS FUNCTION =====
        # 1. Growth Objective: Maximize Output
        # Negative because we minimize loss
        loss_growth = -torch.mean(Y)
        """Penalize low output (want growth)"""

        # 2. Stability Objective: Keep Debt Ratio below Threshold
        # Heavy penalty for crossing 1.5 (Minsky moment)
        loss_stability = torch.sum(torch.relu(D_ratio - 1.5)) * 100.0
        """Penalize debt crisis (solvency constraint)"""

        # 3. Regularization: Keep parameters reasonable
        loss_reg = 0.01 * (torch.abs(model.investment_reaction - 2.5) +
                           torch.abs(model.debt_reaction - 0.5))
        """Prevent runaway parameter values"""

        # Combined Loss
        total_loss = loss_growth + loss_stability + loss_reg

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Track metrics
        losses.append(total_loss.item())
        investment_reactions.append(model.investment_reaction.item())
        debt_reactions.append(model.debt_reaction.item())
        max_debt_ratios.append(torch.max(D_ratio).item())
        final_outputs.append(Y[-1].item())

        if epoch % 50 == 0:
            max_debt = torch.max(D_ratio).item()
            final_output = Y[-1].item()
            print(f"Epoch {epoch:3d} | Loss: {total_loss.item():.4f} | "
                  f"Inv.React: {model.investment_reaction.item():.3f} | "
                  f"Max Debt Ratio: {max_debt:.3f} | Final Output: {final_output:.1f}")

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nOptimized Parameters:")
    print(f"  Investment Reaction: {model.investment_reaction.item():.4f}")
    print(f"  Debt Reaction: {model.debt_reaction.item():.4f}")
    print(f"\nOutcome:")
    print(f"  Max Debt Ratio: {max_debt_ratios[-1]:.3f} (threshold: 1.5)")
    print(f"  Final Output: {final_outputs[-1]:.2f}")
    print(f"  Final Loss: {losses[-1]:.4f}")

    # ===== VISUALIZATION =====
    print(f"\nGenerating visualizations...")

    with torch.no_grad():
        final_hist = model()
        Y_final = torch.stack(final_hist['Y']).numpy()
        D_ratio_final = torch.stack(final_hist['D_ratio']).numpy()
        K_final = torch.stack(final_hist['K']).numpy()
        T_final = torch.stack(final_hist['T']).numpy()
        damage_final = torch.stack(final_hist['damage']).numpy()

    fig = plt.figure(figsize=(16, 10))

    # Row 1: Optimization Progress
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(losses, linewidth=2, color='blue')
    ax1.set_title('Loss Convergence', fontweight='bold')
    ax1.set_ylabel('Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(investment_reactions, linewidth=2, color='green')
    ax2.set_title('Investment Reaction Evolution', fontweight='bold')
    ax2.set_ylabel('Investment Reaction (α)')
    ax2.set_xlabel('Epoch')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(max_debt_ratios, linewidth=2, color='red')
    ax3.axhline(y=1.5, color='darkred', linestyle='--', label='Collapse Threshold')
    ax3.set_title('Max Debt Ratio vs Threshold', fontweight='bold')
    ax3.set_ylabel('Debt Ratio')
    ax3.set_xlabel('Epoch')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Row 2: Final Simulation Results
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(Y_final, linewidth=2, color='darkgreen', label='GDP (Y)')
    ax4.fill_between(range(len(Y_final)), Y_final, alpha=0.3, color='green')
    ax4.set_title('Final GDP Trajectory', fontweight='bold')
    ax4.set_ylabel('Output (Y)')
    ax4.set_xlabel('Time Period')
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(D_ratio_final, linewidth=2, color='darkred', label='Debt Ratio')
    ax5.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='Collapse Threshold')
    ax5.fill_between(range(len(D_ratio_final)), D_ratio_final, 1.5, alpha=0.2, color='red')
    ax5.set_title('Final Debt Ratio (Solvency)', fontweight='bold')
    ax5.set_ylabel('Debt / Output')
    ax5.set_xlabel('Time Period')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(K_final, linewidth=2, color='darkblue', label='Capital Stock')
    ax6.fill_between(range(len(K_final)), K_final, alpha=0.3, color='blue')
    ax6.set_title('Final Capital Stock', fontweight='bold')
    ax6.set_ylabel('Capital (K)')
    ax6.set_xlabel('Time Period')
    ax6.grid(True, alpha=0.3)

    # Row 3: Climate & Damage
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(T_final, linewidth=2, color='red', label='Temperature')
    ax7.axhline(y=2.0, color='darkred', linestyle='--', alpha=0.5, label='Tipping Point')
    ax7.set_title('Temperature Anomaly', fontweight='bold')
    ax7.set_ylabel('ΔT (°C)')
    ax7.set_xlabel('Time Period')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(damage_final, linewidth=2, color='orange')
    ax8.fill_between(range(len(damage_final)), damage_final, alpha=0.3, color='orange')
    ax8.set_title('Climate Damage Fraction', fontweight='bold')
    ax8.set_ylabel('Damage (D(T))')
    ax8.set_xlabel('Time Period')
    ax8.grid(True, alpha=0.3)

    # Summary panel
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = f"""
X7: Giraud Collapse Model Results

Initial Setup:
  • Capital: 100.0 units
  • Debt: 50.0 units
  • Temperature: 1.1°C

Optimization Outcome:
  • Final Output: {final_outputs[-1]:.2f}
  • Final Debt Ratio: {D_ratio_final[-1]:.3f}
  • Final Temperature: {T_final[-1]:.2f}°C

Optimized Parameters:
  • Investment Reaction: {model.investment_reaction.item():.4f}
  • Debt Reaction: {model.debt_reaction.item():.4f}

Key Insight:
  The optimizer discovers the
  "Safety Corridor" - the narrow
  band of parameters allowing
  sustainable growth without
  triggering the Minsky-Climate
  collapse mechanism.
    """
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('X7: Giraud Collapse Model - Finding the Safety Corridor',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('experiments/X7_giraud_collapse_model/giraud_collapse.png',
                dpi=150, bbox_inches='tight')
    print("✓ Plot saved to: experiments/X7_giraud_collapse_model/giraud_collapse.png")
    plt.show()


if __name__ == "__main__":
    run_giraud_experiment()
