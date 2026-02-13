"""
X5: Minimal Differentiable Economic Trigger with Variable β Annealing

A bare-bones demonstration of how variable β (inverse temperature) solves
the "chattering" problem in discrete economic models.

The Problem: Binary policy triggers (e.g., "spend if unemployment > threshold")
create oscillations because the response is all-on/all-off.

The Solution: Use sigmoid triggers (soft-max via β), optimize with annealing
(β starts low for smooth exploration, increases for precise stability).

This is the simplest possible model that exhibits the chattering behavior
found in your Stella LowGrow model, and shows how β-annealing fixes it.

Architecture:
- State: Unemployment rate (U)
- Control: Government spending (G)
- Trigger: If U > U_trigger, activate spending (sigmoid controlled by β)
- Dynamics: U changes based on natural decay and stimulus recovery
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class StimulusModel(nn.Module):
    """
    Minimal economic model with policy trigger.

    State Dynamics:
    U(t+1) = U(t) + decay_rate - recovery_rate * G(t)

    Where:
    - U = Unemployment rate [0, 1]
    - G = Government spending level [0, 2]
    - decay_rate = Natural increase in unemployment (0.05 per step)
    - recovery_rate = How much spending reduces unemployment (0.1 per unit G)

    Policy Trigger (Variable β):
    G(t) = G_max * sigmoid(β * (U(t) - U_trigger))

    Where:
    - U_trigger = Unemployment threshold (learnable parameter)
    - β = Inverse temperature controlling trigger hardness
    - Low β: Fuzzy, soft response (smooth gradient for learning)
    - High β: Hard, sharp response (approaches true Heaviside)

    Args:
        beta (float): Inverse temperature (1/T). Default 1.0 (fuzzy).
    """

    def __init__(self, beta=1.0):
        super(StimulusModel, self).__init__()

        # =====================================================================
        # Economic Parameters (Fixed)
        # =====================================================================

        self.u_target = 0.05  # Target unemployment rate (5%)
        self.recovery_rate = 0.1  # How much each unit of G reduces U
        self.decay_rate = 0.05  # Natural increase in U per timestep
        self.g_max = 2.0  # Maximum government spending

        # =====================================================================
        # Policy Control Knob (Learnable)
        # =====================================================================

        # The trigger threshold: What unemployment level triggers spending?
        # Initial guess: 6% (slightly above target)
        self.u_trigger = nn.Parameter(torch.tensor(0.06))

        # =====================================================================
        # Temperature Control
        # =====================================================================

        # Inverse temperature (β = 1/T)
        # Controls how "soft" or "hard" the trigger is
        self.beta = beta

    def get_trigger_response(self, u):
        """
        Compute government spending based on unemployment via sigmoid.

        G = G_max * sigmoid(β * (U - U_trigger))

        This is a "soft-max" version of the hard threshold:
        - If β=0 (infinite temperature): G is flat (no response)
        - If β=1 (room temperature): G transitions smoothly
        - If β→∞ (zero temperature): G approaches Heaviside (0 or G_max)

        Args:
            u (torch.Tensor): Unemployment rate

        Returns:
            torch.Tensor: Government spending level
        """
        # Sigmoid centered on u_trigger
        trigger_signal = torch.sigmoid(self.beta * (u - self.u_trigger))
        g = self.g_max * trigger_signal
        return g

    def forward(self, u_initial, steps=100):
        """
        Simulate the economy through time.

        Process:
        1. For each timestep:
           a. Compute spending based on current unemployment
           b. Update unemployment based on dynamics
           c. Record history
        2. Return trajectories

        Args:
            u_initial (torch.Tensor): Starting unemployment rate
            steps (int): Number of timesteps to simulate

        Returns:
            tuple: (unemployment history, spending history)
        """
        u = u_initial
        u_history = []
        g_history = []

        for _ in range(steps):
            # Compute government response
            g = self.get_trigger_response(u)

            # Update unemployment
            # dU/dt = decay - recovery * G
            u_next = u + (self.decay_rate - self.recovery_rate * g)

            # Constrain to realistic range [1%, 20%]
            u = torch.clamp(u_next, 0.01, 0.20)

            # Record
            u_history.append(u)
            g_history.append(g)

        return torch.stack(u_history), torch.stack(g_history)

    def set_temperature(self, T):
        """
        Set inverse temperature (β = 1/T).

        Args:
            T (float): Temperature. β = 1/T.
        """
        self.beta = 1.0 / max(T, 0.01)


# ============================================================================
# EXPERIMENT 1: HARD vs. SOFT BASELINE COMPARISON
# ============================================================================

def experiment_chattering_vs_stability():
    """
    Demonstrate the chattering problem in hard triggers vs. stability in soft triggers.

    This is the core insight: Binary logic creates oscillations.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: CHATTERING vs. STABILITY")
    print("=" * 70)
    print()
    print("Scenario 1 (Hard): Binary trigger (β=100)")
    print("  - Government spending is 0% or 100%")
    print("  - Creates oscillations (chattering)")
    print()
    print("Scenario 2 (Soft): Sigmoid trigger (β=5)")
    print("  - Government spending ramps smoothly")
    print("  - Allows stability")
    print()

    # Hard case: β very high (binary logic)
    hard_model = StimulusModel(beta=100.0)
    u_hard, g_hard = hard_model(torch.tensor(0.05), steps=100)

    # Soft case: β moderate (sigmoid logic)
    soft_model = StimulusModel(beta=5.0)
    u_soft, g_soft = soft_model(torch.tensor(0.05), steps=100)

    print(f"Hard Case (β=100):")
    print(f"  Mean U: {u_hard.mean():.4f}, Std: {u_hard.std():.4f}")
    print(f"  Mean G: {g_hard.mean():.4f}, Std: {g_hard.std():.4f}")
    print()
    print(f"Soft Case (β=5):")
    print(f"  Mean U: {u_soft.mean():.4f}, Std: {u_soft.std():.4f}")
    print(f"  Mean G: {g_soft.mean():.4f}, Std: {g_soft.std():.4f}")
    print()
    print("→ The hard case oscillates (high std); soft case is stable (low std)")
    print()

    return u_hard, g_hard, u_soft, g_soft


# ============================================================================
# EXPERIMENT 2: OPTIMIZATION WITH FIXED BETA
# ============================================================================

def experiment_optimization_fixed_beta():
    """
    Optimize the trigger threshold to minimize unemployment gap.

    With fixed β=5, show that gradients allow the optimizer to find the
    optimal trigger value.
    """
    print("=" * 70)
    print("EXPERIMENT 2: OPTIMIZATION WITH FIXED β")
    print("=" * 70)
    print()
    print("Goal: Find optimal U_trigger to keep unemployment at 5%")
    print("Method: Adam optimizer with fixed β=5")
    print()

    model = StimulusModel(beta=5.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []
    triggers = []

    print("Running 200 epochs of optimization...")

    for epoch in range(200):
        optimizer.zero_grad()

        # Simulate
        u_hist, _ = model(torch.tensor(0.05), steps=100)

        # Loss: Mean squared error from target
        loss = torch.mean((u_hist - 0.05) ** 2)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track
        losses.append(loss.item())
        triggers.append(model.u_trigger.item())

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}: Loss = {loss.item():.6f}, U_trigger = {model.u_trigger.item():.4f}")

    print()
    print(f"Final optimized U_trigger: {model.u_trigger.item():.4f}")
    print(f"This means: Spend when unemployment exceeds {model.u_trigger.item()*100:.2f}%")
    print()

    # Verify: Run with optimized trigger
    with torch.no_grad():
        u_opt, g_opt = model(torch.tensor(0.05), steps=100)

    print(f"With optimized trigger:")
    print(f"  Mean U: {u_opt.mean():.4f} (target: 0.05)")
    print(f"  Std U:  {u_opt.std():.4f} (lower is more stable)")
    print()

    return losses, triggers, u_opt, g_opt


# ============================================================================
# EXPERIMENT 3: ANNEALING (VARIABLE BETA)
# ============================================================================

def experiment_annealing():
    """
    Optimize with annealing: Start fuzzy (low β), gradually harden (increase β).

    This demonstrates how annealing can escape local minima and find
    robust solutions.
    """
    print("=" * 70)
    print("EXPERIMENT 3: ANNEALING (VARIABLE β)")
    print("=" * 70)
    print()
    print("Process:")
    print("  Phase 1 (Epochs 0-50):   β = 1.0  (Fuzzy exploration)")
    print("  Phase 2 (Epochs 50-150): β = 5.0  (Moderate transition)")
    print("  Phase 3 (Epochs 150-300): β = 20.0 (Sharp refinement)")
    print()

    model = StimulusModel(beta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    losses = []
    betas = []
    triggers = []

    print("Running 300 epochs with annealing schedule...")

    for epoch in range(300):
        # Anneal temperature
        if epoch < 50:
            model.set_temperature(1.0)  # β = 1.0
        elif epoch < 150:
            # Linear interpolation from T=1 to T=0.2 (β: 1.0 to 5.0)
            t = (epoch - 50) / 100.0
            T = 1.0 - 0.8 * t
            model.set_temperature(T)
        else:
            model.set_temperature(0.05)  # β = 20.0

        optimizer.zero_grad()

        # Simulate
        u_hist, _ = model(torch.tensor(0.05), steps=100)

        # Loss
        loss = torch.mean((u_hist - 0.05) ** 2)

        # Backward
        loss.backward()
        optimizer.step()

        # Track
        losses.append(loss.item())
        betas.append(model.beta)
        triggers.append(model.u_trigger.item())

        if epoch % 50 == 0:
            print(
                f"  Epoch {epoch:3d}: β = {model.beta:6.2f}, Loss = {loss.item():.6f}, "
                f"U_trigger = {model.u_trigger.item():.4f}"
            )

    print()
    print(f"Final trigger: {model.u_trigger.item():.4f}")
    print()

    # Test at different β values to verify robustness
    print("Robustness Check: Testing trigger at different β values")
    with torch.no_grad():
        for test_beta in [1.0, 5.0, 20.0, 100.0]:
            model.set_temperature(1.0 / test_beta)
            u_test, _ = model(torch.tensor(0.05), steps=100)
            print(f"  β = {test_beta:6.1f}: Mean U = {u_test.mean():.4f}, Std = {u_test.std():.4f}")

    print()

    return losses, betas, triggers


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(exp1_results, exp2_results, exp3_results):
    """Create comprehensive visualization of all experiments."""
    u_hard, g_hard, u_soft, g_soft = exp1_results
    losses_fixed, triggers_fixed, u_opt, g_opt = exp2_results
    losses_anneal, betas_anneal, triggers_anneal = exp3_results

    fig = plt.figure(figsize=(16, 12))

    # =====================================================================
    # EXPERIMENT 1: Hard vs. Soft
    # =====================================================================

    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(u_hard.detach().numpy(), label="Hard (β=100)", alpha=0.7, linewidth=2)
    ax1.plot(u_soft.detach().numpy(), label="Soft (β=5)", linewidth=2)
    ax1.axhline(y=0.05, color='red', linestyle='--', label='Target (5%)')
    ax1.set_title('Unemployment: Chattering vs. Stability', fontweight='bold')
    ax1.set_ylabel('Unemployment Rate')
    ax1.set_xlabel('Time Step')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(g_hard.detach().numpy(), label="Hard G", alpha=0.7, linewidth=2)
    ax2.plot(g_soft.detach().numpy(), label="Soft G", linewidth=2)
    ax2.set_title('Government Spending: Binary vs. Smooth', fontweight='bold')
    ax2.set_ylabel('Spending Level')
    ax2.set_xlabel('Time Step')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3, 3, 3)
    ax3.bar(['Hard (β=100)', 'Soft (β=5)'],
            [u_hard.std().item(), u_soft.std().item()],
            color=['red', 'green'], alpha=0.7)
    ax3.set_title('Unemployment Volatility (Std Dev)', fontweight='bold')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_ylim(0, 0.015)

    # =====================================================================
    # EXPERIMENT 2: Fixed β Optimization
    # =====================================================================

    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(losses_fixed, linewidth=2, color='blue')
    ax4.set_title('Loss Convergence (Fixed β=5)', fontweight='bold')
    ax4.set_ylabel('Loss (MSE)')
    ax4.set_xlabel('Epoch')
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(triggers_fixed, linewidth=2, color='green')
    ax5.set_title('Optimized Trigger Evolution', fontweight='bold')
    ax5.set_ylabel('U_trigger')
    ax5.set_xlabel('Epoch')
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(u_opt.detach().numpy(), label='Optimized', linewidth=2)
    ax6.axhline(y=0.05, color='red', linestyle='--', label='Target')
    ax6.set_title('Final Unemployment (Optimized)', fontweight='bold')
    ax6.set_ylabel('Unemployment Rate')
    ax6.set_xlabel('Time Step')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # =====================================================================
    # EXPERIMENT 3: Annealing
    # =====================================================================

    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(losses_anneal, linewidth=2, color='purple')
    ax7.set_title('Loss with Annealing', fontweight='bold')
    ax7.set_ylabel('Loss (MSE)')
    ax7.set_xlabel('Epoch')
    ax7.grid(True, alpha=0.3)

    ax8 = plt.subplot(3, 3, 8)
    ax8_twin = ax8.twinx()
    ax8.plot(triggers_anneal, linewidth=2, color='orange', label='U_trigger')
    ax8_twin.plot(betas_anneal, linewidth=2, color='blue', alpha=0.5, label='β')
    ax8.set_title('Trigger & Temperature Evolution', fontweight='bold')
    ax8.set_ylabel('U_trigger', color='orange')
    ax8_twin.set_ylabel('β (Inverse Temperature)', color='blue')
    ax8.set_xlabel('Epoch')
    ax8.grid(True, alpha=0.3)

    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(betas_anneal, linewidth=2, color='red')
    ax9.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='Phase transitions')
    ax9.axvline(x=150, color='gray', linestyle='--', alpha=0.5)
    ax9.set_title('β Annealing Schedule', fontweight='bold')
    ax9.set_ylabel('β (Inverse Temperature)')
    ax9.set_xlabel('Epoch')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all experiments."""
    print("\n" + "=" * 70)
    print("X5: MINIMAL DIFFERENTIABLE ECONOMIC TRIGGER WITH VARIABLE β")
    print("=" * 70)

    # Experiment 1: Chattering vs. Stability
    exp1_results = experiment_chattering_vs_stability()

    # Experiment 2: Fixed β Optimization
    exp2_results = experiment_optimization_fixed_beta()

    # Experiment 3: Annealing
    exp3_results = experiment_annealing()

    # Visualization
    print("\nGenerating visualization...")
    visualize_results(exp1_results, exp2_results, exp3_results)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print()
    print("1. CHATTERING PROBLEM:")
    print("   Hard triggers (β=100) create oscillations because binary")
    print("   logic causes the system to overshoot equilibrium repeatedly.")
    print()
    print("2. SOFT OPTIMIZATION:")
    print("   Soft triggers (low β) allow gradient descent to find the")
    print("   optimal trigger point that minimizes unemployment volatility.")
    print()
    print("3. ANNEALING:")
    print("   Variable β enables simulated annealing: start fuzzy (explore),")
    print("   cool gradually (refine), end precise (lock in robust policy).")
    print()
    print("4. STELLA CONNECTION:")
    print("   Your LowGrow model exhibits the same chattering because it uses")
    print("   discrete triggers. Applying variable β annealing would smooth")
    print("   the oscillations and enable automated policy discovery.")
    print()


if __name__ == "__main__":
    main()
