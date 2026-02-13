"""
X3: Tropical Supply Chain - "Hello World" for Bottleneck-Constrained Economics

A minimal demonstration of the Tropical (Min-Plus) Semi-ring applied to supply chains.
Uses a simple 4-node linear chain (Source → Factory → Distributor → Customer)
to show how gradient descent discovers resilience solutions.

The key insight: In a bottleneck-constrained system, the optimizer can compute
the exact buffer required at each node to maintain target output despite upstream shocks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class TropicalSupplyChain(nn.Module):
    """
    A differentiable supply chain using Tropical (Min-Plus) algebra.

    Topology: Linear 4-node chain
    - Node 0: Source (raw materials)
    - Node 1: Factory (manufacturing)
    - Node 2: Distributor (logistics)
    - Node 3: Customer (final demand)

    The tropical semi-ring uses the MIN operator to represent bottlenecks:
    "Output is limited by the minimum of capacity and upstream supply."

    Learnable Parameters:
    - buffers: Inventory/redundancy at each node (the "control knobs")

    Args:
        beta (float): Inverse temperature (1/T) for soft-min hardness
                     High beta → pure tropical min (hard bottleneck)
                     Low beta → soft approximation (forgiving bottleneck)
    """

    def __init__(self, beta=20.0):
        super(TropicalSupplyChain, self).__init__()
        self.num_nodes = 4

        # Bill-of-Materials: Adjacency matrix defining supply chain topology
        # A[i, j] = 1 if node j is a required input to node i
        # Linear chain: each node depends on the previous one
        self.register_buffer('BOM', torch.tensor([
            [0, 0, 0, 0],  # Node 0 (Source): No dependencies
            [1, 0, 0, 0],  # Node 1 (Factory): Requires Node 0
            [0, 1, 0, 0],  # Node 2 (Distributor): Requires Node 1
            [0, 0, 1, 0]   # Node 3 (Customer): Requires Node 2
        ], dtype=torch.float32))

        self.beta = beta

        # Learnable buffer parameters (logits for sigmoid)
        # Initialized at -5.0 (sigmoid ≈ 0.007) to start with minimal buffers
        self.register_parameter('buffer_logits', nn.Parameter(torch.full((1, 4), -5.0)))

    def soft_min(self, x, dim=-1):
        """
        Differentiable "Dequantization" of the Tropical Min operator.

        Uses Log-Sum-Exp trick: soft_min(x) ≈ -log(sum(exp(-β*x))) / β

        As β→∞, this converges to the true min function.
        As β→0, this approaches a smooth average.

        Args:
            x (torch.Tensor): Input tensor
            dim (int): Dimension along which to compute soft min

        Returns:
            torch.Tensor: Soft minimum values
        """
        return -(1.0 / self.beta) * torch.logsumexp(-self.beta * x, dim=dim)

    def forward(self, shocks):
        """
        Forward pass: Propagate shocks through supply chain.

        Process:
        1. Each node has initial capacity = exogenous_shock + buffer
        2. For linear chain: output[i] = min(capacity[i], input[i-1])
        3. Return final outputs at all nodes

        The "Tropical Collapse": If any node fails, all downstream nodes fail.
        Buffers act as shock absorbers to prevent propagation.

        Args:
            shocks (torch.Tensor): Exogenous capacity shocks [1 × num_nodes]
                                  (e.g., supply disruptions, natural disasters)

        Returns:
            torch.Tensor: Final output at each node after bottleneck constraints
        """
        # Add buffers to shock resilience
        # Sigmoid ensures buffers are in [0, 1] range
        effective_capacity = torch.clamp(
            shocks + torch.sigmoid(self.buffer_logits),
            0.0, 1.0
        )

        # Propagate through linear chain
        node_outputs = [effective_capacity[:, 0]]

        for i in range(1, self.num_nodes):
            # Current node's output limited by:
            # - Its own capacity
            # - Input from upstream (previous node)
            current_capacity = effective_capacity[:, i]
            upstream_input = node_outputs[i - 1]

            # Tropical composition: min(capacity, input)
            combined = torch.stack([current_capacity, upstream_input], dim=-1)
            node_outputs.append(self.soft_min(combined, dim=-1))

        return torch.stack(node_outputs, dim=1)


# ============================================================================
# EXPERIMENT: REVERSE STRESS TEST
# ============================================================================

def run_experiment():
    """
    Reverse Stress Test: Find minimum buffers needed to absorb shock.

    Scenario:
    - The Source (Node 0) experiences a severe disruption: 20% capacity
    - Other nodes are unaffected: 100% capacity
    - Goal: Maintain Customer (Node 3) at 80% output

    Question: Where should we place inventory buffers to absorb this shock?

    Expected Result:
    - Most buffer should be at Node 0 (Source) — the bottleneck
    - Minimal buffers at downstream nodes
    - This reveals the network's "critical path"
    """

    print("=" * 70)
    print("TROPICAL SUPPLY CHAIN: REVERSE STRESS TEST")
    print("=" * 70)
    print()
    print("Scenario: Source disruption to 20% capacity")
    print("Goal: Maintain Customer output at 80% using strategic buffers")
    print()

    # Initialize model
    model = TropicalSupplyChain(beta=30.0)  # High beta for hard bottlenecks
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # Define shock: Source drops to 20%, others normal
    shocks = torch.tensor([[0.2, 1.0, 1.0, 1.0]])
    target_customer_output = 0.8

    # Tracking for visualization
    losses = []
    buffer_evolution = []

    print(f"Initial buffer values: {torch.sigmoid(model.buffer_logits).detach().numpy().flatten()}")
    print()
    print("Running optimization...")
    print()

    # Optimization loop
    for epoch in range(151):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(shocks)
        customer_output = outputs[0, 3]  # Node 3 (Customer)

        # Loss function: (1) Hit target, (2) Minimize buffer cost
        loss_target_miss = (customer_output - target_customer_output) ** 2
        loss_buffer_cost = 0.01 * torch.sum(torch.sigmoid(model.buffer_logits))

        total_loss = loss_target_miss + loss_buffer_cost

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Tracking
        losses.append(total_loss.item())
        buffer_evolution.append(torch.sigmoid(model.buffer_logits).detach().numpy().flatten())

        if epoch % 50 == 0:
            print(
                f"Epoch {epoch:3d} | Total Loss: {total_loss.item():.6f} | "
                f"Customer Output: {customer_output.item():.4f} | "
                f"Target: {target_customer_output:.4f}"
            )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    # Final buffer allocation
    final_buffers = torch.sigmoid(model.buffer_logits).detach().numpy().flatten()
    nodes = ["Source", "Factory", "Distributor", "Customer"]
    total_buffer = sum(final_buffers)

    print("Optimal Buffer Allocation (Inventory/Redundancy):")
    print()
    for i, name in enumerate(nodes):
        pct = 100 * final_buffers[i] / total_buffer if total_buffer > 0 else 0
        print(f"  {name:12}: {final_buffers[i]:.4f}  ({pct:5.1f}% of total)")

    print()
    print(f"Total Buffer Cost: {total_buffer:.4f}")
    print()

    # Final output
    with torch.no_grad():
        final_outputs = model(shocks)[0].detach().numpy()

    print("Final Supply Chain Output:")
    print()
    for i, name in enumerate(nodes):
        print(f"  {name:12}: {final_outputs[i]:.4f}")

    print()
    print("Key Insight: The optimizer identified Node 0 (Source) as the critical")
    print("bottleneck and allocated most buffer there, while minimizing redundancy")
    print("at downstream nodes. This matches network theory: in a linear chain,")
    print("the most upstream node is the chokepoint.")
    print()

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Loss convergence
    axes[0].plot(losses, linewidth=2, color='blue')
    axes[0].set_title('Loss Convergence', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss (Target Miss + Buffer Cost)')
    axes[0].grid(True, alpha=0.3)

    # Right plot: Buffer evolution
    buffer_evolution = torch.tensor(buffer_evolution).numpy()
    for i, name in enumerate(nodes):
        axes[1].plot(buffer_evolution[:, i], label=name, linewidth=2)

    axes[1].set_title('Buffer Evolution (Control Knobs)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Buffer Size (Normalized)')
    axes[1].legend(loc='center right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/X3_tropical_supply_chain/tropical_supply_chain.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved to: experiments/X3_tropical_supply_chain/tropical_supply_chain.png")

    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
