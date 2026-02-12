import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class DifferentiableIO(nn.Module):
    def __init__(self, num_sectors):
        super(DifferentiableIO, self).__init__()
        self.num_sectors = num_sectors

        # 1. BASELINE TECHNICAL COEFFICIENTS (Learnable)
        # Represents the "ideal" or "pre-industrial" efficiency of the economy.
        self.A_raw = nn.Parameter(torch.randn(num_sectors, num_sectors) * 0.01)

        # 2. CLIMATE DAMAGE PARAMETERS (Based on the "Recalibrating Climate Risk" report)
        # s: Sensitivity (steepness of the tipping point)
        # threshold: The temperature (C) where non-linear damages accelerate
        self.tp_sensitivity = nn.Parameter(torch.tensor(2.5))
        self.tp_threshold = nn.Parameter(torch.tensor(2.0))

        # Gamma: Sectoral vulnerability (how much climate damage inflates input requirements)
        # For simplicity, we assume all sectors are equally vulnerable here.
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def get_damage_fraction(self, temperature):
        """
        Tipping Point Damage Function: D = 1 - 1 / (1 + exp(s * (T - T_thresh)))
        Models a non-linear economic collapse as temperature rises.
        """
        return 1.0 - (1.0 / (1.0 + torch.exp(self.tp_sensitivity * (temperature - self.tp_threshold))))

    def get_A(self, temperature=None):
        """
        Returns the damaged technical coefficients matrix A.
        As temperature increases, A_ij increases (efficiency decreases).
        """
        # Base A constrained to [0, 0.5] for stability
        A_baseline = torch.sigmoid(self.A_raw) * 0.5

        if temperature is not None:
            damage = self.get_damage_fraction(temperature)
            # Climate damage acts as an 'inefficiency multiplier'
            # A_damaged = A_baseline * (1 + gamma * damage)
            # Note: If A_damaged rows/cols approach 1, the economy becomes singular (collapse).
            A_damaged = A_baseline * (1.0 + self.gamma * damage)
            return A_damaged

        return A_baseline

    def forward(self, final_demand, temperature=None):
        """
        Leontief Solution with Climate Feedback: x = (I - A(T))^-1 * d
        """
        A = self.get_A(temperature)
        I = torch.eye(self.num_sectors)

        # Solving the linear system (I - A)x = d
        # Gradients flow through the solver to A, sensitivity, and threshold.
        total_output = torch.linalg.solve(I - A, final_demand)
        return total_output

# --- Example Usage: Calibrating & Analyzing Climate Risk ---

# Observed Data (Pre-damage state)
observed_x = torch.tensor([150.0, 200.0, 180.0])
observed_d = torch.tensor([50.0, 80.0, 60.0])
temp_baseline = torch.tensor(1.1)  # Current warming level

model = DifferentiableIO(num_sectors=3)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("Step 1: Calibrating Baseline Economy...")
for epoch in range(501):
    optimizer.zero_grad()
    # Assume observed data is from the current (1.1C) temperature
    predicted_x = model(observed_d, temperature=temp_baseline)
    loss = criterion(predicted_x, observed_x)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

# Step 2: Stress Testing the Economy (The "Control Knob")
print("\nStep 2: Simulating Climate Tipping Point Impact...")
temp_range = torch.linspace(1.1, 4.0, 30)
outputs = []
damages = []

with torch.no_grad():
    for t in temp_range:
        x = model(observed_d, temperature=t)
        d = model.get_damage_fraction(t)
        outputs.append(x.sum().item())
        damages.append(d.item())

# Visualization of the Differentiable Manifold
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(temp_range.numpy(), outputs, color='blue', lw=2)
plt.axvline(x=2.0, color='red', linestyle='--', label='Tipping Threshold (2°C)')
plt.title("Total Economic Output (x) vs Temperature")
plt.xlabel("Global Mean Temperature (°C)")
plt.ylabel("Aggregate Output")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(temp_range.numpy(), damages, color='orange', lw=2)
plt.title("Climate Damage Fraction (D)")
plt.xlabel("Global Mean Temperature (°C)")
plt.ylabel("Efficiency Loss %")

plt.tight_layout()
plt.show()

print("\nAnalysis Complete: The model now quantifies how non-linear climate")
print("damage propagates through sectoral interdependencies.")
