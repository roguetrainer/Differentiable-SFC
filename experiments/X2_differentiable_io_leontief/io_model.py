import torch
import torch.nn as nn
import torch.optim as optim

class DifferentiableIO(nn.Module):
    def __init__(self, num_sectors):
        super(DifferentiableIO, self).__init__()
        self.num_sectors = num_sectors

        # We define the technical coefficients matrix 'A' as a learnable parameter.
        # Initializing with small random values.
        # In a real scenario, we would initialize with a prior (historical IO table).
        self.A_raw = nn.Parameter(torch.randn(num_sectors, num_sectors) * 0.01)

    def get_A(self):
        # Constraints: Technical coefficients must be non-negative.
        # We use a sigmoid or softplus to ensure 0 <= A_ij.
        # We also want to ensure the column sums are <= 1 for accounting consistency.
        return torch.sigmoid(self.A_raw) * 0.5

    def forward(self, final_demand):
        """
        Leontief Solution: x = (I - A)^-1 * d
        Where:
            x is Total Output
            I is Identity Matrix
            A is Technical Coefficients Matrix
            d is Final Demand
        """
        A = self.get_A()
        I = torch.eye(self.num_sectors)

        # Solving the linear system (I - A)x = d
        # Using torch.linalg.solve is differentiable and more stable than torch.inverse.
        total_output = torch.linalg.solve(I - A, final_demand)
        return total_output


# --- Example Usage: Calibrating a 3-Sector Economy ---

# Mock Observed Data (e.g., Energy, Services, Manufacturing)
# Target Output (x) we want to explain
observed_x = torch.tensor([150.0, 200.0, 180.0])
# Observed Final Demand (d)
observed_d = torch.tensor([50.0, 80.0, 60.0])

model = DifferentiableIO(num_sectors=3)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("Calibrating Technical Coefficients Matrix...")

for epoch in range(1001):
    optimizer.zero_grad()

    # Predict total output based on current A and observed demand
    predicted_x = model(observed_d)

    # Loss is the difference between predicted and observed total output
    loss = criterion(predicted_x, observed_x)

    # Adding a regularization term to encourage sparsity or stay close to a prior
    # sparsity_loss = torch.norm(model.get_A(), 1) * 0.01
    # total_loss = loss + sparsity_loss

    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

# Final Calibrated Matrix
with torch.no_grad():
    calibrated_A = model.get_A()
    print("\nCalibrated Technical Coefficients (A):")
    print(calibrated_A)

    # Verification: Does (I-A)x = d?
    I = torch.eye(3)
    verification_d = (I - calibrated_A) @ observed_x
    print("\nVerification (Calculated Demand from Calibrated A):")
    print(verification_d)
    print("Observed Demand:")
    print(observed_d)
