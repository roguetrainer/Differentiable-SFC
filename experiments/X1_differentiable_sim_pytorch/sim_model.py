import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. THE DIFFERENTIABLE SFC ENGINE
class SIM_Model(nn.Module):
    def __init__(self, steps=100):
        super(SIM_Model, self).__init__()
        self.steps = steps

        # Behavioral Parameters (We can make these "Learnable" or fixed)
        self.alpha1 = nn.Parameter(torch.tensor(0.6))  # Propensity to consume (income)
        self.alpha2 = nn.Parameter(torch.tensor(0.4))  # Propensity to consume (wealth)

        # Policy Knob: The Tax Rate (theta)
        # We want to find the optimal tax rate to stabilize the system
        self.theta = nn.Parameter(torch.tensor(0.15))

        # Exogenous Variable: Government Spending
        self.G = torch.tensor(20.0)

    def forward(self, h_initial):
        h = h_initial
        y_history = []
        h_history = []

        for t in range(self.steps):
            # SFC Identities (SIM Model)
            # Y = (G + a2*H_prev) / (1 - a1*(1 - theta))
            # Note: We use the analytical solution of the simultaneous equations
            # to ensure the "accounting identity" holds at every step.
            denominator = 1 - self.alpha1 * (1 - self.theta)
            y = (self.G + self.alpha2 * h) / denominator

            t_tax = self.theta * y
            yd = y - t_tax
            c = self.alpha1 * yd + self.alpha2 * h

            # Update Stock: Delta H = YD - C
            h = h + (yd - c)

            y_history.append(y)
            h_history.append(h)

        return torch.stack(y_history), torch.stack(h_history)

# 2. DEFINE THE "HEALTH" OF THE ECONOMY (LOSS FUNCTION)
def economic_loss_function(y_history, h_history, target_gdp=100.0):
    # Goal 1: Reach a target GDP (Full employment proxy)
    gap_loss = torch.mean((y_history - target_gdp)**2)

    # Goal 2: Stability (Minimize Volatility/Oscillations)
    # Penalize the variance of the growth rate
    growth_rate = y_history[1:] / y_history[:-1] - 1
    volatility_loss = torch.var(growth_rate) * 1000

    # Goal 3: Financial Sustainability (Don't let wealth explode/collapse)
    wealth_drift = torch.abs(h_history[-1] - h_history[0])

    return gap_loss + volatility_loss + (0.1 * wealth_drift)

# 3. TRAINING / OPTIMIZATION LOOP
model = SIM_Model(steps=50)
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Starting Policy Optimization...")
for epoch in range(200):
    optimizer.zero_grad()

    # Initial Wealth
    h0 = torch.tensor(10.0)

    # Run differentiable simulation
    y_hist, h_hist = model(h0)

    # Calculate Loss
    loss = economic_loss_function(y_hist, h_hist)

    # BACKPROPAGATE THROUGH TIME (The "Magic" Step)
    loss.backward()

    # Update the Policy (theta) and Params
    optimizer.step()

    # Constraints: Keep theta between 0 and 1
    with torch.no_grad():
        model.theta.clamp_(0.01, 0.95)

    if epoch % 40 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Opt Tax Rate: {model.theta.item():.4f}")

# Final Results
y_final, h_final = model(torch.tensor(10.0))
plt.figure(figsize=(10, 4))
plt.plot(y_final.detach().numpy(), label="GDP (Y)")
plt.plot(h_final.detach().numpy(), label="Wealth (H)")
plt.title(f"Optimized SIM Model (Tax Rate: {model.theta.item():.2%})")
plt.legend()
plt.show()
