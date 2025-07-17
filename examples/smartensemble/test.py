import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from quantiq import SmartEnsemble

# Define a toy regression model that returns (mean, logvar)
class ToyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)  # Output: mean and logvar
        )

    def forward(self, x):
        out = self.base(x)
        mean = out[:, 0:1]
        logvar = out[:, 1:2]
        return mean, logvar

# Generate synthetic 1D regression data: y = 2x + noise
torch.manual_seed(0)
X = torch.linspace(-5, 5, 200).unsqueeze(1)
y = 2 * X + torch.randn_like(X) * 2

# Wrap in DataLoader
train_loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

# Define the ensemble
ensemble = SmartEnsemble(model_fn=ToyRegressor, num_models=3, task="regression")

# Train
ensemble.fit(
    dataloader=train_loader,
    optimizer_fn=lambda m: torch.optim.Adam(m.parameters(), lr=0.01),
    epochs=5,
    log_interval=1,
    clip_grad_norm=1.0
)

# Predict on new points
test_x = torch.tensor([[-2.0], [0.0], [2.0], [10.0]])  # Last one is OOD
mean, epi, alea = ensemble.predict(test_x)

print("Predictions:")
for i, x in enumerate(test_x):
    print(f"x = {x.item():5.2f} | mean = {mean[i].item():.2f} ± (epi={epi[i].item():.2f}, alea={alea[i].item():.2f})")

# Calibrate thresholds and test risk scoring
ensemble.calibrate_uncertainty_thresholds(train_loader)
for i in range(len(test_x)):
    label = ensemble.score_risk(epi[i], alea[i])
    print(f"Risk label for x = {test_x[i].item():5.2f}: {label}")

# Optional: visualize
ensemble.visualize_uncertainty()
