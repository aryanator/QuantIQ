import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Callable, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt

class SmartEnsemble(nn.Module):
    def __init__(self, model_fn: Callable[[], nn.Module], num_models: int = 5, 
                 task: str = "regression", adv_train: bool = False, epsilon: float = 0.01):
        super().__init__()
        assert task in {"regression", "classification"}
        self.models = nn.ModuleList([model_fn() for _ in range(num_models)])
        self.task = task
        self.adv_train = adv_train
        self.epsilon = epsilon
        self.num_models = num_models
        self.epi_thresh = None
        self.alea_thresh = None

    def _nll_regression_loss(self, mean, logvar, y):
        return (logvar + (y - mean)**2 / logvar.exp()).mean()

    def _fgsm_attack(self, model, x, y, loss_fn):
        x_adv = x.detach().clone().requires_grad_(True)
        output = model(x_adv)
        if self.task == "regression":
            mean, logvar = output
            loss = self._nll_regression_loss(mean, logvar, y)
        else:
            loss = loss_fn(output, y)
        loss.backward()
        return x + self.epsilon * x_adv.grad.sign()

    def fit(self,
            dataloader: DataLoader,
            optimizer_fn: Callable[[nn.Module], torch.optim.Optimizer],
            epochs: int = 10,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            scheduler_fn: Optional[Callable[[torch.optim.Optimizer], Any]] = None,
            clip_grad_norm: Optional[float] = None,
            log_interval: int = 10,
            extra_forward_args: Optional[Dict[str, Any]] = None,
            verbose: bool = True):

        extra_forward_args = extra_forward_args or {}

        for model_idx, model in enumerate(self.models):
            model.to(device)
            model.train()
            optimizer = optimizer_fn(model)
            scheduler = scheduler_fn(optimizer) if scheduler_fn else None

            for epoch in range(epochs):
                for batch_idx, (x, y) in enumerate(dataloader):
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()

                    if self.adv_train:
                        x_adv = self._fgsm_attack(model, x, y, loss_fn=nn.CrossEntropyLoss() if self.task == 'classification' else None)
                        x_aug = torch.cat([x, x_adv])
                        y_aug = torch.cat([y, y])
                    else:
                        x_aug, y_aug = x, y

                    output = model(x_aug, **extra_forward_args) if extra_forward_args else model(x_aug)

                    if self.task == "regression":
                        mean, logvar = output
                        loss = self._nll_regression_loss(mean, logvar, y_aug)
                    else:
                        loss = nn.CrossEntropyLoss()(output, y_aug)

                    loss.backward()
                    if clip_grad_norm:
                        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optimizer.step()
                    if scheduler: scheduler.step()

                    if verbose and batch_idx % log_interval == 0:
                        print(f"[Ensemble {model_idx}] Epoch {epoch + 1}/{epochs} Batch {batch_idx} Loss: {loss.item():.4f}")

    def finetune(self, *args, **kwargs):
        self.fit(*args, **kwargs)

    def predict(self, x: torch.Tensor, device: str = "cuda" if torch.cuda.is_available() else "cpu"
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(device)
        all_outputs = []

        for model in self.models:
            model.eval()
            model.to(device)
            with torch.no_grad():
                output = model(x)
                all_outputs.append(output)

        if self.task == "regression":
            means = torch.stack([o[0] for o in all_outputs])
            logvars = torch.stack([o[1] for o in all_outputs])

            mean_pred = means.mean(dim=0)
            epistemic_var = means.var(dim=0)
            aleatoric_var = logvars.exp().mean(dim=0)

            return mean_pred, epistemic_var.sqrt(), aleatoric_var.sqrt()

        else:
            probs = torch.stack([F.softmax(o, dim=-1) for o in all_outputs])
            mean_probs = probs.mean(dim=0)

            entropy_mean = -torch.sum(mean_probs * (mean_probs + 1e-8).log(), dim=-1)
            entropies = -torch.sum(probs * (probs + 1e-8).log(), dim=-1)
            entropy_aleatoric = entropies.mean(dim=0)

            return mean_probs, entropy_mean, entropy_aleatoric

    def calibrate_uncertainty_thresholds(self, dataloader: DataLoader, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        epi_vals, alea_vals = [], []
        for x, _ in dataloader:
            _, epi, alea = self.predict(x.to(device))
            epi_vals.append(epi.detach().cpu())
            alea_vals.append(alea.detach().cpu())
        epi_all = torch.cat(epi_vals)
        alea_all = torch.cat(alea_vals)
        self.epi_thresh = epi_all.mean() + epi_all.std()
        self.alea_thresh = alea_all.mean() + alea_all.std()

    def score_risk(self, epistemic: torch.Tensor, aleatoric: torch.Tensor) -> str:
        if self.epi_thresh is None or self.alea_thresh is None:
            raise ValueError("Call calibrate_uncertainty_thresholds() before scoring risk.")

        # Step 1: Normalize and cap the scores
        epi_score = (epistemic / self.epi_thresh).clamp(0, 2)
        alea_score = (aleatoric / self.alea_thresh).clamp(0, 2)

        # Step 2: Mean across output dims to get one scalar per sample
        risk_score = (epi_score + alea_score).mean()  # scalar

        # Step 3: Risk label
        if risk_score < 1.0:
            return "🟢 Low risk"
        elif risk_score < 2.5:
            return "🟡 Medium risk"
        else:
            return "🔴 High risk"


    def visualize_uncertainty(self, input_range=(-5, 5), num_points=200, device="cuda" if torch.cuda.is_available() else "cpu"):
        x_vals = torch.linspace(*input_range, num_points).unsqueeze(1).to(device)
        mean, epi, alea = self.predict(x_vals)
        risk_labels = [self.score_risk(e, a) for e, a in zip(epi, alea)]

        x_vals = x_vals.cpu().squeeze().numpy()
        mean = mean.cpu().squeeze().numpy()
        epi = epi.cpu().squeeze().numpy()
        alea = alea.cpu().squeeze().numpy()

        color_map = {"🟢 Low risk": "green", "🟡 Medium risk": "orange", "🔴 High risk": "red"}
        risk_colors = [color_map[label] for label in risk_labels]

        plt.figure(figsize=(12, 6))
        plt.title("Prediction with Epistemic & Aleatoric Uncertainty and Risk Levels")
        plt.plot(x_vals, mean, label="Mean Prediction", color="blue")
        plt.fill_between(x_vals, mean - epi, mean + epi, alpha=0.2, label="Epistemic Uncertainty", color="purple")
        plt.fill_between(x_vals, mean - alea, mean + alea, alpha=0.2, label="Aleatoric Uncertainty", color="gold")
        plt.scatter(x_vals, mean, c=risk_colors, s=10, alpha=0.7, label="Risk Level")
        plt.xlabel("Input")
        plt.ylabel("Prediction / Uncertainty")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()