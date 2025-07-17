
# SmartEnsemble: Deep Ensemble for Uncertainty-Aware Learning

## Overview

`SmartEnsemble` is a PyTorch-based wrapper for training deep ensembles to quantify **epistemic** and **aleatoric** uncertainty. It's designed to work with any PyTorch model and supports:

- Regression and classification tasks
- FGSM-based adversarial training (optional)
- Epistemic and aleatoric uncertainty estimation
- Risk scoring and uncertainty calibration
- Visualization for regression models

---

## Usage Example

```python
from quantiq.smartensemble import SmartEnsemble

# Define base model
def model_fn():
    return MyRegressor()  # should return a nn.Module with (mean, logvar) for regression

ensemble = SmartEnsemble(model_fn=model_fn, num_models=5, task="regression")
ensemble.fit(train_loader, optimizer_fn=lambda m: torch.optim.Adam(m.parameters()), epochs=10)
mean, epistemic, aleatoric = ensemble.predict(x_test)
```

---

## Constructor

```python
SmartEnsemble(model_fn, num_models=5, task='regression', adv_train=False, epsilon=0.01)
```

### Parameters

- `model_fn (Callable[[], nn.Module])`: A function returning a fresh model instance.
- `num_models (int)`: Number of models in the ensemble.
- `task (str)`: `"regression"` or `"classification"`.
- `adv_train (bool)`: Enable adversarial training via FGSM.
- `epsilon (float)`: FGSM perturbation magnitude.

---

## Methods

### `fit(...)`

Trains all ensemble members.

**Key args**:
- `dataloader`: PyTorch `DataLoader`
- `optimizer_fn`: Function returning an optimizer
- `scheduler_fn`: Optional learning rate scheduler
- `clip_grad_norm`: Optional gradient clipping threshold
- `extra_forward_args`: Optional extra args passed to model
- `log_interval`: Logging frequency
- `device`: Training device

### `predict(x)`

Returns:
- **Regression**: `(mean, epistemic_std, aleatoric_std)`
- **Classification**: `(mean_probs, entropy_mean, entropy_aleatoric)`

### `calibrate_uncertainty_thresholds(dataloader)`

Uses validation data to set thresholds for classifying high vs low uncertainty.

### `score_risk(epistemic, aleatoric)`

Returns qualitative risk:
- 🟢 Low risk
- 🟡 Medium risk
- 🔴 High risk

### `visualize_uncertainty(...)`

For regression tasks — plots mean prediction, uncertainty bands, and risk levels.

---

## Notes

- Your model must return `(mean, logvar)` for regression.
- For classification, outputs should be logits.
- Supports adversarial robustness via FGSM.
- Use `.finetune()` as alias for `.fit()`.

---

## License

MIT
