# QuantIQ 🔍

**QuantIQ** is a unified Python library for building robust, uncertainty-aware deep learning systems. It brings together lightweight, modular tools to help researchers and practitioners gain insight into model reliability and risk through principled uncertainty estimation.

## 📦 What's Inside

Quantiq currently includes two powerful uncertainty quantification tools:

### 1. DropWise 🔁
A plug-and-play PyTorch/HuggingFace wrapper for Monte Carlo Dropout–based uncertainty estimation in Transformers.

- Supports classification, regression, QA, and token tagging
- Computes entropy, confidence, and class-wise variances
- Enables dropout during inference for Bayesian-style sampling

📖 [Full DropWise Documentation](https://github.com/aryanator/QuantIQ/blob/main/quantiq/dropwise/README.md)

---

### 2. SmartEnsemble 🧠
A deep ensemble wrapper for PyTorch models with support for adversarial training and dual-mode (epistemic + aleatoric) uncertainty estimation.

- Works with any PyTorch model
- Enables risk scoring and calibration
- Includes built-in visualization and prediction APIs

📖 [Full SmartEnsemble Documentation](https://github.com/aryanator/QuantIQ/blob/main/quantiq/smartensemble/README.md)

---

## 🔧 Installation

```bash
pip install quantiq
```

Or install from source:

```bash
git clone https://github.com/aryanator/QuantIQ.git
cd quantiq
pip install -e .
```

---

## 🧪 Use Cases

- Safety-critical predictions (medical AI, self-driving, finance)
- Uncertainty-aware active learning
- Robust ML pipelines with explainable confidence
- Research experiments involving confidence, entropy, risk

---

## 📚 Documentation & Examples

Explore examples, API usage, and task-specific walkthroughs in the GitHub repository:  
🔗 https://github.com/aryanator/QuantIQ

---

## 📝 License

MIT License

---

Built by [Aryan Patil](https://github.com/aryanator) to make uncertainty estimation simpler, smarter, and production-ready.
