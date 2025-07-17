# DropWise

**DropWise** is a lightweight uncertainty estimation wrapper built into the [`quantiq`](https://pypi.org/project/quantiq) toolkit. It enables Monte Carlo Dropout–based predictive uncertainty in Transformers via a clean PyTorch/HuggingFace interface. DropWise supports classification, QA, token tagging, and regression with confidence-aware outputs.

---

## 🚀 Features

- ✅ Dropout-enabled **epistemic + aleatoric uncertainty** estimation during inference  
- ✅ Supports **entropy**, **confidence margin**, and **per-class std dev** out of the box  
- ✅ Plug-and-play for **Transformers** from Hugging Face  
- ✅ Task-adaptive output formatting: classification, QA, tagging, regression  
- ✅ Batch inference, GPU/CPU compatible, modular integration

---

## 🤖 Supported Tasks

| Task Type               | Example Model                                 |
|------------------------|------------------------------------------------|
| `sequence-classification` | `distilbert-base-uncased-finetuned-sst-2-english`  
| `token-classification`    | `dslim/bert-base-NER`  
| `question-answering`      | `deepset/bert-base-cased-squad2`  
| `regression`              | `roberta-base` (with regression head)

> ⚠️ Ensure your model has dropout layers for MC Dropout to work.

---

## 📦 Installation

```bash
pip install quantiq
```

---

## 🧠 Usage

### 📘 Sequence Classification

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from quantiq.dropwise import DropWise

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

predictor = DropWise(model, tokenizer, task_type="sequence-classification", num_passes=20)
results = predictor(["The movie was fantastic!"])
print(results[0])
```

```python
!pip install quantiq==0.1.1
from quantiq import DropWise
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

texts = ["The movie was fantastic!", "Awful experience.", "gbowbfk"]

predictor = DropWise(model, tokenizer, task_type="sequence-classification", num_passes=20)

for text in texts:
  result = predictor(text)
  print(result)
  if result[0]["entropy"] > 0.5:
      print(f"⚠️ Uncertain: {result}")
  else:
      print(f"✅ Confident: {result}")
```


[{'input': 'The movie was fantastic!', 'predicted_class': 1, 'entropy': 0.0015349582536146045, 'confidence': 0.999842643737793, 'confidence_margin': 0.9996853470802307, 'std_dev': [0.20731636881828308, 0.20824693143367767], 'probs': [0.00015731189341749996, 0.999842643737793]}]
✅ Confident: [{'input': 'The movie was fantastic!', 'predicted_class': 1, 'entropy': 0.0015349582536146045, 'confidence': 0.999842643737793, 'confidence_margin': 0.9996853470802307, 'std_dev': [0.20731636881828308, 0.20824693143367767], 'probs': [0.00015731189341749996, 0.999842643737793]}]
[{'input': 'Awful experience.', 'predicted_class': 0, 'entropy': 0.00342096877284348, 'confidence': 0.9996137619018555, 'confidence_margin': 0.9992276430130005, 'std_dev': [0.27654603123664856, 0.20306527614593506], 'probs': [0.9996137619018555, 0.00038614307413809]}]
✅ Confident: [{'input': 'Awful experience.', 'predicted_class': 0, 'entropy': 0.00342096877284348, 'confidence': 0.9996137619018555, 'confidence_margin': 0.9992276430130005, 'std_dev': [0.27654603123664856, 0.20306527614593506], 'probs': [0.9996137619018555, 0.00038614307413809]}]
[{'input': 'gbowbfk', 'predicted_class': 0, 'entropy': 0.638390302658081, 'confidence': 0.6639358401298523, 'confidence_margin': 0.3278716206550598, 'std_dev': [0.6975364089012146, 0.6409682035446167], 'probs': [0.6639358401298523, 0.3360642194747925]}]
⚠️ Uncertain: [{'input': 'gbowbfk', 'predicted_class': 0, 'entropy': 0.638390302658081, 'confidence': 0.6639358401298523, 'confidence_margin': 0.3278716206550598, 'std_dev': [0.6975364089012146, 0.6409682035446167], 'probs': [0.6639358401298523, 0.3360642194747925]}]



---

### 🏷️ Token Classification (NER)

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
from quantiq import DropWise

model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

predictor = DropWise(model, tokenizer, task_type="token-classification", num_passes=15)
results = predictor(["Hugging Face is based in New York City."])
print(results[0]['token_predictions'])
```

---

### ❓ Question Answering

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from quantiq import DropWise

model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")

qa_input = "Where is Hugging Face based? [SEP] Hugging Face Inc. is a company based in New York City."
predictor = DropWise(model, tokenizer, task_type="question-answering", num_passes=10)
results = predictor([qa_input])
print(results[0]['answer'])
```

---

### 📈 Regression

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from quantiq import DropWise

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

predictor = DropWise(model, tokenizer, task_type="regression", num_passes=20)
results = predictor(["The child is very young."])
print(results[0]['predicted_score'], "+/-", results[0]['uncertainty'])
```

---

## 📊 Output Dictionary (per sample)

| Field               | Description                                      |
|--------------------|--------------------------------------------------|
| `predicted_class`  | Index of most probable class (classification)    |
| `predicted_score`  | Scalar prediction (regression only)              |
| `confidence`       | Highest softmax probability                      |
| `entropy`          | Predictive entropy (higher = less confident)     |
| `std_dev`          | Std. dev. across MC dropout passes               |
| `probs`            | Per-class probabilities from averaged softmax    |
| `margin`           | Confidence gap between top-2 classes             |
| `answer`           | QA output span string                            |
| `token_predictions`| Token-level prediction+confidence list (NER)     |

---

## 🧪 Testing

```bash
python tests/test_predictor.py
```

---

## 📂 Structure

```
quantiq/
└── dropwise/
    ├── predictor.py
    ├── core.py
    └── tasks/
        ├── sequence_classification.py
        ├── token_classification.py
        ├── question_answering.py
        └── regression.py
```

---

## 📝 License

MIT License

---

Built with ❤️ for uncertainty-aware decision making in AI systems.