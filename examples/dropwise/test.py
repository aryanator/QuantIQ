# !pip install quantiq==0.1.1
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