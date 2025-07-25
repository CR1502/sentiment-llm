from tqdm import tqdm
import pandas as pd
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned model
model_path = "./models/gpt-neo-sentiment"
print(f"Loading model from {model_path}")
model = GPTNeoForCausalLM.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Load validation data
df = pd.read_csv("data/val.csv")

# Create prompt column
df["prompt"] = df["review"].apply(lambda r: f"Review: {r}\nSentiment:")

# Predict
preds = []
for prompt in tqdm(df["prompt"], desc="Generating predictions"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    prediction = decoded.split("Sentiment:")[-1].strip().lower()
    preds.append(prediction)

# Clean up predictions
preds = [p if p in ["positive", "negative"] else "unknown" for p in preds]
df["predicted"] = preds

# Drop unknowns if any
df = df[df["predicted"] != "unknown"]

# Convert to binary labels
label_map = {"positive": 1, "negative": 0}
y_true = df["sentiment"].map(int)
y_pred = df["predicted"].map(label_map)

# Print metrics
print("\nEvaluation Results:")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))

"""
Evaluation Results:
Accuracy: 0.735357917570499
F1 Score: 0.7836879432624113

Classification Report:
              precision    recall  f1-score   support

    negative       0.83      0.55      0.66       215
    positive       0.69      0.90      0.78       246

    accuracy                           0.74       461
   macro avg       0.76      0.72      0.72       461
weighted avg       0.76      0.74      0.73       461
"""