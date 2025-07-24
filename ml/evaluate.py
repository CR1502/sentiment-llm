import pandas as pd
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tune model
model_path = "./models/gpt-neo-sentiment"
print(f"Loading model from {model_path}")
model = GPTNeoForCausalLM.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Load validation data
df = pd.read_csv("data/val.csv")

# Predict
preds = []
for prompt in df["prompt"]:
    inputs = tokenizer(prompt, return_tensor="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        top_k=50
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    prediction = decoded.split("Sentiment:")[-1].strip().lower()
    preds.append(prediction)

preds = [p if p in ["positive", "negative"] else "unknown" for p in preds]
df["predicted"] = preds

# Convert to binary
label_map = {"positive": 1, "negative": 0}
y_true = df["sentiment"].map(int)
y_pred = df["predicted"].map(label_map)

# Drop unknows if any
df = df[df["predicted"] != "unknown"]
y_true = df["sentiment"].map(int)
y_pred = df["predicted"].map(label_map)

# Print Metrics
print("\nEvaluation Results:")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))