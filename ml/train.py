# ml/train.py

import os
import pandas as pd
from datasets import Dataset
from transformers import (
    GPTNeoForCausalLM,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Load preprocessed datasets
train_df = pd.read_csv("data/train.csv").sample(500, random_state=42)
val_df = pd.read_csv("data/val.csv").sample(500, random_state=42)

# Format the input as instruction-style prompts
def format_prompt(example):
    return f"Review: {example['review']}\nSentiment:"

train_df["text"] = train_df.apply(format_prompt, axis=1)
val_df["text"] = val_df.apply(format_prompt, axis=1)

# Convert to Hugging Face Datasets
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

# Load tokenizer & model
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Important for GPT-Neo

# Tokenize input text
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Load the model
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./models/gpt-neo-sentiment",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    report_to="tensorboard"
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-style models predict next token, not masked tokens
)

# Initialize Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator
)

# Train!
trainer.train()

# Save model and tokenizer
trainer.save_model("./models/gpt-neo-sentiment")
tokenizer.save_pretrained("./models/gpt-neo-sentiment")

print("Training complete and model saved to ./models/gpt-neo-sentiment")