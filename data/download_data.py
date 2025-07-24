from datasets import load_dataset
import pandas as pd

# Loading Amazon Polarity dataset
print("Loading Dataset")
dataset = load_dataset("mteb/amazon_polarity")

# Converting train and test splits into pandas
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

# Renaming columns for better understanding
train_df = train_df[['text', 'label']].rename(columns={"text": "review", "label": "sentiment"})
test_df = test_df[['text', 'label']].rename(columns={"text": "review", "label": "sentiment"})

# Saving the data as CSV files
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Done, Saved the train and test datasets")