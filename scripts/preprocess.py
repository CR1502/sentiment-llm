import pandas as pd
import re
import string
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

print("Reading training data..")
df = pd.read_csv("data/train.csv")

print("Cleaning reviews..")
df["review"] = df["review"].astype(str).apply(clean_text)

print("Splitting training data into train and val..")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment"])

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)

print("Preprocessing complete.")