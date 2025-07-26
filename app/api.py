from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

app = FastAPI()

class ReviewInput(BaseModel):
    review: str

# Load model and tokenizer once on startup
model_path = "./models/gpt-neo-sentiment"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPTNeoForCausalLM.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

@app.post("/predict")
def predict_sentiment(data: ReviewInput):
    review_text = data.review.strip()
    if not review_text:
        raise HTTPException(status_code=400, detail="Empty review")

    prompt = f"Review: {review_text}\nSentiment:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Model output: {decoded}")

    # Extract prediction string
    prediction = decoded.split("Sentiment:")[-1].strip().lower()

    # Keyword-based fallback matching
    positive_keywords = ["positive", "good", "great", "love", "amazing"]
    negative_keywords = ["negative", "bad", "terrible", "hate", "awful"]

    if any(word in prediction for word in positive_keywords):
        sentiment = "positive"
        confidence = 0.85
    elif any(word in prediction for word in negative_keywords):
        sentiment = "negative"
        confidence = 0.75
    else:
        sentiment = "unknown"
        confidence = 0.5

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "raw_model_output": prediction
    }