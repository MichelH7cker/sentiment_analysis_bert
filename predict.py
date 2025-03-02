from transformers import BertTokenizer, BertForSequenceClassification
import torch
import sys

# Load Model
modelo_path = "model_bert"
print("Loading model...")
model = BertForSequenceClassification.from_pretrained(modelo_path).eval()
tokenizer = BertTokenizer.from_pretrained(modelo_path)

# Function to Predict Sentiment
def predict_sentiment(text):
    tokens = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    output = model(**tokens)
    pred = torch.argmax(output.logits, axis=1).item()
    return "positive" if pred == 1 else "negative"

# Read input from command line
if len(sys.argv) > 1:
    text_input = " ".join(sys.argv[1:])
    print(f"Input text: {text_input}")
    print("Predicted Sentiment:", predict_sentiment(text_input))
else:
    print("Usage: python predict.py 'Your text here'")