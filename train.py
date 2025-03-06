import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Free GPU memory before training
torch.cuda.empty_cache()

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load Dataset (Using a smaller subset for testing)
df = pd.read_csv("dataset/training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1", header=None)
df.columns = ["sentiment", "id", "data", "query", "user", "text"]
df = df[["sentiment", "text"]]
df["sentiment"] = df["sentiment"].map({0: 0, 4: 1})

# Reduce dataset for testing (Optional)
df = df.sample(n=50000, random_state=42)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset Class
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt")
        self.labels = torch.tensor(labels.tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# Split Data
train_texts, test_texts, train_labels, test_labels = train_test_split(df["text"], df["sentiment"], test_size=0.2, random_state=42)
train_dataset = TweetDataset(train_texts, train_labels, tokenizer)
test_dataset = TweetDataset(test_texts, test_labels, tokenizer)

# Load Pretrained BERT Model
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(DEVICE)

# Define compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = torch.argmax(torch.tensor(pred.predictions), axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    fp16=True,  # Enable mixed precision
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,  # Log less often
    load_best_model_at_end=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics  # Auto-evaluation
)

# Train Model
print("Starting training...")
trainer.train()

# Save Model
print("Saving the trained model...")
output_dir = "model_bert_classification"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved in {output_dir}")

# Evaluate Model
print("Evaluating model...")
preds = trainer.predict(test_dataset)
y_pred = torch.argmax(torch.tensor(preds.predictions), axis=1)
y_true = torch.tensor(preds.label_ids)

accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)