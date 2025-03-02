import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load Dataset
df = pd.read_csv("dataset/training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1", header=None)
df.columns = ["sentiment", "id", "data", "query", "user", "text"]
df = df[["sentiment", "text"]]
df["sentiment"] = df["sentiment"].map({0: 0, 4: 1})

df.head()

# Tokenizer
print("Tokenizing texts...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer(list(df["text"].values), padding=True, truncation=True, max_length=128, return_tensors="pt")

# Dataset Class
class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Prepare Data
dataset = TweetDataset(tokens, df["sentiment"].values)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Load Pretrained BERT Model
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(DEVICE)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
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