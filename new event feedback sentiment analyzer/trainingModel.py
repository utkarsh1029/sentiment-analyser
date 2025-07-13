import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# load the dataset
train_df = pd.read_csv("your csv")
test_df = pd.read_csv("your csv")

# map sentiments to numbers
sentiment_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
train_df["True_Sentiment"] = train_df["True_Sentiment"].map(sentiment_mapping).astype(int)
test_df["True_Sentiment"] = test_df["True_Sentiment"].map(sentiment_mapping).astype(int)

# extract text and labels
train_texts = train_df['Feedback_Comment'].tolist()
test_texts = test_df['Feedback_Comment'].tolist()
train_labels = train_df['True_Sentiment'].tolist()
test_labels = test_df['True_Sentiment'].tolist()

# load tokenizer and tokenize the text
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256)

train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)

# create torch dataset
class EventDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = EventDataset(train_encodings, train_labels)
test_dataset = EventDataset(test_encodings, test_labels)

# load the BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.1,
    logging_dir="./logs",
)

# create trainer and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# evaluate model accuracy
preds_output = trainer.predict(test_dataset)
preds = np.argmax(preds_output.predictions, axis=1)
accuracy = accuracy_score(test_labels, preds)
print(f"Test Accuracy: {accuracy:.4f}")

# save the model and tokenizer
joblib.dump(model, "model (2).pkl")
joblib.dump(tokenizer, "tokenizer (2).pkl")
print("Model and tokenizer saved successfully!")

# load them again to test
def load_model():
    loaded_model = joblib.load("model (2).pkl")
    loaded_tokenizer = joblib.load("tokenizer (2).pkl")
    return loaded_model, loaded_tokenizer

model, tokenizer = load_model()
print("Model and tokenizer loaded successfully!")

# function to predict sentiment from new text
def predict_sentiment(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)

    with torch.no_grad():
        output = model(**inputs)
    prediction = torch.argmax(output.logits, dim=1).item()

    if prediction == 0:
        return "Negative"
    elif prediction == 1:
        return "Neutral"
    else:
        return "Positive"

# try it out
sample_text = "The event was amazing and well-organized!"
print(f"Feedback: {sample_text} → Sentiment: {predict_sentiment(sample_text)}")