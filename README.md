# 🎯 Sentiment Analysis Using BERT (Optimized for Event Feedback)

This project fine-tunes a pre-trained BERT model to classify college fest feedback into **Positive**, **Neutral**, or **Negative** sentiments. It uses Hugging Face Transformers, PyTorch, and Scikit-learn. The code is written in a clean and minimal way, with optimizations applied to reduce training time without sacrificing accuracy.


---

How the Code Works

Load the Dataset

The dataset contains event feedback with labels like `"Positive"`, `"Neutral"`, or `"Negative"`. These are mapped to integer values for training (0 = Negative, 1 = Neutral, 2 = Positive).

```python
train_df = pd.read_csv("college_fest_feedback_large_unique.csv")
test_df = pd.read_csv("college_fest_feedback_test_unique.csv")
```

---

Tokenization

We use the `bert-base-uncased` tokenizer to tokenize feedback comments. Padding and truncation ensure consistent length.

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
```

---

PyTorch Dataset

A custom PyTorch dataset class wraps the encoded inputs and labels, making them compatible with Hugging Face’s `Trainer`.

---

Load Pre-trained BERT

We use a pre-trained BERT model with a classification head configured for 3 classes.

```python
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
```

---

Training the Model

Training is done using Hugging Face’s `Trainer`. The training arguments were customized to speed up training:

```python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.1,
    logging_dir="./logs"
)
```

**Customizations Done:**
- **Batch size increased to 16** to reduce training steps per epoch.
- **Fewer epochs (3)** to prevent overfitting while reducing time.
- **Balanced weight decay (0.1)** to maintain regularization.
  
> These tweaks helped reduce training time **without hurting model accuracy**.

---

Evaluation

The model is evaluated using accuracy:

```python
accuracy = accuracy_score(test_labels, preds)
```

Example Output:
```
Test Accuracy: 0.8972
```

---

Save Model & Tokenizer

The trained model and tokenizer are saved using `joblib`:

```python
joblib.dump(model, "model.pkl")
joblib.dump(tokenizer, "tokenizer.pkl")
```

---

Predict New Feedback

You can load the model later and run sentiment predictions on new text:

```python
def predict_sentiment(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    prediction = torch.argmax(output.logits, dim=1).item()
    return ["Negative", "Neutral", "Positive"][prediction]
```

---

Requirements

Install the dependencies using:

```bash
pip install torch transformers datasets scikit-learn pandas joblib
```

---

Future Improvements

- Add web API using FastAPI or Flask.
- Streamlit-based interactive demo.
- Add multilingual support using mBERT.
- Model quantization for mobile usage.

---

Author

**Utkarsh Gabhne**  
Built and fine-tuned a custom BERT-based sentiment classifier with optimized training logic for speed and performance.
