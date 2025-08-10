import argparse
import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTETomek
from data import load_labeled_csv
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import Adam
from tqdm import tqdm

# ----------------------------
# Text cleaning for TF-IDF
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------
# Torch Dataset for BERT
# ----------------------------
class LyricsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ----------------------------
# Train BERT model
# ----------------------------
def train_bert(train_texts, train_labels, test_texts, test_labels, num_labels, epochs=3, batch_size=16, lr=2e-5):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = LyricsDataset(train_texts, train_labels, tokenizer, max_len=128)
    test_dataset = LyricsDataset(test_texts, test_labels, tokenizer, max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, axis=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print("\n Classification Report (BERT):")
    print(classification_report(true_labels, preds, zero_division=0))
    print(f"Macro F1: {f1_score(true_labels, preds, average='macro'):.4f}")
    print(f"Micro F1: {f1_score(true_labels, preds, average='micro'):.4f}")
    
    print("Confusion Matrix (BERT):")
    print(confusion_matrix(true_labels, preds))

    return model, tokenizer

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tfidf", "bert"], default="tfidf")
    parser.add_argument("--data", required=True, help="Path to labeled CSV")
    parser.add_argument("--out", default="model.pkl")
    parser.add_argument("--vectorizer_out", default="vectorizer.pkl")
    args = parser.parse_args()

    # Load dataset
    train_df, test_df = load_labeled_csv(args.data)

    if args.model == "tfidf":
        # Clean text
        train_df["lyrics"] = train_df["lyrics"].apply(clean_text)
        test_df["lyrics"] = test_df["lyrics"].apply(clean_text)

        # Vectorize with n-grams
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(train_df["lyrics"])
        X_test = vectorizer.transform(test_df["lyrics"])
        y_train = train_df["labels"]
        y_test = test_df["labels"]

        # Balance with SMOTETomek
        print("Applying SMOTETomek for class balancing...")
        smote_tomek = SMOTETomek(random_state=42)
        X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

        # Grid search for Logistic Regression
        param_grid = {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}
        grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, scoring="f1_macro", cv=3, n_jobs=-1)
        grid.fit(X_train_resampled, y_train_resampled)
        clf = grid.best_estimator_

        # Predictions
        y_pred = clf.predict(X_test)
        print("\nClassification Report (TF-IDF):")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
        print(f"Micro F1: {f1_score(y_test, y_pred, average='micro'):.4f}")
        
        print("Confusion Matrix (TF-IDF):")
        print(confusion_matrix(y_test, y_pred))

        # Save model & vectorizer
        joblib.dump(clf, args.out)
        joblib.dump(vectorizer, args.vectorizer_out)
        print(f" Model saved to {args.out}")
        print(f" Vectorizer saved to {args.vectorizer_out}")

    elif args.model == "bert":
        # Map labels to integers
        label_map = {label: idx for idx, label in enumerate(sorted(train_df["labels"].unique()))}
        train_labels = train_df["labels"].map(label_map).tolist()
        test_labels = test_df["labels"].map(label_map).tolist()

        model, tokenizer = train_bert(
            train_df["lyrics"].tolist(),
            train_labels,
            test_df["lyrics"].tolist(),
            test_labels,
            num_labels=len(label_map),
            epochs=3
        )

        # Save BERT model & tokenizer
        model.save_pretrained("bert_model")
        tokenizer.save_pretrained("bert_tokenizer")
        joblib.dump(label_map, "label_map.pkl")
        print(" BERT model & tokenizer saved.")
        
        

if __name__ == "__main__":
    main()
