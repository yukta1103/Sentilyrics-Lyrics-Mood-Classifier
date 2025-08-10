import argparse
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from train import LyricsDataset
from torch.utils.data import DataLoader

def predict_tfidf(model_path, vectorizer_path, texts):
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    X = vectorizer.transform(texts)
    preds = clf.predict(X)
    return preds

def predict_bert(model_dir, texts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForSequenceClassification.from_pretrained(model_dir)
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    dataset = LyricsDataset(texts, [0]*len(texts), tokenizer, max_len=128)  # dummy labels
    loader = DataLoader(dataset, batch_size=16)

    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, axis=1).cpu().numpy()
            preds.extend(batch_preds)

    return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["tfidf", "bert"], required=True)
    parser.add_argument("--model_path", required=True, help="For tfidf: model file; for bert: model directory")
    parser.add_argument("--vectorizer_path", help="Required if model_type=tfidf")
    parser.add_argument("--input_file", required=True, help="Path to text file with one input text per line")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    if args.model_type == "tfidf":
        if not args.vectorizer_path:
            raise ValueError("vectorizer_path is required for tfidf model")
        preds = predict_tfidf(args.model_path, args.vectorizer_path, texts)

    elif args.model_type == "bert":
        preds = predict_bert(args.model_path, texts)

    print("Predictions:")
    for text, pred in zip(texts, preds):
        print(f"[{pred}] {text}")
