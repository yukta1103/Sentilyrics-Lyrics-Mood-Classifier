import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

EMOTIONS = ['anger','disgust','fear','joy','sadness','surprise']

def predict_tfidf(model_path, texts):
    clf = joblib.load(model_path)
    preds = clf.predict(texts)
    return [[EMOTIONS[i] for i, v in enumerate(p) if v == 1] for p in preds]

def predict_bert(model_dir, texts):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    enc = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        probs = torch.sigmoid(model(**enc).logits).numpy()
    preds = (probs >= 0.5).astype(int)
    return [[EMOTIONS[i] for i, v in enumerate(p) if v == 1] for p in preds]