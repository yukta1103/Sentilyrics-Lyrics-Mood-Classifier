import argparse
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from data import load_labeled_csv


def eval_tfidf(model_path, data_path):
    clf = joblib.load(model_path)
    df = pd.read_csv(data_path).dropna(subset=['lyrics'])
    X = df['lyrics'].astype(str).tolist()
    y_true = df['labels'].tolist()
    y_pred = clf.predict(X)
    print(classification_report(y_true, y_pred, target_names=['anger','disgust','fear','joy','sadness','surprise']))


def eval_bert(model_dir, data_path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    df = pd.read_csv(data_path).dropna(subset=['lyrics'])
  