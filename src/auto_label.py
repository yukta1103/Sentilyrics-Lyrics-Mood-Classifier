# src/auto_label.py
import pandas as pd
import os
os.environ["USE_TF"] = "0"
from transformers import pipeline
import argparse
from tqdm import tqdm

# Mapping from model's labels to Ekman's 6 emotions
# Adjust this based on the model output
LABEL_MAP = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "sadness": "sadness",
    "surprise": "surprise"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CSV with 'lyrics' column")
    parser.add_argument("--output", default="data/labeled_emotions.csv")
    parser.add_argument("--max_rows", type=int, default=5000, help="Limit rows for testing")
    args = parser.parse_args()

    print("Loading dataset...")
    df = pd.read_csv(args.input)
    if args.max_rows:
        df = df.head(args.max_rows)

    print("Loading Hugging Face model...")
    clf = pipeline("text-classification",
                   model="j-hartmann/emotion-english-distilroberta-base",
                   return_all_scores=True,
                   truncation=True)

    def label_lyric(text):
        if not isinstance(text, str) or not text.strip():
            return ""
        scores = clf(text[:512])[0]
        selected = []
        for s in scores:
            label = LABEL_MAP.get(s["label"].lower())
            if label and s["score"] >= 0.5:
                selected.append(label)
        return "|".join(selected) if selected else LABEL_MAP.get(scores[0]["label"].lower(), "")

    print("Auto-labeling...")
    tqdm.pandas()
    df["labels"] = df["lyrics"].progress_apply(label_lyric)

    print(f"Saving to {args.output}...")
    df[["lyrics", "labels"]].to_csv(args.output, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
