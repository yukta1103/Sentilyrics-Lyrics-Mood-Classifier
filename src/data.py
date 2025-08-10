import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import ast

EMOTIONS = ['anger','disgust','fear','joy','sadness','surprise']

class LyricsEmotionDataset(Dataset):
    """PyTorch dataset wrapper for tokenized inputs (for Transformers)."""
    def __init__(self, texts, labels, tokenizer=None, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        item = {}
        if self.tokenizer:
            enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length)
            item['input_ids'] = enc['input_ids']
            item['attention_mask'] = enc['attention_mask']
        item['labels'] = self.labels[idx]
        return item


def load_labeled_csv(path, label_col='labels', text_col='lyrics', test_size=0.1, random_state=42):
    df = pd.read_csv(path)
    df = df.dropna(subset=[text_col])
    # normalize labels: allow JSON list or pipe-separated strings
    def parse_label(x):
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return x
        try:
            if isinstance(x, str) and x.strip().startswith('['):
                return ast.literal_eval(x)
        except Exception:
            pass
        if isinstance(x, str):
            if '|' in x:
                return [i.strip() for i in x.split('|') if i.strip()]
            return [i.strip() for i in x.split(',') if i.strip()]
        return []

    df['labels_parsed'] = df[label_col].apply(parse_label)
    # binarize into EMOTIONS order
    def to_vector(lbls):
        out = [1 if e in lbls else 0 for e in EMOTIONS]
        return out
    df['label_vec'] = df['labels_parsed'].apply(to_vector)

    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test


