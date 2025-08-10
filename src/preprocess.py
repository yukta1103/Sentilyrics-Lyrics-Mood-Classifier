import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

RE_CLEAN = re.compile(r"[^\w\s']")


def clean_lyrics(text, remove_stopwords=False):
    if not isinstance(text, str):
        return ''
    text = text.strip().lower()
    text = RE_CLEAN.sub(' ', text)
    text = re.sub('\s+', ' ', text)
    if remove_stopwords:
        tokens = [t for t in text.split() if t not in STOPWORDS]
        return ' '.join(tokens)
    return text