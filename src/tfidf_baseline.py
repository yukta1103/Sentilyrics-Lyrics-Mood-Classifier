import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report


def train_tfidf(train_df, test_df, text_col='lyrics', label_col='label_vec', out_path=None):
    X_train = train_df[text_col].astype(str).tolist()
    X_test = test_df[text_col].astype(str).tolist()
    y_train = list(train_df[label_col])
    y_test = list(test_df[label_col])

    clf = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
        ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000)))
    ])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print('F1 (macro):', f1_score(y_test, preds, average='macro'))
    print(classification_report(y_test, preds, target_names=['anger','disgust','fear','joy','sadness','surprise']))
    if out_path:
        joblib.dump(clf, out_path)
    return clf