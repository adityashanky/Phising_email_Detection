import re
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

URL_PATTERN = re.compile(r"https?://|www\\.", re.IGNORECASE)
SUSPICIOUS_TERMS = [
    "urgent",
    "verify",
    "account",
    "password",
    "click",
    "update",
    "billing",
    "invoice",
    "confirm",
    "secure",
    "bank",
    "suspend",
    "login",
    "unauthorized",
    "alert",
    "payment",
    "refund",
    "verify",
    "limited",
    "expires",
]
META_FEATURE_NAMES = [
    "url_count",
    "exclamation_count",
    "digit_count",
    "uppercase_ratio",
    "suspicious_term_count",
]


def extract_meta_features(text: str):
    text = text or ""
    lower_text = text.lower()
    url_count = len(URL_PATTERN.findall(text))
    exclamation_count = text.count("!")
    digit_count = sum(char.isdigit() for char in text)
    uppercase_ratio = sum(char.isupper() for char in text) / max(len(text), 1)
    suspicious_term_count = sum(lower_text.count(term) for term in SUSPICIOUS_TERMS)
    return np.array(
        [url_count, exclamation_count, digit_count, uppercase_ratio, suspicious_term_count],
        dtype=float,
    )


class EmailFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words="english", ngram_range=(1, 2), max_features=5000):
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
        )

    def fit(self, X, y=None):
        texts = [str(text) for text in X]
        self.vectorizer.fit(texts)
        return self

    def transform(self, X):
        texts = [str(text) for text in X]
        tfidf_matrix = self.vectorizer.transform(texts)
        meta_matrix = csr_matrix([extract_meta_features(text) for text in texts])
        return hstack([tfidf_matrix, meta_matrix], format="csr")

    def get_feature_names_out(self):
        return np.concatenate([self.vectorizer.get_feature_names_out(), np.array(META_FEATURE_NAMES)])
