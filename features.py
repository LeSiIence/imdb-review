import os
from typing import Tuple

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def load_full_texts_labels(csv_path: str = 'IMDB_Dataset_Preprocessed.csv') -> Tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(csv_path)
    return data['review'].astype(str).values, data['sentiment'].values


def load_dev_test_texts_labels(csv_path: str = 'IMDB_Dataset_Preprocessed.csv') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_csv(csv_path)
    split_point = len(data) // 2
    dev_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    X_dev_text = dev_data['review'].astype(str).values
    y_dev = dev_data['sentiment'].values
    X_test_text = test_data['review'].astype(str).values
    y_test = test_data['sentiment'].values
    return X_dev_text, y_dev, X_test_text, y_test


def split_train_validation(X_dev, y_dev, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X_dev, y_dev, test_size=test_size, random_state=random_state, stratify=y_dev
    )


def build_tfidf_features(
    X_dev_text,
    X_test_text,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    norm: str = 'l2',
):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, norm=norm)
    X_dev = vectorizer.fit_transform(X_dev_text)
    X_test = vectorizer.transform(X_test_text)
    return X_dev, X_test, vectorizer


def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return [t for t in text.strip().split() if t]


def train_word2vec(
    sentences,
    vector_size: int = 300,
    window: int = 5,
    min_count: int = 2,
    sg: int = 1,
    workers: int = None,
    seed: int = 42,
    epochs: int = 5,
) -> Word2Vec:
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers,
        seed=seed,
        epochs=epochs,
    )
    return model


def build_sentence_vectors(texts, model: Word2Vec, vector_size: int):
    vectors = []
    for txt in texts:
        tokens = tokenize(txt)
        word_vecs = [model.wv[w] for w in tokens if w in model.wv]
        if word_vecs:
            vec = np.mean(word_vecs, axis=0)
        else:
            vec = np.zeros(vector_size, dtype=np.float32)
        vectors.append(vec)
    return np.vstack(vectors)


