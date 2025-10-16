import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return [t for t in text.strip().split() if t]


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


def train_word2vec_logreg():
    print("加载数据...")
    data = pd.read_csv('IMDB_Dataset_Preprocessed.csv')

    split_point = len(data) // 2
    dev_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]

    X_dev_text = dev_data['review'].values
    y_dev = dev_data['sentiment'].values
    X_test_text = test_data['review'].values
    y_test = test_data['sentiment'].values

    print("训练Word2Vec...")
    sentences = [tokenize(s) for s in data['review'].astype(str).values]
    vector_size = 300
    w2v = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=5,
        min_count=2,
        sg=1,
        workers=max(1, (os.cpu_count() or 2) - 1),
        seed=42,
        epochs=5,
    )

    print("构造句向量...")
    X_dev = build_sentence_vectors(X_dev_text, w2v, vector_size)
    X_test = build_sentence_vectors(X_test_text, w2v, vector_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.2, random_state=42, stratify=y_dev
    )

    print("训练逻辑回归模型...")
    lr = LogisticRegression(max_iter=500, random_state=42, solver='liblinear')
    lr.fit(X_train, y_train)

    test_pred = lr.predict(X_test)

    accuracy = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, pos_label='positive')
    recall = recall_score(y_test, test_pred, pos_label='positive')
    f1 = f1_score(y_test, test_pred, pos_label='positive')

    cm = confusion_matrix(y_test, test_pred, labels=['negative', 'positive'])

    print("\n=== 模型性能指标 ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\n=== 混淆矩阵 ===")
    print("               预测")
    print("实际    Negative  Positive")
    print(f"Negative   {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"Positive   {cm[1,0]:6d}    {cm[1,1]:6d}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.subplot(1, 2, 2)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1]
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{value:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    return lr, w2v


if __name__ == "__main__":
    model, w2v_model = train_word2vec_logreg()
    print("训练完成！")


