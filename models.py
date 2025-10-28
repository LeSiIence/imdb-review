from typing import List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


def train_logreg(X_train, y_train) -> LogisticRegression:
    clf = LogisticRegression(max_iter=500, random_state=42, solver='liblinear')
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(model, X, y) -> Tuple[float, float, float, float]:
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, pos_label='positive')
    recall = recall_score(y, y_pred, pos_label='positive')
    f1 = f1_score(y, y_pred, pos_label='positive')
    return accuracy, precision, recall, f1


def search_best_k(
    metric_name: str,
    X_train,
    y_train,
    X_val,
    y_val,
    candidate_k_list: List[int],
) -> Tuple[int, float]:
    if metric_name not in {'euclidean', 'cosine'}:
        raise ValueError("metric_name must be 'euclidean' or 'cosine'")

    best_k = None
    best_acc = -1.0

    for k in candidate_k_list:
        algorithm = 'brute' if metric_name == 'cosine' else 'auto'
        clf = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric_name,
            algorithm=algorithm,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val))
        if acc > best_acc or (acc == best_acc and (best_k is None or k < best_k)):
            best_acc = acc
            best_k = k

    return best_k, best_acc


def build_knn(metric_name: str, k: int) -> KNeighborsClassifier:
    algorithm = 'brute' if metric_name == 'cosine' else 'auto'
    return KNeighborsClassifier(
        n_neighbors=k,
        metric=metric_name,
        algorithm=algorithm,
        n_jobs=-1,
    )


