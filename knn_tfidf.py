import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score

from features import load_dev_test_texts_labels, build_tfidf_features, split_train_validation
from models import search_best_k, build_knn


def evaluate_on_test(metric_name: str, k: int, X_train, y_train, X_test, y_test) -> float:
    clf = build_knn(metric_name, k)
    clf.fit(X_train, y_train)
    test_pred = clf.predict(X_test)
    return accuracy_score(y_test, test_pred)


def main():
    print('加载数据...')
    X_dev_text, y_dev, X_test_text, y_test = load_dev_test_texts_labels()

    print('TF-IDF 特征提取...')
    # 取消 L2 归一化以观察欧氏距离与余弦的差异
    X_dev, X_test, _ = build_tfidf_features(X_dev_text, X_test_text, norm=None)

    X_train, X_val, y_train, y_val = split_train_validation(X_dev, y_dev)

    candidate_k = [1, 3, 5, 7, 9, 11, 15]

    print('\n在验证集上搜索最优 K（欧氏距离）...')
    best_k_euclidean, best_val_acc_euclidean = search_best_k(
        'euclidean', X_train, y_train, X_val, y_val, candidate_k
    )
    print(f"欧氏距离 最优K: {best_k_euclidean}, 验证集准确率: {best_val_acc_euclidean:.4f}")

    print('\n在验证集上搜索最优 K（余弦相似度）...')
    best_k_cosine, best_val_acc_cosine = search_best_k(
        'cosine', X_train, y_train, X_val, y_val, candidate_k
    )
    print(f"余弦相似度 最优K: {best_k_cosine}, 验证集准确率: {best_val_acc_cosine:.4f}")

    print('\n在测试集上评估...')
    test_acc_euclidean = evaluate_on_test(
        'euclidean', best_k_euclidean, X_train, y_train, X_test, y_test
    )
    test_acc_cosine = evaluate_on_test(
        'cosine', best_k_cosine, X_train, y_train, X_test, y_test
    )

    print('\n=== 测试集准确率对比（TF-IDF）===')
    print(f"欧氏距离-K={best_k_euclidean}: {test_acc_euclidean:.6f}")
    print(f"余弦相似度-K={best_k_cosine}: {test_acc_cosine:.6f}")

    better = '欧氏距离' if test_acc_euclidean >= test_acc_cosine else '余弦相似度'
    print(f"\n更优度量：{better}")


if __name__ == '__main__':
    main()


