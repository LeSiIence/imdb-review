import warnings
warnings.filterwarnings('ignore')

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

from features import load_full_texts_labels, build_tfidf_features


def stratified_subsample(texts: np.ndarray, labels: np.ndarray, max_points: int = 5000, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    if len(texts) <= max_points:
        return texts, labels

    rng = np.random.default_rng(random_state)
    unique_labels, counts = np.unique(labels, return_counts=True)
    desired_per_label = {lab: int(max_points * cnt / len(labels)) for lab, cnt in zip(unique_labels, counts)}

    idx_list = []
    for lab in unique_labels:
        lab_idx = np.where(labels == lab)[0]
        k = min(len(lab_idx), max(1, desired_per_label[lab]))
        idx_list.append(rng.choice(lab_idx, size=k, replace=False))

    indices = np.concatenate(idx_list)
    rng.shuffle(indices)
    return texts[indices], labels[indices]


def visualize_tfidf_with_pca_tsne(max_points: int = 5000, random_state: int = 42):
    print('加载全部文本与标签...')
    texts, labels = load_full_texts_labels()

    print(f'分层子采样到最多 {max_points} 个样本以便可视化...')
    texts_sub, labels_sub = stratified_subsample(texts, labels, max_points=max_points, random_state=random_state)

    print('TF-IDF 特征提取...')
    X_sub, _, _ = build_tfidf_features(texts_sub, texts_sub)

    print('PCA (TruncatedSVD) 到 2 维...')
    pca2 = TruncatedSVD(n_components=2, random_state=random_state)
    X_pca2 = pca2.fit_transform(X_sub)

    print('先用 SVD 降到 100 维，再做 t-SNE 到 2 维...')
    svd100 = TruncatedSVD(n_components=100, random_state=random_state)
    X_100 = svd100.fit_transform(X_sub)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init='pca',
        random_state=random_state,
        verbose=1,
        metric='cosine',
    )
    X_tsne2 = tsne.fit_transform(X_100)

    print('绘制散点图...')
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x=X_pca2[:, 0], y=X_pca2[:, 1], hue=labels_sub,
        palette={'negative': 'tab:blue', 'positive': 'tab:orange'},
        s=10, linewidth=0, alpha=0.7, legend='brief'
    )
    plt.title('TF-IDF + PCA (2D)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Sentiment', loc='best')

    plt.subplot(1, 2, 2)
    sns.scatterplot(
        x=X_tsne2[:, 0], y=X_tsne2[:, 1], hue=labels_sub,
        palette={'negative': 'tab:blue', 'positive': 'tab:orange'},
        s=10, linewidth=0, alpha=0.7, legend='brief'
    )
    plt.title('TF-IDF + t-SNE (2D)')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend(title='Sentiment', loc='best')

    plt.tight_layout()
    plt.savefig('tfidf_pca_tsne.png', dpi=200)
    plt.show()
    print('保存图像：tfidf_pca_tsne.png')

    # 最终版：仅保留固定参数的 t-SNE 结果


def main():
    visualize_tfidf_with_pca_tsne(max_points=5000, random_state=42)


if __name__ == '__main__':
    main()


