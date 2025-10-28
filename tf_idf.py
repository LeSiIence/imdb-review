import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from features import load_dev_test_texts_labels, build_tfidf_features, split_train_validation
from models import train_logreg
def train_logistic_regression():
    # 数据划分（前50%为dev，后50%为test）
    print("加载数据...")
    X_dev_text, y_dev, X_test_text, y_test = load_dev_test_texts_labels()
    
    print("TF-IDF特征提取...")
    X_dev, X_test, vectorizer = build_tfidf_features(X_dev_text, X_test_text)
    
    X_train, X_val, y_train, y_val = split_train_validation(X_dev, y_dev)
    
    print("训练逻辑回归模型...")
    lr = train_logreg(X_train, y_train)
    
    # 预测测试集
    test_pred = lr.predict(X_test)
    
    # 计算关键指标
    accuracy = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, pos_label='positive')
    recall = recall_score(y_test, test_pred, pos_label='positive')
    f1 = f1_score(y_test, test_pred, pos_label='positive')
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, test_pred, labels=['negative', 'positive'])
    
    # 显示结果
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
    
    #混淆矩阵
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    #子图2
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
    

    return lr, vectorizer

if __name__ == "__main__":
    model, vectorizer = train_logistic_regression()
    print("训练完成！")