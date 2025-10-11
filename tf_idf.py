import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def train_logistic_regression():
    # 加载预处理后的数据
    print("加载数据...")
    data = pd.read_csv('IMDB_Dataset_Preprocessed.csv')
    
    # Kaggle约束：前50%作为开发集，后50%作为测试集
    split_point = len(data) // 2
    dev_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    
    X_dev_text = dev_data['review'].values
    y_dev = dev_data['sentiment'].values
    X_test_text = test_data['review'].values
    y_test = test_data['sentiment'].values
    
    print("TF-IDF特征提取...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_dev = vectorizer.fit_transform(X_dev_text)
    X_test = vectorizer.transform(X_test_text)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.2, random_state=42, stratify=y_dev
    )
    
    print("训练逻辑回归模型...")
    lr = LogisticRegression(max_iter=500, random_state=42, solver='liblinear')
    lr.fit(X_train, y_train)
    
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