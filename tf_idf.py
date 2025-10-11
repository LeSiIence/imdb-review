import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import IMDBPreprocessor
import warnings
warnings.filterwarnings('ignore')

class IMDBTfIdfProcessor:
    def __init__(self, preprocessed_csv_path):
        self.csv_path = preprocessed_csv_path
        self.data = None
        self.vectorizer = None

        self.X_dev = None
        self.y_dev = None
        self.X_test = None
        self.y_test = None

        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        
    def load_data(self):
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"加载成功，共有 {len(self.data)} 条记录")
            print(f"列名: {list(self.data.columns)}")

            if 'sentiment' in self.data.columns:
                print(f"情感标签分布:\n{self.data['sentiment'].value_counts()}")
            
            return True
        except Exception as e:
            print(f"fuck: 加载失败: {e}")
            return False
    
    def kaggle_split_data(self, text_column='review', label_column='sentiment'):
        if self.data is None:
            print("数据为空")
            return False
        
        print("\n=== Kaggle约束数据分割 ===")
        total_samples = len(self.data)
        split_point = total_samples // 2
        
        print(f"总样本数: {total_samples}")
        print(f"前50%: {split_point} 条")
        print(f"后50%: {total_samples - split_point} 条")
        
        dev_data = self.data.iloc[:split_point].copy()
        test_data = self.data.iloc[split_point:].copy()
        
        #提取文本和标签
        X_dev_text = dev_data[text_column].values
        self.y_dev = dev_data[label_column].values
        X_test_text = test_data[text_column].values
        self.y_test = test_data[label_column].values
        
        print(f"\n开发集情感分布:")
        dev_sentiment_counts = pd.Series(self.y_dev).value_counts()
        print(dev_sentiment_counts)
        
        print(f"\n测试集情感分布:")
        test_sentiment_counts = pd.Series(self.y_test).value_counts()
        print(test_sentiment_counts)
        
        #创建TF-IDF vectorizer 只在前50%数据上拟合
        print("\n=== TF-IDF特征提取 ===")
        
        self.vectorizer = TfidfVectorizer(
            max_features=10000,      # 最大特征数
            ngram_range=(1, 2),      # 1-gram和2-gram
            min_df=2,                # 忽略出现次数少于2的词
            max_df=0.95,             # 忽略出现频率超过95%的词
            lowercase=True,          # 转换为小写
            strip_accents='ascii'    # 移除重音符号
        )

        print("在前50%数据上拟合TF-IDF向量化器...")
        self.vectorizer.fit(X_dev_text)

        print("转换开发集为TF-IDF特征...")
        self.X_dev = self.vectorizer.transform(X_dev_text)
        
        print("转换测试集为TF-IDF特征...")
        self.X_test = self.vectorizer.transform(X_test_text)
        
        print(f"\n特征矩阵信息:")
        print(f"开发集特征矩阵形状: {self.X_dev.shape}")
        print(f"测试集特征矩阵形状: {self.X_test.shape}")
        print(f"词汇表大小: {len(self.vectorizer.vocabulary_)}")
        
        return True
    
    def split_dev_data(self, test_size=0.2, random_state=42):
        if self.X_dev is None or self.y_dev is None:
            print("先执行Kaggle数据分割！")
            return False
        
        print(f"\n=== 开发集二次分割 ===")
        print(f"前50%数据分割 训练集:{1-test_size:.0%} / 验证集:{test_size:.0%}")
        
        #分层采样，保持类别平衡
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_dev, self.y_dev,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y_dev  #保持类别比例
        )
        
        print(f"训练集形状: {self.X_train.shape}")
        print(f"验证集形状: {self.X_val.shape}")
        
        print(f"\n训练集标签分布:")
        train_counts = pd.Series(self.y_train).value_counts()
        print(train_counts)
        
        print(f"\n验证集标签分布:")
        val_counts = pd.Series(self.y_val).value_counts()
        print(val_counts)
        
        return True
    
    def get_feature_names(self, top_n=20):
        """获取最重要的特征名称"""
        if self.vectorizer is None:
            print("先进行TF-IDF转换！")
            return None
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        #计算每个特征的平均TF-IDF分数
        mean_scores = np.mean(self.X_dev.toarray(), axis=0)
        
        #获取top特征
        top_indices = np.argsort(mean_scores)[-top_n:][::-1]
        top_features = [(feature_names[i], mean_scores[i]) for i in top_indices]
        
        print(f"\nTop {top_n} TF-IDF特征:")
        for i, (feature, score) in enumerate(top_features, 1):
            print(f"{i:2d}. {feature:15s} (分数: {score:.4f})")
        
        return top_features
    
    def analyze_tfidf_stats(self):
        if self.X_dev is None:
            print("先进行TF-IDF转换！")
            return
        
        print("\n=== TF-IDF矩阵统计分析 ===")
        
        #稀疏性
        dev_density = self.X_dev.nnz / (self.X_dev.shape[0] * self.X_dev.shape[1])
        test_density = self.X_test.nnz / (self.X_test.shape[0] * self.X_test.shape[1])        
        print(f"开发集矩阵稀疏性: {(1-dev_density)*100:.2f}% (密度: {dev_density*100:.2f}%)")
        print(f"测试集矩阵稀疏性: {(1-test_density)*100:.2f}% (密度: {test_density*100:.2f}%)")
        
        #TF-IDF分数
        dev_scores = self.X_dev.data
        test_scores = self.X_test.data        
        print(f"\n开发集TF-IDF分数统计:")
        print(f"  平均值: {np.mean(dev_scores):.4f}")
        print(f"  标准差: {np.std(dev_scores):.4f}")
        print(f"  最大值: {np.max(dev_scores):.4f}")
        print(f"  最小值: {np.min(dev_scores):.4f}")        
        print(f"\n测试集TF-IDF分数统计:")
        print(f"  平均值: {np.mean(test_scores):.4f}")
        print(f"  标准差: {np.std(test_scores):.4f}")
        print(f"  最大值: {np.max(test_scores):.4f}")
        print(f"  最小值: {np.min(test_scores):.4f}")
    
    def save_features(self, output_dir="features"):
        import pickle
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        np.savez_compressed(
            os.path.join(output_dir, 'tfidf_features.npz'),
            X_train=self.X_train.toarray() if self.X_train is not None else None,
            X_val=self.X_val.toarray() if self.X_val is not None else None,
            X_test=self.X_test.toarray(),
            y_train=self.y_train,
            y_val=self.y_val,
            y_test=self.y_test
        )
        
        with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"特征和vectorizer已保存到 {output_dir} 目录")

def main():
    preprocessed_file = "IMDB_Dataset_Preprocessed.csv"

    print(f"寻找预处理文件: {preprocessed_file}")

    import os
    if not os.path.exists(preprocessed_file):
        preprocessor = IMDBPreprocessor("IMDB Dataset.csv")
        if preprocessor.load_data():
            preprocessor.preprocess_text_column('review', remove_stopwords_flag=False)
            preprocessor.save_processed_data(preprocessed_file)
        else:
            return
    
    tfidf_processor = IMDBTfIdfProcessor(preprocessed_file)
    
    if not tfidf_processor.load_data():
        return
    
    if not tfidf_processor.kaggle_split_data():
        return

    if not tfidf_processor.split_dev_data():
        return

    tfidf_processor.get_feature_names(top_n=30)
    tfidf_processor.analyze_tfidf_stats()

    tfidf_processor.save_features()
    
    print("\n=== TF-IDF特征提取完成 ===")
    print(f"训练集: {tfidf_processor.X_train.shape}")
    print(f"验证集: {tfidf_processor.X_val.shape}")
    print(f"测试集: {tfidf_processor.X_test.shape}")



if __name__ == "__main__":
    main()