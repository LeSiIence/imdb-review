import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SentimentClassifier:
    def __init__(self, features_path="features/tfidf_features.npz", 
                 vectorizer_path="features/tfidf_vectorizer.pkl"):
        """
        情感分类器
        Args:
            features_path: 特征文件路径
            vectorizer_path: 向量化器文件路径
        """
        self.features_path = features_path
        self.vectorizer_path = vectorizer_path
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        
        # 数据
        self.X_train = None
        self.X_val = None  
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # 模型
        self.models = {}
        self.results = {}
    
    def load_data(self):
        """加载保存的特征和向量化器"""
        print("=== 加载保存的数据 ===")
        
        # 1. 加载特征矩阵
        try:
            data = np.load(self.features_path, allow_pickle=True)
            self.X_train = data['X_train']
            self.X_val = data['X_val'] 
            self.X_test = data['X_test']
            self.y_train = data['y_train']
            self.y_val = data['y_val']
            self.y_test = data['y_test']
            data.close()
            
            print(f"✅ 特征矩阵加载成功")
            print(f"   训练集: {self.X_train.shape}")
            print(f"   验证集: {self.X_val.shape}")  
            print(f"   测试集: {self.X_test.shape}")
            
        except Exception as e:
            print(f"❌ 加载特征矩阵失败: {e}")
            return False
        
        # 2. 加载向量化器
        try:
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print(f"✅ 向量化器加载成功")
            
        except Exception as e:
            print(f"❌ 加载向量化器失败: {e}")
            return False
        
        # 3. 编码标签 (string -> numeric)
        all_labels = np.concatenate([self.y_train, self.y_val, self.y_test])
        self.label_encoder.fit(all_labels)
        
        self.y_train_encoded = self.label_encoder.transform(self.y_train)
        self.y_val_encoded = self.label_encoder.transform(self.y_val)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)
        
        print(f"✅ 标签编码完成: {self.label_encoder.classes_}")
        
        return True
    
    def train_models(self):
        """训练多个机器学习模型"""
        print("\n=== 训练机器学习模型 ===")
        
        # 只使用逻辑回归模型
        model_configs = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        print(f"📊 数据规模: 训练集 {self.X_train.shape[0]:,} 条，特征维度 {self.X_train.shape[1]:,}")
        print(f"🕐 预计训练时间: SVM可能需要几分钟，请耐心等待...")
        
        for i, (name, model) in enumerate(model_configs.items(), 1):
            print(f"\n{'='*50}")
            print(f"🚀 [{i}/{len(model_configs)}] 训练 {name}")
            print(f"{'='*50}")
            
            # 显示逻辑回归参数
            print("⚙️  逻辑回归参数:")
            print(f"   - 最大迭代次数: {model.max_iter}")
            print(f"   - 随机种子: {model.random_state}")
            print(f"   - 正则化参数 C: {model.C}")
                
            start_time = time.time()
            
            try:
                # 训练模型
                print(f"⏰ 开始时间: {time.strftime('%H:%M:%S')}")
                print("🔄 逻辑回归训练中...")
                
                model.fit(self.X_train, self.y_train_encoded)
                
                training_time = time.time() - start_time
                print(f"✅ 训练完成! 耗时: {training_time:.2f}秒")
                
                self.models[name] = model
                
                # 在验证集上评估
                print("📝 验证集评估中...")
                val_start = time.time()
                val_pred = model.predict(self.X_val)
                val_time = time.time() - val_start
                val_accuracy = accuracy_score(self.y_val_encoded, val_pred)
                
                print(f"🎯 验证集准确率: {val_accuracy:.4f}")
                print(f"⚡ 预测耗时: {val_time:.2f}秒")
                print(f"📊 输入特征数: {model.n_features_in_:,}")
                
            except Exception as e:
                training_time = time.time() - start_time
                print(f"❌ 训练失败 (耗时 {training_time:.2f}秒): {e}")
        
        print(f"\n🏁 所有模型训练完成! 成功训练 {len(self.models)} 个模型")
    
    def evaluate_on_test_set(self):
        """在测试集上评估模型，详细展示混淆矩阵分析"""
        print("\n=== 测试集评估结果 ===")
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"📊 {name} 详细评估结果")
            print(f"{'='*60}")
            
            # 预测
            y_pred = model.predict(self.X_test)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)
            
            # 计算混淆矩阵
            cm = confusion_matrix(self.y_test_encoded, y_pred)
            
            # 获取标签映射 (0=negative, 1=positive)
            labels = self.label_encoder.classes_
            label_to_idx = {label: idx for idx, label in enumerate(labels)}
            
            # 提取混淆矩阵的四个值
            # 假设 0=negative, 1=positive
            if len(cm) == 2:
                tn, fp, fn, tp = cm.ravel()
                
                print(f"🔍 混淆矩阵分析:")
                print(f"{'='*40}")
                print(f"                预测结果")
                print(f"实际     Negative  Positive")
                print(f"Negative    {tn:4d}     {fp:4d}    (TN=真阴性, FP=假阳性)")
                print(f"Positive    {fn:4d}     {tp:4d}    (FN=假阴性, TP=真阳性)")
                print(f"{'='*40}")
                
                print(f"\n📈 混淆矩阵详细解释:")
                print(f"• TN (真阴性): {tn:,} - 正确预测为 Negative 的样本")
                print(f"• TP (真阳性): {tp:,} - 正确预测为 Positive 的样本") 
                print(f"• FN (假阴性): {fn:,} - 错误预测为 Negative 的样本 (实际是 Positive)")
                print(f"• FP (假阳性): {fp:,} - 错误预测为 Positive 的样本 (实际是 Negative)")
                
                # 计算各类别的精确率、召回率、F1
                print(f"\n🎯 分类别指标:")
                print(f"{'='*50}")
                
                # Negative类别指标
                neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
                neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0  
                neg_f1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0
                
                # Positive类别指标  
                pos_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                pos_f1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall) if (pos_precision + pos_recall) > 0 else 0
                
                print(f"Negative 类别:")
                print(f"  精确率 (Precision): {neg_precision:.4f}")
                print(f"  召回率 (Recall):    {neg_recall:.4f}")
                print(f"  F1分数:            {neg_f1:.4f}")
                
                print(f"\nPositive 类别:")
                print(f"  精确率 (Precision): {pos_precision:.4f}")
                print(f"  召回率 (Recall):    {pos_recall:.4f}")
                print(f"  F1分数:            {pos_f1:.4f}")
            
            # 计算整体指标
            accuracy = accuracy_score(self.y_test_encoded, y_pred)
            precision_macro = precision_score(self.y_test_encoded, y_pred, average='macro')
            recall_macro = recall_score(self.y_test_encoded, y_pred, average='macro')
            f1_macro = f1_score(self.y_test_encoded, y_pred, average='macro')
            
            precision_weighted = precision_score(self.y_test_encoded, y_pred, average='weighted')
            recall_weighted = recall_score(self.y_test_encoded, y_pred, average='weighted')
            f1_weighted = f1_score(self.y_test_encoded, y_pred, average='weighted')
            
            print(f"\n🏆 整体性能指标:")
            print(f"{'='*50}")
            print(f"总体准确率 (Accuracy):        {accuracy:.4f}")
            print(f"宏平均精确率 (Macro Precision): {precision_macro:.4f}")
            print(f"宏平均召回率 (Macro Recall):    {recall_macro:.4f}")
            print(f"宏平均F1分数 (Macro F1):       {f1_macro:.4f}")
            print(f"加权平均精确率 (Weighted Precision): {precision_weighted:.4f}")
            print(f"加权平均召回率 (Weighted Recall):    {recall_weighted:.4f}")
            print(f"加权平均F1分数 (Weighted F1):       {f1_weighted:.4f}")
            
            # 保存结果
            self.results[name] = {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'tn': tn, 'tp': tp, 'fn': fn, 'fp': fp
            }
            
            # 详细分类报告
            print(f"\n📋 Scikit-learn 分类报告:")
            print(f"{'='*50}")
            target_names = self.label_encoder.classes_
            print(classification_report(self.y_test_encoded, y_pred, 
                                      target_names=target_names))
    
    def plot_confusion_matrix(self):
        """绘制逻辑回归的混淆矩阵"""
        if not self.results:
            print("没有评估结果可以绘制")
            return
            
        name = list(self.models.keys())[0]  # 获取唯一的模型名称
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 混淆矩阵热力图
        cm = self.results[name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=ax1)
        ax1.set_title(f'{name}\n混淆矩阵')
        ax1.set_xlabel('预测标签')
        ax1.set_ylabel('真实标签')
        
        # 性能指标柱状图
        metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']
        values = [
            self.results[name]['accuracy'],
            self.results[name]['precision_macro'],
            self.results[name]['recall_macro'],
            self.results[name]['f1_macro']
        ]
        
        bars = ax2.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax2.set_title(f'{name}\n性能指标')
        ax2.set_ylabel('分数')
        ax2.set_ylim(0, 1)
        
        # 在柱状图上添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('logistic_regression_results.png', dpi=300, bbox_inches='tight')
        print("📊 图表已保存为 'logistic_regression_results.png'")
        plt.show()
    
    def plot_detailed_analysis(self):
        """绘制详细的分析图表"""
        if not self.results:
            print("没有评估结果可以绘制")
            return
            
        name = list(self.models.keys())[0]
        result = self.results[name]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 混淆矩阵组成部分饼图
        tn, tp, fn, fp = result['tn'], result['tp'], result['fn'], result['fp']
        labels = ['TN (真阴性)', 'TP (真阳性)', 'FN (假阴性)', 'FP (假阳性)']
        sizes = [tn, tp, fn, fp]
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('混淆矩阵组成分布')
        
        # 2. 正确vs错误预测
        correct = tn + tp
        incorrect = fn + fp
        ax2.bar(['正确预测', '错误预测'], [correct, incorrect], 
                color=['green', 'red'], alpha=0.7)
        ax2.set_title('预测准确性')
        ax2.set_ylabel('样本数量')
        for i, v in enumerate([correct, incorrect]):
            ax2.text(i, v + 50, str(v), ha='center', va='bottom')
        
        # 3. 各类别性能对比
        neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        pos_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        categories = ['Negative', 'Positive']
        precision_scores = [neg_precision, pos_precision]
        recall_scores = [neg_recall, pos_recall]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax3.bar(x - width/2, precision_scores, width, label='Precision', alpha=0.8)
        ax3.bar(x + width/2, recall_scores, width, label='Recall', alpha=0.8)
        ax3.set_ylabel('分数')
        ax3.set_title('各类别精确率vs召回率')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # 4. 整体指标雷达图风格的条形图
        metrics = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1\n(Macro)']
        values = [
            result['accuracy'],
            result['precision_macro'], 
            result['recall_macro'],
            result['f1_macro']
        ]
        
        bars = ax4.barh(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax4.set_xlabel('分数')
        ax4.set_title('整体性能指标')
        ax4.set_xlim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{value:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 详细分析图表已保存为 'detailed_analysis.png'")
        plt.show()
    
    def predict_new_text(self, text):
        """预测新文本的情感"""
        if self.vectorizer is None or len(self.models) == 0:
            print("请先加载数据和训练模型！")
            return None
        
        # 预处理文本
        from preprocess import IMDBPreprocessor
        processor = IMDBPreprocessor("")
        clean_text = processor.remove_html_tags(text)
        clean_text = processor.convert_to_lowercase(clean_text)
        clean_text = processor.remove_special_characters(clean_text)
        clean_text = processor.remove_extra_whitespace(clean_text)
        
        # 转换为TF-IDF特征
        text_features = self.vectorizer.transform([clean_text])
        
        print(f"\n=== 文本情感预测 ===")
        print(f"原文: {text}")
        print(f"预处理后: {clean_text}")
        print(f"\n各模型预测结果:")
        
        results = {}
        for name, model in self.models.items():
            pred = model.predict(text_features)[0]
            pred_label = self.label_encoder.inverse_transform([pred])[0]
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(text_features)[0]
                confidence = max(proba)
                results[name] = {
                    'prediction': pred_label,
                    'confidence': confidence,
                    'probabilities': dict(zip(self.label_encoder.classes_, proba))
                }
                print(f"{name:20s}: {pred_label:8s} (置信度: {confidence:.4f})")
            else:
                results[name] = {'prediction': pred_label}
                print(f"{name:20s}: {pred_label}")
        
        return results

def main():
    # 创建分类器
    classifier = SentimentClassifier()
    
    # 加载数据
    if not classifier.load_data():
        return
    
    # 训练模型
    classifier.train_models()
    
    # 测试集评估
    classifier.evaluate_on_test_set()
    
    # 绘制结果
    classifier.plot_confusion_matrix()
    classifier.plot_detailed_analysis()
    
    # 创建结果摘要
    print("\n" + "="*80)
    print("📊 逻辑回归模型最终结果摘要")
    print("="*80)
    
    name = list(classifier.results.keys())[0]
    result = classifier.results[name]
    
    print(f"🏆 模型: {name}")
    print(f"🎯 测试集准确率: {result['accuracy']:.4f}")
    print(f"� 混淆矩阵统计:")
    print(f"   • TN (真阴性): {result['tn']:,}")
    print(f"   • TP (真阳性): {result['tp']:,}") 
    print(f"   • FN (假阴性): {result['fn']:,}")
    print(f"   • FP (假阳性): {result['fp']:,}")
    print(f"📈 整体性能:")
    print(f"   • 宏平均精确率: {result['precision_macro']:.4f}")
    print(f"   • 宏平均召回率: {result['recall_macro']:.4f}")
    print(f"   • 宏平均F1分数: {result['f1_macro']:.4f}")
    
    # 演示预测新文本
    print("\n" + "="*60)
    print("🔮 演示新文本预测")
    print("="*60)
    
    test_texts = [
        "This movie is absolutely amazing! I loved every minute of it.",
        "Terrible film, waste of time. Very disappointed.",
        "The movie was okay, nothing special but not bad either."
    ]
    
    for text in test_texts:
        classifier.predict_new_text(text)
        print("-" * 50)

if __name__ == "__main__":
    main()