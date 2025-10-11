# IMDB电影评论情感分析项目

 基于机器学习的IMDB电影评论情感分析，支持GPU加速训练

##  项目结构

```
imdb-sentiment-analysis/
│
├── preprocess.py          # 数据预处理脚本
├── tf_idf.py             # TF-IDF特征提取
├── cuda_training.py      # CUDA GPU加速训练
├── gpu_accelerated.py    # GPU加速对比测试
├── test_preprocess.py    # 预处理效果测试
│
├── features/             # 特征文件目录
├── .gitignore           # Git忽略文件
└── README.md            # 项目说明
```

## 开始

### 1. 环境要求

- Python 3.8+
- pandas, numpy, scikit-learn
- beautifulsoup4 (HTML清理)
- nltk (自然语言处理)

### 2. 安装依赖

```bash
# 基础依赖
pip install pandas numpy scikit-learn beautifulsoup4 nltk matplotlib seaborn

```

### 3. 使用方法

#### 步骤1: 数据预处理
```bash
python preprocess.py
```
- 移除HTML标签
- 转换为小写
- 清理特殊字符
- 规范化空白字符


