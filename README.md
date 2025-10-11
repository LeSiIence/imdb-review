# IMDB电影评论情感分析

IMDB电影评论情感分析(基于TF-IDF表征和Word2Vec表征)

## 数据集来源

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download&select=IMDB+Dataset.csv

按照kaggle要求，采用前50%数据进行训练和验证，后50%进行测试。

## 引用

IMDB数据集最初由斯坦福大学发布，用于情感分析研究：

> Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. In *Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies* (pp. 142-150).

**BibTeX引用:**
```bibtex
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
    author = {Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T. and Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
    title = {Learning Word Vectors for Sentiment Analysis},
    booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
    month = {June},
    year = {2011},
    address = {Portland, Oregon, USA},
    publisher = {Association for Computational Linguistics},
    pages = {142--150},
    url = {http://www.aclweb.org/anthology/P11-1015}
}
```

##  项目结构

```
imdb-sentiment-analysis/
│
├── preprocess.py          # 数据预处理脚本
├── tf_idf.py             # TF-IDF特征提取
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


