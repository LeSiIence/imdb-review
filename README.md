# IMDB电影评论情感分析

IMDB电影评论情感分析(基于TF-IDF表征和Word2Vec表征)

## 思考题解答：

1. **哪种表征方法在这个任务上表现更好？为什么？**

实际结果如混淆矩阵和性能柱状图所示（下方给出）。在大多数情况下，TF-IDF方法在该任务上简单直接，处理稀疏文本数据时拥有较稳健的表现，尤其是在训练数据不够丰富时；而Word2Vec方法可以捕捉词语之间的语义关系，对于表达方式多样或文本较长的场景更有潜力。实验中，**TF-IDF模型的准确率和F1分数更高，说明TF-IDF在本数据集上表现更好**，原因可能是IMDB影评用词分布较为规律，词频特征就已足够有效。

2. **Word2Vec方法中，直接将所有词向量“平均”得到句子向量，有什么优点和缺点？**

- **优点**：
  - 简单高效，易于实现，计算成本低，训练和推理速度快。
  - 能够一定程度融合整句的整体语义信息。
  - 对词序不敏感，鲁棒性较高。

- **缺点**：
  - 丢失了词序和局部上下文信息，难以区分不同顺序表达的句子。
  - 停用词、无区分性词语的影响未被抑制，可能稀释关键信息。
  - 对长句和短句的分布不敏感，不考虑词语的重要性。

---

**模型混淆矩阵示例：**


- **TF-IDF + Logistic Regression**

<img src="tf_idf_confusion_matrix.png" alt="TF-IDF Confusion Matrix" width="800"/>

- **Word2Vec + Logistic Regression**

<img src="word2vec_confusion_matrix.png" alt="Word2Vec Confusion Matrix" width="800"/>


## 数据集来源

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download&select=IMDB+Dataset.csv

按照kaggle要求，采用前50%数据进行训练和验证（做80%、20%分割），后50%进行测试。

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
├── preprocess.py                 # 数据预处理脚本
├── tf_idf.py                     # TF-IDF特征提取并使用逻辑回归，输出评测结果
├── tf_idf_confusion_matrix.png   # 评测结果
├── word2vec.py                   # Word2Vec特征提取并使用逻辑回归，输出评测结果
├── word2vec_confusion_matrix.png # 评测结果
├── IMDB Dataset.csv              # 原始数据集（available at kaggle）
├── IMDB Dataset_preprocessed.csv # 预处理结果
│  
├── .gitignore
└── README.md
```

## 开始

### 1. 环境要求

- numpy==1.26.4
- pandas==2.2.2
- scikit-learn==1.4.2
- matplotlib==3.8.4
- seaborn==0.13.2
- gensim==4.3.2
- beautifulsoup4==4.12.3
- lxml==5.2.1
- nltk==3.8.1

### 2. 安装依赖

```bash
# 基础依赖
conda create -n datasci_imdb_new -c conda-forge -y python=3.10
conda activate datasci_imdb_new
conda install -c conda-forge -y numpy pandas scikit-learn matplotlib seaborn
conda install -c conda-forge -y beautifulsoup4 lxml nltk
conda install -c conda-forge -y gensim
python -m ipykernel install --user --name=datasci_imdb_new --display-name "Python (datasci_imdb_new)"
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


#### 步骤2: 使用 TF-IDF + 逻辑回归训练与评测

```bash
python tf_idf.py
```
- 自动载入预处理后的 `IMDB_Dataset_Preprocessed.csv`
- 输出主要模型评测指标（Accuracy, Precision, Recall, F1 Score）
- 显示并保存混淆矩阵与性能柱状图

#### 步骤3: 使用 Word2Vec + 逻辑回归训练与评测

```bash
python word2vec.py
```
- 自动载入预处理后的 `IMDB_Dataset_Preprocessed.csv`
- 输出主要模型评测指标（Accuracy, Precision, Recall, F1 Score）
- 显示并保存混淆矩阵与性能柱状图






