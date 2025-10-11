import pandas as pd
import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

class IMDBPreprocessor:
    def __init__(self, csv_file_path):     
        self.csv_file_path = csv_file_path
        self.data = None
        
    def load_data(self):  
        print("加载数据集")
        try:
            self.data = pd.read_csv(self.csv_file_path)
            
            return True
        except Exception as e:
            print(f"fuck: 加载数据集失败: {e}")
            return False
    
    def remove_html_tags(self, text):
        if pd.isna(text):
            return ""
        
        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text()        
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()#正则：'\s+'表示空白字符后再出现1+个空白
        
        return clean_text
    
    def convert_to_lowercase(self, text):        
        if pd.isna(text):
            return ""
        return text.lower()
        
    def remove_special_characters(self, text):
        if pd.isna(text):
            return ""        
        
        clean_text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'"()-]', '', text)       
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def remove_extra_whitespace(self, text):
        if pd.isna(text):
            return ""

        clean_text = re.sub(r'\s+', ' ', text).strip()#双重移除
        
        return clean_text
    
    def remove_stopwords(self, text):
        if pd.isna(text):
            return ""
        return text
    
    def preprocess_text_column(self, column_name, remove_stopwords_flag=False):
        if column_name not in self.data.columns:
            print(f"fuck：列 '{column_name}' 不存在")
            return
        print('移除html')
        self.data[column_name] = self.data[column_name].apply(self.remove_html_tags)

        print("tolower")
        self.data[column_name] = self.data[column_name].apply(self.convert_to_lowercase)

        print("移除特殊字符")
        self.data[column_name] = self.data[column_name].apply(self.remove_special_characters)

        print("移除多余空白")
        self.data[column_name] = self.data[column_name].apply(self.remove_extra_whitespace)

        if remove_stopwords_flag:
            print("移除停用词")
            self.data[column_name] = self.data[column_name].apply(self.remove_stopwords)
        
        print(f"列 '{column_name}' 预处理完成")

    def save_processed_data(self, output_path):
        if self.data is not None:
            try:
                self.data.to_csv(output_path, index=False, encoding='utf-8')
                print(f"已保存到: {output_path}")
            except Exception as e:
                print(f"fuck: 保存文件时出错: {e}")

def main():
    input_file = "IMDB Dataset.csv"
    output_file = "IMDB_Dataset_Preprocessed.csv"   
    preprocessor = IMDBPreprocessor(input_file)
    if not preprocessor.load_data():
        return   
    # 预处理review列
    text_columns = ['review']   
    for column in text_columns:
        if column in preprocessor.data.columns:
            #remove_stopwords_choice = input(f"是否为列 '{column}' 移除停用词？(y/n，默认n): ").lower().strip()
            #remove_stopwords_flag = remove_stopwords_choice == 'y'
            remove_stopwords_flag = False
            preprocessor.preprocess_text_column(column, remove_stopwords_flag)
    preprocessor.save_processed_data(output_file)   
    print("预处理完成！")

if __name__ == "__main__":
    main()