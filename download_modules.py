import nltk 
nltk.download('stopwords')
nltk.download('punkt')

from transformers import BertTokenizer
BertTokenizer.from_pretrained('bert-base-uncased')