# !python3 -c "import nltk; nltk.download('all')"
# !pip install contractions
import re, string
import nltk
import contractions
import inflect
import numpy as np
import pandas as pd

# Gensim
import gensim

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def sent_to_words(text):
    return(nltk.word_tokenize(text))

def to_lowercase(words):
    new_words = []
    """Convert all characters to lowercase from list of tokenized words"""
    return [new_words.append(word.lower()) for word in words]

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    return [new_words.append(re.sub(r'[^\w\s]', '', word)) for word in words]

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    return [word for word in words if not word in stopwords.words('english')]
    
def lemmatization(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    return [(lemmatizer.lemmatize(word, pos='v')) for word in words]
   
def get_corpus (data):
  text=replace_contractions(data)
  data_words = sent_to_words(text)
  # Remove Stop Words
  data_words_nostops = remove_stopwords(data_words)
  # Do lemmatization in only verb
  data_lemmatized = lemmatization(data_words_nostops)   

  op=np.asarray(data_lemmatized)
  return op
