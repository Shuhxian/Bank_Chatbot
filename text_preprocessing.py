# !python3 -c "import nltk; nltk.download('all')"

import re
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

# spacy for lemmatization
import spacy

# df = pd.read_excel (r'/content/Database.xlsx')
#df.columns = ['message', 'answer']
# data = df.message.values.tolist() 

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

#tokenize each sentence into a list of words, removing punctuations and unnecessary characters
def sent_to_words(sentences):
    return([word_tokenize(sentence) for sentence in sentences])

# Define functions for stopwords, bigrams, trigrams and lemmatization
# remove stopwords and tokenizing
def remove_stopwords(texts):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    return [[word for word in simple_preprocess(str(doc),min_len=0) if word not in stop_words] for doc in texts]

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

# def make_trigrams(texts):
#     return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    lemmatizer = WordNetLemmatizer()
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def get_corpus (df):
  data = df.message.values.tolist()
  data_words = list(sent_to_words(data))  
  # Remove Stop Words
  data_words_nostops = remove_stopwords(data_words)
  # Form Bigrams
  data_words_bigrams = make_bigrams(data_words_nostops)

  # Do lemmatization keeping only noun, adj, vb, adv
  data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])   

  # Create Dictionary
  id2word = corpora.Dictionary(data_lemmatized)

  # Create Corpus
  texts = data_lemmatized

  # Term Document Frequency
  corpus = [id2word.doc2bow(text) for text in texts]

  #(testing) Human readable format of corpus (term-frequency)
#   [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:4]]
  return df,corpus,id2word,data_words_bigrams


if __name__ == '__main__':
    df,corpus,id2word,bigram=get_corpus(df)
