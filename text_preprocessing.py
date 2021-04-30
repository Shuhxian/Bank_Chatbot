import pandas as pd
import re
import nltk
#nltk.download()
import gensim

def lemmatize(x): 
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in x]

def remove_stopwords(x):
    stopwords = nltk.corpus.stopwords.words('english')
    return [token for token in x if not token in stopwords]

def bigrams(words, bi_min=15):
    bigram = gensim.models.Phrases(words, min_count = bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod

def get_corpus(df):
    df['message']=df['message'].map(lambda x: re.sub("[,\.!'?]", '', x)).str.lower()
    df['token']=df['message'].apply(lambda x:nltk.word_tokenize(x))
    df['token']=df['token'].apply(lambda x: remove_stopwords(x)) 
    df['token']=df['token'].apply(lambda x:lemmatize(x)) 
    words=data.token.values.tolist() 
    bigram_mod = bigrams(words)
    bigram = [bigram_mod[review] for review in words]
    id2word = gensim.corpora.Dictionary(bigram)
    id2word.filter_extremes(no_below=10, no_above=0.35)
    id2word.compactify()
    corpus = [id2word.doc2bow(text) for text in bigram]
    return df,corpus, id2word
