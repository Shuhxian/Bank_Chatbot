import pandas as pd
from google.colab import files
import io
import re
import nltk
#nltk.download()
import gensim
import umap.umap_ as umap
import hdbscan
import pickle
from pprint import pprint
from word_embedding import get_word_embedding

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
    df['token']=df['message'].apply(nltk.word_tokenize)
    df['token']=df['token'].apply(remove_stopwords) 
    df['token']=df['token'].apply(lemmatize) 
    words=df.token.values.tolist() 
    bigram_mod = bigrams(words)
    bigram = [bigram_mod[review] for review in words]
    id2word = gensim.corpora.Dictionary(bigram)
    id2word.compactify()
    corpus = [id2word.doc2bow(text) for text in bigram]
    return df,corpus, id2word, bigram

def get_number_topics(corpus, id2word, bigram, max_topic):
    #Calculate coherence score and determine optimum number of topics
    best_num=-1
    best_score=-1
    for k in range(1,max_topic+1):
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=k, 
                                               random_state=100,
                                               chunksize=100,
                                               passes=10)
    
        coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=bigram, dictionary=id2word, coherence='c_v')
        if coherence_model_lda.get_coherence()>best_score:
          best_num=k
          best_score=coherence_model_lda.get_coherence()  
    return best_num,best_score

def assign_topic(df,lda_model,corpus):
  #Assign topic and their probability to the dataframe
    for i in range(df.shape[0]):
        df.loc[i,'topic']=sorted(lda_model[corpus[i]],reverse=True,key=lambda x:x[1])[0][0]
        df.loc[i,'topic_probability']=sorted(lda_model[corpus[i]],reverse=True,key=lambda x:x[1])[0][1]
    return df

def train_classifier():
  df = pd.read_csv("Book1.csv")

  df,corpus,id2word, bigram=get_corpus(df)

  num_topics,_ = get_number_topics(corpus,id2word,bigram,15)
  lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
  pprint(lda_model.print_topics())
  df=assign_topic(df,lda_model,corpus)
  sentence_embeddings=df['token'].apply(get_word_embedding)
  df_new=pd.DataFrame(columns=[i for i in range(768)],index=[i for i in range(df.shape[0])])
  for i in range(df.shape[0]):
      for j in range(768):
          df_new.iloc[i,j]=sentence_embeddings[i][j]
  df=pd.concat([df,df_new],axis=1)
  embeddings=df.iloc[:,4:774]
  reducer=umap.UMAP(n_neighbors=5, 
                    n_components=5, 
                    metric='cosine',
                    random_state=1)
  umap_embeddings = reducer.fit_transform(embeddings)

  cluster = hdbscan.HDBSCAN(min_cluster_size=3,
                            metric='euclidean',                      
                            cluster_selection_method='eom',
                            prediction_data=True).fit(umap_embeddings)
  lda_model.save('lda.model')
  pickle.dump(corpus, open('corpus.sav', 'wb'))
  pickle.dump(reducer, open('reducer.sav', 'wb'))
  pickle.dump(cluster, open('cluster.sav', 'wb'))
  # Visualize clusters
  #umap_data = umap.UMAP(n_neighbors=5, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
  #result = pd.DataFrame(umap_data, columns=['x', 'y'])
  #result['labels'] = cluster.labels_
  #result['prob']=cluster.probabilities_
  #import matplotlib.pyplot as plt
  #fig, ax = plt.subplots(figsize=(20, 10))
  #outliers = result.loc[result.labels == -1, :]
  #clustered = result.loc[result.labels != -1, :]
  #plt.scatter(outliers.x, outliers.y)
  #plt.scatter(clustered.x, clustered.y, c=clustered.labels)
  #plt.colorbar()

def predict(input_string):
    corpus = pickle.load((open('corpus.sav','rb')))
    lda_model = gensim.models.LdaMulticore(corpus=corpus)
    lda_model.load("lda.model")
    reducer = pickle.load((open('reducer.sav', 'rb')))  
    cluster = pickle.load((open('cluster.sav', 'rb')))

    df2=pd.DataFrame([input_string],columns={'message'})
    df2,corpus2,_,_=get_corpus(df2)
    #Assign topics based on optimum number of topics
    df2[0,'topic']=sorted(lda_model[corpus2[0]],reverse=True,key=lambda x:x[1])[0][0]
    df2[0,'topic_probability']=sorted(lda_model[corpus2[0]],reverse=True,key=lambda x:x[1])[0][1]
    
    #Get the word embeddings
    sentence_embeddings=get_word_embedding(df2.loc[0,'token'])
    
    #Split the tensor into 768 columns for clustering
    df_new=pd.DataFrame(columns=[i for i in range(768)],index=[0])
    for j in range(768):
      df_new.iloc[0,j]=sentence_embeddings[j]
            
    #Concat the tensors with the original dataframe       
    df2=pd.concat([df2,df_new],axis=1)
    #Filter out LDA and word embedding
    embeddings=df2.iloc[:,4:] 
    chat_embeddings=embeddings.iloc[0,:][None,:]
    test_data = reducer.transform(chat_embeddings)
    test_labels, test_prob = hdbscan.approximate_predict(cluster, test_data)
    return test_labels,test_prob

if __name__ == '__main__':
    test_labels,test_prob=predict("Hi")
    print(test_labels,test_prob)
