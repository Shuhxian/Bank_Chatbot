import gensim
from pprint import pprint
import umap.umap_ as umap
import hdbscan
from word_embedding import get_wod_embedding
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

def assign_topic(df):
  #Assign topic and their probability to the dataframe
    for i in range(df.shape[0]):
        df.loc[i,'topic']=sorted(lda_model[corpus[i]],reverse=True,key=lambda x:x[1])[0][0]
        df.loc[i,'topic_probability']=sorted(lda_model[corpus[i]],reverse=True,key=lambda x:x[1])[0][1]
    return df

def train_classifier(df,corpus,id2word,bigram):
    #Get the optimum number of topics
    num_topics,_ = get_number_topics(corpus,id2word,bigram,10)
    #Assign topics based on optimum number of topics
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)
    df=assign_topic(df)
    
    #Get the word embeddings
    sentence_embeddings=df['token'].apply(get_word_embedding) 
    
    #Split the tensor into 768 columns for clustering
    df_new=pd.DataFrame(columns=[i for i in range(768)],index=[i for i in range(df.shape[0])])
    for i in range(df.shape[0]):
        for j in range(768):
            df_new.iloc[i,j]=sentence_embeddings[i][j]
            
    #Concat the tensors with the original dataframe       
    df=pd.concat([df,df_new],axis=1)
    #Filter out LDA and word embedding
    embeddings=df.iloc[:,4:] 

    #Clustering, n_neighbours and min_cluster_size can be adjusted according to sample size
    reducer=umap.UMAP(n_neighbors=5, 
                  n_components=5, 
                  metric='cosine',
                  random_state=1)
    umap_embeddings = reducer.fit_transform(embeddings)
    cluster = hdbscan.HDBSCAN(min_cluster_size=3,
                            metric='euclidean',                      
                            cluster_selection_method='eom').fit(umap_embeddings)

    #Create a dataframe with final label and probability
    #umap_data = umap.UMAP(n_neighbors=5, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    #result = pd.DataFrame(umap_data, columns=['x', 'y'])
    #result['labels'] = cluster.labels_
    #result['prob']=cluster.probabilities_
    return lda_model, reducer,cluster
    
def predict(input_token,input_corpus,lda_model,reducer,cluster):
    df=pd.DataFrame([input_token,input_corpus],columns={'token','corpus'})
    #Assign topics based on optimum number of topics
    df[0,'topic']=sorted(lda_model[input_corpus],reverse=True,key=lambda x:x[1])[0][0]
    df[0,'topic_probability']=sorted(lda_model[input_corpus],reverse=True,key=lambda x:x[1])[0][1]
    
    #Get the word embeddings
    sentence_embeddings=get_word_embedding(input_token)
    
    #Split the tensor into 768 columns for clustering
    df_new=pd.DataFrame(columns=[i for i in range(768)],index=[0])
    for i in range(df.shape[0]):
        for j in range(768):
            df_new.iloc[i,j]=sentence_embeddings[i][j]
            
    #Concat the tensors with the original dataframe       
    df=pd.concat([df,df_new],axis=1)
    #Filter out LDA and word embedding
    embeddings=df.iloc[:,4:] 
    chat_embeddings=embeddings.iloc[0,:][None,:]
    test_data = reducer.transform(chat_embeddings)
    test_labels, test_prob = hdbscan.approximate_predict(cluster, test_data)
    return test_labels,test_prob

