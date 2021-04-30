import gensim
from pprint import pprint

def get_number_topics(corpus, id2word, max_topic):
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

if __name__ == '__main__':
    num_topics,_ = get_number_topics(corpus,id2word,10)
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)
    pprint(lda_model.print_topics())
    df=assign_topic(df)
