import requests
from PIL import Image
from io import BytesIO
import requests
import json
import random
from PIL import Image,ImageDraw
import os
import pprint
import time
import urllib2
import pandas as pd
from datetime import date,timedelta,datetime
import re
import random
from PIL import ImageFilter
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim import parsing, corpora
#os.environ['HTTPS_PROXY']="fpproxy.in.ril.com:8080"
today =datetime.today().strftime('%Y%m%d')
headers = {
    'X-CleverTap-Account-Id': 'xxx-xxx-xxx',
    'X-CleverTap-Passcode': 'xxx-xxx-xxx',
    'Content-Type': 'application/json',
}
r2 = requests.get("http://cdnsrv.jio.com/jiotv.data.cdn.jio.com...{maskedAPI}.../get/?os=android&devicetype=phone")
chan={}
for item in r2.json()['result']:
        chan[item['channel_id']]=item['channel_name']
corpus=[]
list1=[]
cols = ['showId','showname','keywords','description','episode_desc','showGenre','starCast','director','duration']
showKeywords=pd.DataFrame(columns = cols)
for i in chan:
    try:
        r = requests.get('http://cdnsrv.jio.com/jiotv.data.cdn.jio.com/...{maskedAPI}...get?offset=-1&channel_id='+str(i)).json()
    except ValueError:
        print("No value for "+str(i))
    for item in r['epg']:
        list1.append([item['showId'],item['showname'],item['keywords'],item['description'],item['episode_desc'],item['showGenre'],item['starCast'],item['director'],item['duration']])
        corpus.append(item['keywords'])
showKeywords = pd.DataFrame(list1, columns=cols)
flat_list = [item for sublist in corpus for item in sublist]

showKeywords1=showKeywords.drop_duplicates(subset=['showId'], keep='first').reset_index()
showKeywords1=showKeywords1.drop(showKeywords1.columns[0], axis=1)

showKeywords2=showKeywords1.drop_duplicates(subset=['showname','description','episode_desc'], keep='first').reset_index()
showKeywords2=showKeywords2.drop(showKeywords2.columns[0], axis=1)

#This is the MetaData Approach
showKeywords2['nospacekeywords']=showKeywords2['keywords'].apply(lambda x: [i.replace(' ','') for i in x])
showKeywords2['metadata']=showKeywords2['nospacekeywords'].apply(' '.join)
showKeywords2['metadata_spaced']=showKeywords2['keywords'].apply(' '.join)

count_vec = CountVectorizer()
count_vec_matrix = count_vec.fit_transform(showKeywords2['metadata'])
cosine_sim_matrix = cosine_similarity(count_vec_matrix, count_vec_matrix)
mapping = pd.Series(showKeywords2.index,index = showKeywords2['showId'])
showkiid="CHN-006930000PRG-549771701"
movie_index = mapping[showkiid]
similarity_score = list(enumerate(cosine_sim_matrix[movie_index]))
similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
similarity_score = similarity_score[0:15]
movie_indices = [i[0] for i in similarity_score]

pprint.pprint(similarity_score)

showKeywords2.to_csv("C:\Users\Rahul17.Jain\Downloads\RahulReports\Code\\for-keywords.csv",sep=',', encoding='utf-8')


#This is TFIDF Description Clustering Approach¶


rom sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
descriptions=showKeywords2['description'][:].tolist()
descriptions.head(6)

descriptions=showKeywords2[showKeywords2['description']!=""]['description'].tolist()
tfidfvect = TfidfVectorizer(stop_words='english')
X = tfidfvect.fit_transform(descriptions)

first_vector = X[0]
 
dataframe = pd.DataFrame(first_vector.T.todense(), index = tfidfvect.get_feature_names(), columns = ["tfidf"])
dataframe.sort_values(by = ["tfidf"],ascending=False).head()

num = 4
kmeans1 = KMeans(n_clusters = num, init = 'k-means++', max_iter = 5000, n_init = 1)
kmeans1.fit(X)
pprint.pprint(kmeans1.cluster_centers_) #This will print cluster centroids as tf-idf vectors

closest, _ = pairwise_distances_argmin_min(kmeans1.cluster_centers_, X)
closest

from sklearn.metrics import silhouette_score

sil = []
kmax = 20

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans12345 = KMeans(n_clusters = k).fit(X)
  labels = kmeans12345.labels_
  sil.append(silhouette_score(X, labels, metric = 'euclidean'))
 
 
pprint.pprint(sil)


#This is LDA Gensim Approach¶
from gensim import corpora, models
dictionary_LDA = corpora.Dictionary(showKeywords2['keywords'])
dictionary_LDA.filter_extremes(no_below=3)
corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in showKeywords2['keywords']]

num_topics = 10
%time lda_model = models.LdaModel(corpus, num_topics=num_topics, \
                                  id2word=dictionary_LDA, \
                                  passes=4, alpha=[0.01]*num_topics, \
                                  eta=[0.01]*len(dictionary_LDA.keys()))
                                  
                         
for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=10):
    print(str(i)+": "+ topic)

lda_model[corpus[166]]
showKeywords2.index[showKeywords2['showId']=="CHN-007460000PRG-406870000"].tolist()[0]


showKeywords3=showKeywords2
showKeywords3['topic']=""
showKeywords3['topic'].astype(object)
for index in range(0,len(showKeywords3)):
    topic_list=[]
    for topic,probability in lda_model[corpus[index]]:
        if probability>=0.25:
            topic_list.append((topic,probability))
    showKeywords3.at[index,'topic'] = topic_list
showKeywords3.head()


from gensim import corpora, models, similarities
from itertools import chain

lda_corpus = lda_model[corpus]
scores = list(chain(*[[score for topic_id,score in topic] \
                      for topic in [doc for doc in lda_corpus]]))
threshold = sum(scores)/len(scores)
print threshold
print

cluster1 = [j for i,j in zip(lda_corpus,showKeywords2['showname']) if i[0][1] > threshold]
#cluster2 = [j for i,j in zip(lda_corpus,showKeywords2['showname']) if i[1][1] > threshold]
#cluster3 = [j for i,j in zip(lda_corpus,showKeywords2['showname']) if i[2][1] > threshold]

print cluster1
#print cluster2
#print cluster3

%matplotlib inline
import pyLDAvis
import pyLDAvis.gensim
vis = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=corpus, dictionary=dictionary_LDA)
pyLDAvis.enable_notebook()
pyLDAvis.display(vis)

from gensim.models.coherencemodel import CoherenceModel
cm=CoherenceModel(model=lda_model,texts=showKeywords2['keywords'], dictionary=dictionary_LDA, coherence='c_v')

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
hdpmodel = HdpModel(corpus=corpus, id2word=dictionary_LDA)
hdpmodel.show_topics()
