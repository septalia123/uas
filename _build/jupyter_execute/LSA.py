#!/usr/bin/env python
# coding: utf-8

# # Crawling Data

# import scrapy
# 
# 
# class QuotesSpider(scrapy.Spider):
#     name = "quotes"
# 
#     def start_requests(self):
#         urls = [
#             'https://ekbis.sindonews.com/'
# 
#         ]
#         for url in urls:
#             yield scrapy.Request(url=url, callback=self.parse)
# 
#     def parse(self, response):
#         # print(response.url)
#         for i in range(0, 30):
#             for data in response.css('body > div:nth-child(5) > section > div.grid_24 > div.homelist-new > ul'):
#                 yield{
#                     'judul': data.css('li.latest-event.latest-track-' + str(i) + ' > div.homelist-box > div.homelist-title > a::text').extract(),
# 
#                     'waktu': data.css('li.latest-event.latest-track-' + str(i) + ' > div.homelist-box > div.homelist-top > div.homelist-date::text').extract(),
# 
#                     'category': data.css('li.latest-event.latest-track-' + str(i) + ' > div.homelist-box > div.homelist-top > div.homelist-channel::text').extract(),
# 
#                     'isi': data.css('li.latest-event.latest-track-' + str(i) + ' > div.homelist-box > div.homelist-desc::text').extract()
#                 }
#                 
# 

# # Import Module

# In[1]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop-words
stop_words=set(nltk.corpus.stopwords.words('indonesian'))


# # Load Dataset

# In[135]:


df=pd.read_csv('wrapping-text.csv')


# In[136]:


df.head()


# # We will drop the 'publish_date' column as it is useless for our discussion.

# In[137]:


# drop the publish date.
df.drop(['judul'],axis=1,inplace=True)


# In[138]:


df.drop(['waktu'],axis=1,inplace=True)


# In[139]:


df.drop(['category'],axis=1,inplace=True)


# In[140]:


df.head(30)


# # Clean Data & Preprocessing Data

# #### Here I have done the data pre-processing. I have used the lemmatizer and can also use the stemmer. Also the stop words have been used along with the words wit lenght shorter than 3 characters to reduce some stray words.

# In[141]:


import string 
import re #regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

df['hapus angka'] = df['isi'].apply(remove_number)
df.head(10)


# In[142]:


def clean_text(headline):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(headline)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text
  


# In[143]:


# time taking
#nltk.download('wordnet')
df['clean_text_isi']=df['hapus angka'].apply(clean_text)


# In[144]:


df.head()


# #### Can see the difference after removal of stopwords and some shorter words. aslo the words have been lemmatized as in 'calls'--->'call'.

# #### Now drop the unpre-processed column.

# In[145]:


df.drop(['isi'],axis=1,inplace=True)


# In[146]:


df.drop(['hapus angka'],axis=1,inplace=True)


# In[147]:


df.head()


# #### We can also see any particular news headline.

# In[148]:


df['clean_text_isi'][0]


# ### EXTRACTING THE FEATURES AND CREATING THE DOCUMENT-TERM-MATRIX ( DTM )
# In DTM the values are the TFidf values.
# 
# Also I have specified some parameters of the Tfidf vectorizer.
# 
# Some important points:-
# 
# 1) LSA is generally implemented with Tfidf values everywhere and not with the Count Vectorizer.
# 
# 2) max_features depends on your computing power and also on eval. metric (coherence score is a metric for topic model). Try the value that gives best eval. metric and doesn't limits processing power.
# 
# 3) Default values for min_df & max_df worked well.
# 
# 4) Can try different values for ngram_range.

# In[149]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000) # to play with. min_df,max_df,max_features etc...


# In[150]:


vect_text=vect.fit_transform(df['clean_text_isi'])


# #### We can now see the most frequent and rare words in the news headlines based on idf score. The lesser the value; more common is the word in the news headlines.

# In[151]:


print(vect_text.shape)
print(vect_text)


# In[152]:


idf=vect.idf_


# # Topik Modelling

# # Latent Semantic Analysis (LSA)
# The first approach that I have used is the LSA. LSA is basically singular value decomposition.
# 
# SVD decomposes the original DTM into three matrices S=U.(sigma).(V.T). Here the matrix U denotes the document-topic matrix while (V) is the topic-term matrix.
# 
# Each row of the matrix U(document-term matrix) is the vector representation of the corresponding document. The length of these vectors is the number of desired topics. Vector representation for the terms in our data can be found in the matrix V (term-topic matrix).
# 
# So, SVD gives us vectors for every document and term in our data. The length of each vector would be k. We can then use these vectors to find similar words and similar documents using the cosine similarity method.
# 
# We can use the truncatedSVD function to implement LSA. The n_components parameter is the number of topics we wish to extract. The model is then fit and transformed on the result given by vectorizer.
# 
# Lastly note that LSA and LSI (I for indexing) are the same and the later is just sometimes used in information retrieval contexts.

# In[153]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[154]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[155]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)
  


# #### Similalry for other documents we can do this. However note that values dont add to 1 as in LSA it is not probabiltiy of a topic in a document.

# In[156]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# #### Now e can get a list of the important words for each of the 10 topics as shown. For simplicity here I have shown 10 words for each topic.¶

# In[157]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[ ]:





# In[ ]:




