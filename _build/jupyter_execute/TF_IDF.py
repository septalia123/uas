#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


wrp = pd.read_csv('scrapy_berita.csv')


# In[2]:


wrp.shape


# In[3]:


wrp.head()


# In[4]:


#Menggambil data pada tabel isi
wrp['isi']


# # Case Folding

# In[5]:


#mengubah huruf menjadi kecil dengan menggunakan Series.str.lower() pada pandas

wrp['isi'] = wrp['isi'].str.lower()

print (wrp['isi'])


# # Filtering & Tokenisasi

# In[6]:


import nltk
import string 
import re #regex library
#nltk.download('punkt')

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

# ------ Proses Tokenisasi ---------

def remove_all(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
                
wrp['isi'] = wrp['isi'].apply(remove_all)

#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

wrp['isi'] = wrp['isi'].apply(remove_number)

#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

wrp['isi'] = wrp['isi'].apply(remove_punctuation)

#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

wrp['isi'] = wrp['isi'].apply(remove_whitespace_LT)

#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

wrp['isi'] = wrp['isi'].apply(remove_whitespace_multiple)

# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

wrp['isi'] = wrp['isi'].apply(remove_singl_char)

# NLTK word rokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)

wrp['isi_tokenisasi'] = wrp['isi'].apply(word_tokenize_wrapper)

print('Hasil Tokenisasi : \n') 
print(wrp['isi_tokenisasi'].head())
print('\n\n\n')


# # Stopword menggunkan nltk

# In[49]:


nltk.download('stopwords')


# In[7]:


from nltk.corpus import stopwords

# ----------------------- get stopword from NLTK stopword -------------------------------
# get stopword indonesia
list_stopwords = stopwords.words('indonesian')


# ----------------------- add stopword from txt file ------------------------------------
# read txt stopword using pandas
txt_stopword = pd.read_csv("scrapy_berita.csv", names= ["stopwords"], header = None)

# ---------------------------------------------------------------------------------------

# convert list to dictionary
list_stopwords = set(list_stopwords)


#remove stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

wrp['isi_stopword'] = wrp['isi_tokenisasi'].apply(stopwords_removal) 

print(wrp['isi_stopword'].head())


# # Stemming bahasa indonesia menggunakan Sastrawi

# In[20]:


pip install swifter


# In[8]:


wrp.head()


# In[9]:


#import Sastrawi package
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in wrp['isi_stopword']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])
    
print(term_dict)
print("------------------------")


# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

wrp['isi_stemmer'] = wrp['isi_stopword'].swifter.apply(get_stemmed_term)
print(wrp['isi_stemmer'])


# In[32]:


wrp.to_csv('hasilTextProcessing.csv')


# # Frequency Distribution

# In[1]:


pip install scikit-learn


# In[10]:


from nltk.probability import FreqDist

wrp['isi_freq_dist'] = wrp['isi_stemmer'].apply(FreqDist)
wrp['isi_freq_dist']


# In[11]:


wrp.head()


# In[12]:


wrp.to_csv('hasilTextProcessing.csv')


# # Text Frequency Table

# In[13]:


tf_table = wrp['isi_freq_dist']
tf_table = pd.DataFrame(tf_table.tolist()).fillna(0)
#bentuk aljabar
tf_table


# In[14]:


wrp.head()


# In[ ]:





# # Analize

# 

# In[15]:


data = [" ". join(wrp) for wrp in wrp['isi_stemmer']]
data


# # TF

# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

vectorizer = CountVectorizer()
tfidf = TfidfVectorizer()
idf = TfidfVectorizer(use_idf=True)


# In[17]:


bag = vectorizer.fit_transform(data)
vectorizer.vocabulary_


# # IDF

# In[18]:


response = idf.fit_transform(data)
idf.idf_


# In[19]:


response = tfidf.fit_transform(data)
print(response)


# In[20]:


tfidf.get_feature_names()


# In[21]:


response.todense()


# In[22]:


df = pd.DataFrame(response.todense().T, index=tfidf.get_feature_names(), columns=[f'B{i+1}' for i in range (len(data))])
df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 
