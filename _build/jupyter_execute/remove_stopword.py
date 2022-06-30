#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[ ]:


nltk.download('punkt')


# In[ ]:


get_ipython().system('pip install Sastrawi')


# In[ ]:


nltk.download('stopwords')


# In[ ]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Add text
text = "Saya akan makanan nasi di warung"
kata =text.lower()
tokens = word_tokenize(kata)
print(tokens)


english_stopwords = stopwords.words('indonesian')
list=['makan', 'nasi']
english_stopwords.extend(list)
tokens_wo_stopwords = [t for t in tokens if t not in english_stopwords]
print(tokens_wo_stopwords)

print("Text without stop words:", " ".join(tokens_wo_stopwords))
sting=" ".join(tokens_wo_stopwords)
print(sting)
output   = stemmer.stem(sting)
print(output)


# In[ ]:


import numpy as np
#import PyPDF2
#import doctext
import sys
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


# In[ ]:


cv = CountVectorizer()
cv_demo = CountVectorizer() 


# In[ ]:


demo_teks = ["Saya suka yang baik baik, kamu suka yang jelek jelek", "Saya tidak baik "] 
cv_matrix = cv.fit_transform(demo_teks)
res_demo = cv_demo.fit_transform(demo_teks)
print('Arraynya adalah {}'.format(res_demo.toarray()))
print('Feature list: {}'.format(cv_demo.get_feature_names_out()))


# In[ ]:


normal_matrix = TfidfTransformer().fit_transform(res_demo)


# In[ ]:


print(normal_matrix.toarray())


# In[24]:


get_ipython().system('pip install docx-text')


# In[25]:


get_ipython().system('pip install PyPDF3')


# In[26]:


import numpy as np
import PyPDF3
import doctext
import sys
from IPython.display import Image


# In[27]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:


import networkx as nx


# In[29]:


from nltk.tokenize.punkt import PunktSentenceTokenizer


# In[30]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


# In[31]:


get_ipython().system('pwd')


# In[34]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Colab Notebooks')


# In[36]:


pdfFileObj = open('datatext.pdf', 'rb') # digantikan hasil crawling anda
pdfReader = PyPDF3.PdfFileReader(pdfFileObj)
pageObj = pdfReader.getPage(0)
document = pageObj.extractText()


# In[37]:


doc_tokenizer = PunktSentenceTokenizer()
sentences_list = doc_tokenizer.tokenize(document)


# In[38]:


get_ipython().system('pip install sastrawi')


# In[39]:


import string 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()


# In[40]:


import re # impor modul regular expressionkalimat = "Berikut ini adalah 5 negara dengan pendidikan terbaik di dunia adalah Korea Selatan, Jepang, Singapura, Hong Kong, dan Finlandia."
dokumenre=[]
for i in sentences_list:
    hasil = re.sub(r"\d+", "", i)
    dokumenre.append(hasil) 
print(dokumenre)


# In[41]:


dokumen=[]
for i in dokumenre:
    hasil =  i.replace('\n','') 
    dokumen.append(hasil) 
print(dokumen)


# In[42]:


from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize
factory = StopWordRemoverFactory()
stopword  = factory.create_stop_word_remover()


# In[43]:


a=len(dokumen)
dokumenstop=[]
for i in range(0, a):
    sentence = stopword.remove(dokumen[i])
    dokumenstop.append(sentence)
print(dokumenstop)   


# In[44]:


from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize
factory = StopWordRemoverFactory()
dokumenstop=[]
for i in dokumen:
    output = i.translate(str.maketrans("","",string.punctuation))
    dokumenstop.append(output)
    
print(dokumenstop)


# In[45]:


factory = StemmerFactory()
stemmer = factory.create_stemmer()
dokumenstem=[]
for i in dokumenstop:
    output = stemmer.stem(i)
    dokumenstem.append(output)
print(dokumenstem)


# In[62]:


vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dokumenstem)


# In[63]:


print(vectorizer.vocabulary_)


# In[64]:


print(vectorizer.get_feature_names_out())


# In[65]:


matrik_vsm=bag.toarray()
#print(matrik_vsm)
matrik_vsm.shape


# In[66]:


matrik_vsm[0]


# In[67]:


import pandas as pd
a=vectorizer.get_feature_names_out()


# In[77]:


print(len(matrik_vsm[:,1]))
#dfb =pd.DataFrame(data=matrik_vsm,index=df,columns=[a])
dfb =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dfb


# ### Kata yang paling besar kata (pada) /term frequency dalam dokumen ketiga,paling sering muncul kata tersebut, tetapi setelah ditransformasi kedalam  tf-idfs, kata tersebut relatif kecil nilai tf-idfnya. Perhatikan kemunculan kata tersebut di  (discriminatory information).
# 
# ##  $$\text{idf} (t,d) = ln\frac{1 + n_d}{1 + \text{df}(d, t)}$$
# 
# $ \text{df}(d, t) =$  banyaknya dokumen yang mengandung term t    
# 
# 
# $ \text n_d =  $ banyaknya dokumen  
# 
# 
# Persamaan tf-idf yang diimplementasikan dalam scikit-learn adalah:
# 
# $$\text{tf-idf}(t,d) = \text{tf}(t,d) \times (\text{idf}(t,d)+1)$$
# 
# 
# $ \text{tf}(t,d) $ banyak kemunculan term dalam suatu dokumen
# 
# Kemudian dinormalisasi
# 
# 
# 
# $$v_{\text{norm}} = \frac{v}{||v||^2} = \frac{v}{\sqrt{v_{1}^{2} + v_{2}^{2} + \dots + v_{n}^{2}}} = \frac{v}{\big (\sum_{i=1}^{n} v_{i}^{2}\big)^\frac{1}{2}}$$
# 
# 
# 
# 

# In[69]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(use_idf=True,norm='l2',smooth_idf=True)
tf=tfidf.fit_transform(vectorizer.fit_transform(dokumenstem)).toarray()


# In[70]:


dfb =pd.DataFrame(data=tf,index=list(range(1, len(tf[:,1])+1, )),columns=[a])
dfb


# 

# 

# 

# 

# 

# In[71]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
 ]
print(corpus)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())

z=X.toarray()
#term frequency is printed
print(z)

vectorizer1 = TfidfVectorizer(min_df=1)
X1 = vectorizer1.fit_transform(corpus)
idf = vectorizer1.idf_
print (dict(zip(vectorizer1.get_feature_names(), idf)))
#printing idf
print(X1.toarray())
#printing tfidf

