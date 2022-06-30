#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import library yang di butuhkan
from openpyxl import load_workbook #library untuk menampilkan dokumen
import pandas as pd #import pandas 
from nltk.tokenize import word_tokenize #import library nltk - tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory #import library sastrawi untuk

wr = load_workbook(filename = 'WrpText.xlsx')
sheet_range = wr['wrp-text']

df = pd.DataFrame(sheet_range.values)
df.columns = ['Judul', 'Waktu', 'Category', 'Isi']
df


# In[3]:


#langkah pertama dengan membuka datasets bagian Komentar nya saja
df[['Isi']] #memanggil / tampil tabel komentar


# In[4]:


isi = [] #deklarasi variabel komentar pada list
isi = df['Isi'].values.tolist() #masukan data kedalam list
isi #cetak data dalam list


# # Text Processing

# In[5]:


list_review=[]
list_review.append(df['Isi'].values.tolist())

print (list_review)


# In[6]:


#TOKENIZE berdasarkan kata

import nltk 
nltk.download('punkt') #download library untuk tokenize
from nltk.tokenize import word_tokenize #library memanggil fungsi tokenisasi

for i in list_review: #mengambil librari yang sebelumnya
    for j in i:
        df_token = word_tokenize(j)
        for k in df_token:
            kecil = k.lower()  #memanggil fugsi lower
            print(kecil) #hasil dari data yang telah di lower kan / di kecilkan


# In[7]:


tokenisasi_isi = [] #melakukan tokenisasi berdasarkan kalimat
for y in isi:
    lowerisi = y.lower()
    proses_tokenisasi = word_tokenize(lowerisi)
    tokenisasi_isi.append(proses_tokenisasi)
    
print(tokenisasi_isi)


# # Stopword

# In[8]:


import nltk
nltk.download('stopwords')


# In[9]:


from nltk.corpus import stopwords
stpw = set(stopwords.words('indonesian'))

stopword = []
for i in tokenisasi_isi:
    stopword.append([word for word in i if not word in stpw])

print(stopword)


# # Filtering

# ## Menghilangkan Angka

# In[10]:


import re
import string

filtering = [] #deklarasi variabel proses filtering
for a in stopword: #perulangan untuk melakukan tokenisasi
    for b in a:
        angka = re.sub(r"\d+", "",b)
        filtering.append(angka)
        
print (filtering)


# ## Menghilangkan Tanda Baca & Karakter Kosong

# In[11]:


filtering2 = [] #deklarasi variabel proses filtering
for q in filtering:
    tanda = q.translate(str.maketrans("","",string.punctuation))
    if tanda != '':
        filtering2.append(tanda)
        
print (filtering2)


# # Steammer per kata

# In[30]:


factory = StemmerFactory()
stemmer = factory.create_stemmer()

#stem = []

for x in filtering2:
    stem.append(stemmer.stem(x))
    print (x, " : ",stemmer.stem(x))
    
#print(stem)
    


# In[ ]:




