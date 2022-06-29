#!/usr/bin/env python
# coding: utf-8

# # TOPIK MODELLING

# Topik Modelling adalah pendekatan yang digunakan untuk menganalisis sekumpulan dokumen yang berbentuk teks yang kemudian dikelompokkan menjadi beberapa topik. Proses topik modelling dapat dilakukan dengan tahapan-tahapan sebagai berikut:
# 1. Crawling Data
# 2. Pre-Processing Data
#    - Cleansing Data
#    - Stopword
# 3. Modelling (LSA)

# # 1. Crawling Data Berita

# Crawling data adalah proses pengambilan data secara online untuk sebuah kebutuhan umum. Proses yang dilakukan yaitu mengimport suatu informasi atau sebuah data yang telah di ambil ke dalam file lokal pada komputer. Crawling data dilakukan untuk mengekstraksi data yang mengacu pengumpulan data dari worldwide web, dokumen-dokumen, file, dan lainnya. 

# Berikut adalah code untuk proses Crawling data website berita menggunakan Scrapy. Website berita yang di crawling adalah https://ekbis.sindonews.com/, data yang di ambil adalah data Judul, Waktu, Category berita, dan Deskripsi berita.

# In[1]:


import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'https://ekbis.sindonews.com/'

        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # print(response.url)
        for i in range(0, 30):
            for data in response.css('body > div:nth-child(5) > section > div.grid_24 > div.homelist-new > ul'):
                yield{
                    'judul': data.css('li.latest-event.latest-track-' + str(i) + ' > div.homelist-box > div.homelist-title > a::text').extract(),

                    'waktu': data.css('li.latest-event.latest-track-' + str(i) + ' > div.homelist-box > div.homelist-top > div.homelist-date::text').extract(),

                    'category': data.css('li.latest-event.latest-track-' + str(i) + ' > div.homelist-box > div.homelist-top > div.homelist-channel::text').extract(),

                    'isi': data.css('li.latest-event.latest-track-' + str(i) + ' > div.homelist-box > div.homelist-desc::text').extract()
                }
                


# # 2. Pre-Processing Data

# Pre-Processing Data merupakan suatu proses mengubah data mentah menjadi bentuk data yang lebih mudah dipahami. Proses dilakukan untuk membersihkan data dari angka, tanda baca, whitespace, dan kata yang tidak penting untuk digunakan. Tahapan dalam pre-processing data yaitu:

# ## Cleansing Data
# 

# Cleaning data atau membersihkan data artinya mengeksekusi ulang data yang telah di peroleh, seperti menghapus atau menghilangkan data-data yang tidak lengkap, tidak relavan, dan tidak akurat. Langkah pertama yang dilakukan untuk membulai proses pre-processing data adalah dengan mengimport modul-module yang diperlukan untuk proses pre-processing.

# ###  1. Import Module

# In[2]:


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


# ### 2. Load Dataset

# setelah proses import module yang digunakan, langkah kedua pada proses cleaning data adalah Load Data. Load Data adalah proses membuka/membaca file dari data yang akan digunakan untuk proses Pre-Processing, data yang digunakan biasanya berbentuk file csv, json dan lainnya. disini saya menggunakan format file csv.

# In[2]:


df=pd.read_csv('wrapping-text.csv')


# In[3]:


df.head()


# df.head() adalah proses untuk menampilkan data awal/teratas yang terdapat a pada file. Jika pada df.head() tidak diberikan parameter angka, hasil yang akan ditampilkan adalah 5 data teratas yang ada pada file.

# ### 3. Drop data yang tidak diperlukan

# Drop data adalah proses penghapusan data pada tabel data yang tidak diperlukan dalam proses pre-processing data. Pada tahap ini data yang dihapus adalah data yang ada pada tabel judul, waktu, dan category karena data yang akan digunakan adalah data yang ada pada tabel isi.

# In[4]:


# drop the publish date.
df.drop(['judul','waktu', 'category'],axis=1, inplace=True)


# tahap setelah penghapusan data yaitu menampilkan hasil data yang telah di proses dengan menggunakan df.head(). pada bagian ini terdapat nilai 30 sebagai parameter dari data yang akan ditampilkan, ini berarti banyaknya data yang akan di tampilkan adalah 30 data dimulai dari hitungan index ke 0.

# In[5]:


df.head(30)


# ### 4. Clean Data & Processing Data

# Proses ini adalah proses untuk membersihkan data dari angka-angka

# In[6]:


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


# ## Stopword

# Proses selanjutnya adalah proses Stopword. Stopword merupakan proses mengabaikan kata-kata yang perlu di abaikan dalam sebuah dokumen dan memilih kata-kata penting yang mewakili suatu dokumen.

# In[7]:


def clean_text(headline):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(headline)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text


# In[8]:


# time taking
#nltk.download('wordnet')
df['clean_text_isi']=df['hapus angka'].apply(clean_text)


# In[9]:


df.head()


# setelah Stopword, proses selanjutnya sama dengan proses sebelumnya yaitu menghapus data pada tabel yang tidak diperlukan dan hanya menyisakan data pada tabel hasil dari proses sebelumnya.

# In[10]:


df.drop(['isi', 'hapus angka'],axis=1,inplace=True)


# In[11]:


df


# Untuk menampilkan isi dari suatu index data pada tabel dapat dilakukan dengan menggunakan code berikut:

# In[12]:


df['clean_text_isi'][10]


# ### Mengekstraksi Fitur dan Membuat Document-Term-Matrix (DTM)

# Term Frequency-Inverse Document Frequency (TF-IDF) adalah algoritma yang digunakan untuk menganalisa hubungan natara frase/kalimat dengan sekumpulan dokumen. Inti algoritma TF-IDF adalah menghitung terlebih dahulu nilai TF dan nilai IDF dari masing -masing dokumen. 
# 
# 
# Term Frequency(TF) adalah jumlah kemunculan sebuah term pada sebuah dokumen
# Nilai TF dihitung menggunakan rumus:
# 
# $$
# \mathrm{tf}(t, d)=\frac{f_{t_{1} d}}{\sum_{t^{\prime} \in d} f_{t^{\prime}, d}}
# $$
# 
# Inverse Document Frequency(IDF) yaitu pengurangan dominasi term yang sering muncul diberbagai dokumen, dengan memperhitungkan kebalikan frekuensi dokumen yang mengandung suatu kata.
# dan nilai IDF dihitung dengan rumus:
# $$
# \operatorname{idf}=\log \left(\frac{D}{d f}\right)
# $$
# 
# setelah mendapatkan nilai TF dan IDF, yang akan kita hitung adalah nilai dari TF-IDF, berikut adalah rumus perhitungannya:
# $$
# \begin{gathered}
# Tf-idf=t f_{i j} * i d f_{j} \\
# Tf-idf=t f_{i j} * \log \left(\frac{D}{d f}\right)
# \end{gathered}
# $$
# 
# Catatan:
# 
# Df = jumlah dokumen yang didalamnya memuat term tertentu
# 
# D = Jumlah Dokumen yang di perbandingkan
# 

# Terdapat beberapa poin penting yaitu:
# 
# 1) LSA umumnya diimplementasikan dengan nilai Tf-Idf di mana-mana dan bukan dengan Count Vectorizer.
# 
# 2) max_features tergantung pada daya komputasi dan juga pada eval. Metrik (skor koherensi adalah metrik untuk model topik). 
# 
# 3) Nilai default untuk min_df &max_df bekerja dengan baik.
# 
# 4) Dapat mencoba nilai yang berbeda untuk ngram_range.

# In[25]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000) 


# In[14]:


vect_text=vect.fit_transform(df['clean_text_isi'])


# In[15]:


print(vect_text.shape)
print(vect_text)


# In[16]:


vect.get_feature_names()


# In[17]:


vect_text.todense()


# In[18]:


df = pd.DataFrame(vect_text.todense().T, index=vect.get_feature_names(), columns=[f'{i+1}' for i in range (len(df))])
df


# In[19]:


idf=vect.idf_


# Dari hasil di atas dapat dilihat kata-kata yang paling sering muncul dan langka pada document berdasarkan skor idf. Semakin rendah nilainya berarti kata tersebut jarang muncul di document, sedangkan semakin tinggi nilainya menandakan bahwa kata tersebut sering muncul pada document.

# ## Modelling Latent Semantic Analysis (LSA)

# LSA pada dasarnya adalah dekomposisi nilai tunggal.
# 
# SVD menguraikan DTM asli menjadi tiga matriks S=U.(sigma). (V.T). Di sini matriks U menunjukkan matriks dokumen-topik sedangkan (V) adalah matriks topik-istilah.
# 
# Setiap baris matriks U (document-term matrix) adalah representasi vektor dari dokumen yang sesuai. Panjang vektor ini adalah jumlah topik yang diinginkan. Representasi vektor untuk istilah-istilah dalam data kami dapat ditemukan dalam matriks V (matriks term-topic).
# 
# Jadi, SVD memberi kita vektor untuk setiap dokumen dan istilah dalam data kita. Panjang setiap vektor adalah k. Kita kemudian dapat menggunakan vektor-vektor ini untuk menemukan kata-kata serupa dan dokumen serupa menggunakan metode kesamaan kosinus.
# 
# Kita dapat menggunakan fungsi truncatedSVD untuk mengimplementasikan LSA. Parameter n_components adalah jumlah topik yang ingin kami ekstrak. Model kemudian cocok dan diubah pada hasil yang diberikan oleh vectorizer.
# 
# Terakhir perhatikan bahwa LSA dan LSI (I untuk pengindeksan) adalah sama dan yang lebih baru hanya kadang-kadang digunakan dalam konteks pengambilan informasi.
# 
# Singular Value Decomposition (SVD) adalah teknik reduksi dimensi yang bermanfaat untuk memperkecil nilai kompleksitas dalam pemrogresan term-document matrix. rumus SVD yaitu:
# 
# $$
# \begin{array}{ll}
# A_{m n}=U_{m m} x S_{m n} x V_{n n}^{T} \\\\\\
# \mathrm{~A}_{m n}= { matriks awal } \\
# \mathrm{U}_{m m}= { matriks ortogonal U } \\
# \mathrm{S}_{m n}= { matriks diagonal S } \\
# \mathrm{V}_{n n}^{\top}= { transpose matriks ortogonal } \mathrm{V}
# \end{array}
# $$

# In[20]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[21]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[22]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)


# Similalry dengan dokumen lain kita bisa melakukan ini. Namun perlu perhatikan bahwa nilai tidak ditambahkan ke 1 karena dalam LSA itu bukan probabiltiy dari suatu topik dalam dokumen.

# In[23]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# Untuk memperoleh kata-kata penting pada tiap dokument, yaitu dengan menggunakan code di bawah ini.

# In[24]:


# most important words for each topic
vocab = vect.get_feature_names_out()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[ ]:





# In[ ]:




