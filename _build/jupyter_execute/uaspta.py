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

# import scrapy
# 
# 
# class QuotesSpider(scrapy.Spider):
#     name = "quotes"
# 
#     def start_requests(self):
#         urls = [
#             'https://pta.trunojoyo.ac.id/welcome/detail/080211100070',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080111100002',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080211100050',
#             'https://pta.trunojoyo.ac.id/welcome/detail/090111100077',
#             'https://pta.trunojoyo.ac.id/welcome/detail/070111100093',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080111100029',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080111100011',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080111100023',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080111100007',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080111100047',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080111100082',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080111100054',
#             'https://pta.trunojoyo.ac.id/welcome/detail/090111100053',
#             'https://pta.trunojoyo.ac.id/welcome/detail/090111100003',
#             'https://pta.trunojoyo.ac.id/welcome/detail/090111100068',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080111100010',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080111100031',
#             'https://pta.trunojoyo.ac.id/welcome/detail/080111100084',
#             'https://pta.trunojoyo.ac.id/welcome/detail/050111100448'
#         ]
#         for url in urls:
#             yield scrapy.Request(url=url, callback=self.parse)
# 
#     def parse(self, response):
#         # print(response.url)
#         yield {
#             'judul': response.css('#content_journal > ul > li > div:nth-child(2) > a::text').extract(),
#             'penulis': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(2) > span::text').extract(),
#             'dosen_pembimbing_1': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(3) > span::text').extract(),
#             'dosen_pembimbing_2': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(4) > span::text').extract(),
#             'abstrak': response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(2) > p::text').extract(),
#         }
#         # content_journal > ul > li:nth-child(1) > div:nth-child(1) > a
#         # content_journal > ul > li:nth-child(1) > div:nth-child(1) > a
# 

# # 2. Pre-Processing Data

# Pre-Processing Data merupakan suatu proses mengubah data mentah menjadi bentuk data yang lebih mudah dipahami. Proses dilakukan untuk membersihkan data dari angka, tanda baca, whitespace, dan kata yang tidak penting untuk digunakan. Tahapan dalam pre-processing data yaitu:

# ## Cleansing Data
# 

# Cleaning data atau membersihkan data artinya mengeksekusi ulang data yang telah di peroleh, seperti menghapus atau menghilangkan data-data yang tidak lengkap, tidak relavan, dan tidak akurat. Langkah pertama yang dilakukan untuk membulai proses pre-processing data adalah dengan mengimport modul-module yang diperlukan untuk proses pre-processing.

# ###  1. Import Module

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


# ### 2. Load Dataset

# setelah proses import module yang digunakan, langkah kedua pada proses cleaning data adalah Load Data. Load Data adalah proses membuka/membaca file dari data yang akan digunakan untuk proses Pre-Processing, data yang digunakan biasanya berbentuk file csv, json dan lainnya. disini saya menggunakan format file csv.

# In[87]:


df=pd.read_csv('pta2.csv')


# In[88]:


df.head()


# df.head() adalah proses untuk menampilkan data awal/teratas yang terdapat a pada file. Jika pada df.head() tidak diberikan parameter angka, hasil yang akan ditampilkan adalah 5 data teratas yang ada pada file.

# ### 3. Drop data yang tidak diperlukan

# Drop data adalah proses penghapusan data pada tabel data yang tidak diperlukan dalam proses pre-processing data. Pada tahap ini data yang dihapus adalah data yang ada pada tabel judul, waktu, dan category karena data yang akan digunakan adalah data yang ada pada tabel isi.

# In[89]:


# drop the publish date.
df.drop(['judul','penulis'],axis=1, inplace=True)


# tahap setelah penghapusan data yaitu menampilkan hasil data yang telah di proses dengan menggunakan df.head(). pada bagian ini terdapat nilai 30 sebagai parameter dari data yang akan ditampilkan, ini berarti banyaknya data yang akan di tampilkan adalah 30 data dimulai dari hitungan index ke 0.

# In[90]:


df.head(20)


# 
# 
# ### 4. Clean Data & Processing Data

# 
# Proses ini adalah proses untuk membersihkan data dari angka-angka

# In[91]:


import string 
import re #regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

df['hapus angka'] = df['abstrak'].apply(remove_number)
df.head(10)


# In[92]:


def remove_all(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
df['remov all'] = df['hapus angka'].apply(remove_all)
df.head(10)


# In[93]:


#mengubah huruf menjadi kecil dengan menggunakan Series.str.lower() pada pandas

df['remov all'] = df['remov all'].str.lower()

print (df['remov all'])


# ## Stopword

# Proses selanjutnya adalah proses Stopword. Stopword merupakan proses mengabaikan kata-kata yang perlu di abaikan dalam sebuah dokumen dan memilih kata-kata penting yang mewakili suatu dokumen.

# In[94]:


def clean_text(headline):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(headline)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text


# In[95]:


# time taking
#nltk.download('wordnet')
df['clean_text_isi']=df['hapus angka'].apply(clean_text)


# In[96]:


df.head()


# setelah Stopword, proses selanjutnya sama dengan proses sebelumnya yaitu menghapus data pada tabel yang tidak diperlukan dan hanya menyisakan data pada tabel hasil dari proses sebelumnya.

# In[97]:


df.drop(['abstrak','hapus angka', 'clean_text_isi'],axis=1,inplace=True)


# In[98]:


df


# Untuk menampilkan isi dari suatu index data pada tabel dapat dilakukan dengan menggunakan code berikut:

# In[99]:


df['remov all'][10]


# In[105]:


df.to_csv("abstrak_pre-processing")


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

# In[112]:


import pandas as pd
import numpy as np

df = pd.read_csv("abstrak_pre-processing", usecols=['remov all'])
df.columns = ['hasil-akhir']

df.head()


# In[114]:


from sklearn.feature_extraction.text import CountVectorizer

document = df['hasil-akhir']
a=len(document)

#create a vectorizer object
vectorizer = CountVectorizer()

vectorizer.fit(document)
# Printing the identified Unique words along with their indices
print("Vocabulary: ", vectorizer.vocabulary_)


# In[117]:


# Encode the Document
vector = vectorizer.transform(document)

# Summarizing the Encoded Texts
print("Encoded Document is:")
vector.toarray()


# In[116]:


a = vectorizer.get_feature_names()

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
tf = tfidf.fit_transform(vectorizer.fit_transform(document)).toarray()

dfb = pd.DataFrame(data=tf,index=list(range(1, len(tf[:,1])+1, )),columns=[a])
dfb

dfb.to_csv("hasil-paling-akhir.csv")


# # K-Means

# In[80]:


from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[118]:


#--- Mengubah Variabel Data Frame Menjadi Array ---
x_array =  np.array(dfb)
print(x_array)


# In[81]:


scaler = StandardScaler()
scaler.fit(df)
df_scaled = scaler.transform(df)


# In[82]:


print (df_scaled)


# In[ ]:





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

# In[75]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[76]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[77]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)


# Similalry dengan dokumen lain kita bisa melakukan ini. Namun perlu perhatikan bahwa nilai tidak ditambahkan ke 1 karena dalam LSA itu bukan probabiltiy dari suatu topik dalam dokumen.

# In[78]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# Untuk memperoleh kata-kata penting pada tiap dokument, yaitu dengan menggunakan code di bawah ini.

# In[79]:


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




