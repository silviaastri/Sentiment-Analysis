#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
import re
import sys
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

data = pd.read_excel("D:/dataguna.xlsx")
Review = data['text']
print(data)

#Split Data

data_pos = data [ data['klasifikasi'] == 1 ]
data_pos = data_pos ['text']

data_neg = data [ data['klasifikasi'] == 0 ]
data_neg = data_neg ['text']

#Case Folding
datalower_pos=[]
for line in data_pos:
 a=line.lower()
 datalower_pos.append(a)

datalower_neg=[]
for line in data_neg:
 a=line.lower()
 datalower_neg.append(a)
    
#Cleaning Number
dataclearnumber_pos = []
for line in datalower_pos:
    result = re.sub("\d"," ", line)
    dataclearnumber_pos.append(result)

dataclearnumber_neg = []
for line in datalower_neg:
    result = re.sub("\d"," ", line)
    dataclearnumber_neg.append(result)
    
#Cleaning Emoticon
dataclearemoticon_pos = []
for line in dataclearnumber_pos:
    result = re.sub(r'<.*?>'," ", line)
    dataclearemoticon_pos.append(result)

dataclearemoticon_neg = []
for line in dataclearnumber_neg:
    result = re.sub(r'<.*?>'," ", line)
    dataclearemoticon_neg.append(result)

#Cleaning Punctuation
dataclearpuntuation_pos = []
for line in dataclearemoticon_pos:
    result = re.sub(r"[^\w\s]"," ", line)
    dataclearpuntuation_pos.append(result)

dataclearpuntuation_neg = []
for line in dataclearemoticon_neg:
    result = re.sub(r"[^\w\s]"," ", line)
    dataclearpuntuation_neg.append(result)

#Stemming
factory=StemmerFactory()
stemmer=factory.create_stemmer()
datastemmed_pos=map(lambda x: stemmer.stem(x), datalower_pos)
dataclean_pos=map(lambda x: x.translate(str.maketrans('', '',string.punctuation)), datastemmed_pos)
dataclean_pos=list(dataclean_pos)

factory=StemmerFactory()
stemmer=factory.create_stemmer()
datastemmed_neg=map(lambda x: stemmer.stem(x), datalower_neg)
dataclean_neg=map(lambda x: x.translate(str.maketrans('', '',string.punctuation)), datastemmed_neg)
dataclean_neg=list(dataclean_neg)

#Stopwords and Tokenizing
stopwords = open('D:/stopword.txt', 'r').read()

gunadata_pos = []
gunafinal_pos = []
df_pos = []

for line in dataclean_pos:
   wt_pos = word_tokenize(line)
   wt_pos = [word for word in wt_pos if not word in stopwords and not word[0].isdigit()]
   gunafinal_pos.append(wt_pos)
   df_pos.append(" ".join(wt_pos))
for l in gunafinal_pos:
    gunadata_pos+= l
final_pos={v: gunadata_pos.count(v) for v in set(gunadata_pos)}

gunadata_neg = []
gunafinal_neg = []
df_neg = []

for line in dataclean_neg:
   wt_neg = word_tokenize(line)
   wt_neg = [word for word in wt_neg if not word in stopwords and not word[0].isdigit()]
   gunafinal_neg.append(wt_neg)
   df_neg.append(" ".join(wt_neg))
for l in gunafinal_neg:
    gunadata_neg+= l
final_neg={v: gunadata_neg.count(v) for v in set(gunadata_neg)}

#Change Data into 'str'
a = str(df_pos)
positif =re.sub(r"'","",a)

b = str(df_neg)
negatif =re.sub(r"'","",b)

#Word Cloud
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
mpl.rcParams['font.size']=12 #10
mpl.rcParams['savefig.dpi']=100 #72
mpl.rcParams['figure.subplot.bottom']=.1

#Word Cloud Positive
wordcloud = WordCloud(collocations = False,
                     background_color='white',
                     stopwords=stopwords,
                     max_words=50,
                     max_font_size=200,
                     random_state=42
                     ).generate(positif)
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("D:/wordpos.png", dpi=900)

#Word Cloud Negative
wordcloud = WordCloud(collocations = False,
                     background_color='white',
                     stopwords=stopwords,
                     max_words=50,
                     max_font_size=200,
                     random_state=42
                     ).generate(negatif)
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("D:/wordneg.png", dpi=900)


# In[ ]:




