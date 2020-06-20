import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
import re
import sys
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

data = pd.read_excel("D:/dataguna.xlsx")
text = data['text']
print(data)

#Case Folding
datalower = []
for line in text:
  a = line.lower()
  datalower.append(a)
    
#Cleaning Number
dataclearangka = []
for line in datalower:
    result = re.sub("\d"," ", line)
    dataclearangka.append(result)
    print(result)
    
#Cleaning Emoticon
dataclearemoticon = []
for line in dataclearangka:
    result = re.sub(r'<.*?>'," ", line)
    dataclearemoticon.append(result)
    print(result)
    
#Cleaning Punctuation
dataclearpuntuation = []
for line in dataclearemoticon:
    result = re.sub(r"[^\w\s]"," ", line)
    dataclearpuntuation.append(result)
    print(result)

#Stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()
data_stemmed = map(lambda x: stemmer.stem(x), datalower)
databersih = map(lambda x: x.translate(str.maketrans('','', string.punctuation)), data_stemmed)
databersih = list(databersih)

#Stopwords and Tokenizing
stopwords = open('D:/stopword.txt', 'r').read()
gunadata = []
gunafinal = []
df = []

for line in databersih:
   word_token=nltk.word_tokenize(line)
   word_token=[word for word in word_token if not word in stopwords and not word[0].isdigit()]
   gunafinal.append(word_token)
   df.append(" ".join(word_token))
for l in gunafinal:
    gunadata+= l
final_guna={v: gunadata.count(v) for v in set(gunadata)}

#Count Vectorize
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer

Y = data['klasifikasi']
Y_A = pd.DataFrame(Y)

vectorizer = CountVectorizer(min_df=4)
X = vectorizer.fit_transform(df)
X_ = DataFrame(X.A,columns=vectorizer.get_feature_names())

#TF-IDF
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True).fit_transform(X_)
tfidf_nya = (tfidf.toarray())

X_nya = tfidf_nya
tf=DataFrame(tfidf.A, columns= vectorizer.get_feature_names())
