# -*- coding: utf-8 -*-

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20191221"

import pandas as pd
from snownlp import SnowNLP
from textblob import TextBlob
import matplotlib.pyplot as plt
import jieba
import datetime
pd.set_option('display.max_colwidth', -1)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn import preprocessing
tokenizer = RegexpTokenizer('[a-zA-Z]\w+\'?\w*')
path =r'C://Users//DELL//Desktop//NOAH//short_term_rental_dataset//'
reviews_detail = pd.read_csv(path+'reviews_detail.csv')
reviews_detail = reviews_detail.loc[:10].copy()


en_stop_words = set(stopwords.words('english'))
extended_stop_words = en_stop_words | \
                      {
                          'place;', 'host',
                          'stay','beijing', 'location', 'apartment'
                          'the','and'
                      }

ch_stop_words = set(stopwords.words('chinese'))
extended_stop_words2 = ch_stop_words | \
                      {
                          u'的', u'，',u'和', u'是', u'随着', u'对于', u'对',u'等',u'能',u'都',u'。',u' ',u'、',u'中',u'在',u'了',
                          u'通常',u'如果',u'我们',u'需要',u'很',u'也',u'就是',u'非常',u'可以',u'很多',u'就',u'挺',
                          u'到',u'会',u'房东'
                      }

if  u'\u4e00' <= reviews_detail['comments'] <= u'\u9fff':
    reviews_detail['language'] = 'chinese'
else:
    reviews_detail['language'] = 'english'

for i in range(1, reviews_detail.shape[0]):
    if  u'\u4e00' <= reviews_detail['comments'] <= u'\u9fff':
        reviews_detail.at[reviews_detail.index[i],'score'] = SnowNLP(''.join(
            jieba.analyse.textrank(i))).sentiments
            #[word for word in i.split() if (not word in extended_stop_words2)])).sentiments

    else:
        reviews_detail.at[reviews_detail.index[i],'score'] = TextBlob(tokenizer.tokenize(' '.join(
            [word for word in i.lower().split() if (not word in extended_stop_words)])).sentiment.polarity)


min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))

reviews_detail.loc[reviews_detail['language']=='chinese', 'score'] = \
    min_max_scaler.fit_transform(reviews_detail.loc[
                                           reviews_detail['language'] == 'chinese', 'score'] )

reviews_detail.loc[reviews_detail['language']=='english', 'score'] = \
    min_max_scaler.fit_transform(reviews_detail.loc[
                                           reviews_detail['language']=='english', 'score'] )

reviews_detail.to_csv(path+'reviews_detail_adj.csv')

plt.hist(reviews_detail['score'])
plt.show()
