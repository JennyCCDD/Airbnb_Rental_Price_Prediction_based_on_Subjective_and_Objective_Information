# -*- coding: utf-8 -*-
import re
import collections
import numpy as np
import jieba
import wordcloud
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

path =r'C://Users//DELL//Desktop//NOAH//short_term_rental_dataset//'
reviews_detail = pd.read_csv(path+'reviews_detail.csv')
reviews_detail['comments'].to_csv(path+'comments.txt',sep='\t',index=False)

text=''
with open('comments.txt','r',encoding='utf-8' ) as f:
    string_data=f.read()
    f.close()

pattern = re.compile(u'\t|\n|\.|-|:|;|\)|\(|\?|"')
string_data = re.sub(pattern, '', string_data)

seg_list_exact = jieba.cut(string_data, cut_all = False)
object_list = []
remove_words = [u'的', u'，',u'和', u'是', u'随着', u'对于', u'对',u'等',u'能',u'都',u'。',u' ',u'、',u'中',u'在',u'了',
                u'通常',u'如果',u'我们',u'需要',u'很',u'也',u'就是',u'非常',u'可以',u'很多',u'就',u'挺',
                u'到',u'会',u'房东','the','and','place;', 'host',
                'stay','beijing', 'location', 'apartment']

for word in seg_list_exact:
    if word not in remove_words:
        object_list.append(word)


word_counts = collections.Counter(object_list)
word_counts_top10 = word_counts.most_common(10)
print (word_counts_top10)


wc = wordcloud.WordCloud(
    font_path='C:/Windows/Fonts/simhei.ttf',
    max_words=200,
    max_font_size=100)
wc.generate_from_frequencies(word_counts)
plt.imshow(wc)
plt.axis('off')
plt.show()