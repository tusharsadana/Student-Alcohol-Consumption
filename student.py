# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:00:16 2017

@author: Tushar
"""

import pandas as pd

dataset = pd.read_csv('st.csv')
dl = range(1,26)
dl.remove(10)
for i in range(29,33):
    dl.append(i)
il = [26,27,28]

x= dataset.iloc[:,dl].values
y= dataset.iloc[:, il ].values

newdata = pd.DataFrame(x)
newdata[0] = newdata[0].map(lambda r:1 if r=='M' else 0)
newdata[2] = newdata[2].map(lambda r:1 if r=='U' else 0)
newdata[3] = newdata[3].map(lambda r:1 if r=='GT3' else 0)
newdata[4] = newdata[4].map(lambda r:1 if r=='T' else 0)
for i in range(13,21):
    newdata[i] = newdata[i].map(lambda r:1 if r=='yes' else 0)
    
newdata[9] = newdata[9].map(lambda r:1 if r=='father' else 0)

x= newdata.iloc[:,:].values
               
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
x[:,7] = labelencoder_X.fit_transform(x[:,7])
x[:,8] = labelencoder_X.fit_transform(x[:,8])

check = pd.DataFrame(x)
               
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [7])
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:]

onehotencoder = OneHotEncoder(categorical_features = [11])
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:]

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, random_state = 0)





