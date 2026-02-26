# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 00:28:30 2026

@author: USER
"""

import pandas as pd
veriler=pd.read_csv('data.csv')
ulke=veriler.iloc[:,0:1].values
Yas=veriler.iloc[:,1:4].values
cinsiyet= veriler.iloc[:,-1].values

from sklearn import preprocessing
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=['boy','kilo','yas'])
sonuc=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
s=pd.concat([sonuc,sonuc2],axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(s, sonuc3, test_size=0.33,random_state=0)
#s(feature matris),sonuc3(hedef),%33 test %67train, randomstate sabitlik sağlar
#her run da aynı dondurur.
