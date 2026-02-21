# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 21:21:21 2026

@author: USER
"""

import pandas as pd

#NOMINAL VERİYİ NUMERİK VERİ HALİNE GETİRME
veriler= pd.read_csv('data.csv')
ulke= veriler.iloc[:,0:1].values
from sklearn import preprocessing

#her kategoriye bir sayı verir
#ancak model 2>1>0 yani mavi>yesıl>sarı gibi sıralama yapar
#bu yuzden cogu model için risklidir
le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)

#her kategori için ayrı sütun açar
#sıralama algısı yok
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)


#Tree tabanlı modeller (Decision Tree, Random Forest)
#LabelEncoder’dan çok etkilenmez.

#Ama Linear Regression, Logistic Regression gibi modeller
#yanlış öğrenebilir.