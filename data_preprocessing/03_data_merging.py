# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 21:57:34 2026

@author: USER
"""

import pandas as pd

veriler= pd.read_csv('data.csv')
ulke=veriler.iloc[:,0:1].values
Yas =veriler.iloc[:,1:4].values
cinsiyet= veriler.iloc[:,-1].values

from sklearn import preprocessing
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()

#DataFrame'i model için gerekli parçaları seçmek için kullanıyorsun.
sonuc= pd.DataFrame(data=ulke, index=range(22),columns=['fr','tr','us'])
print(sonuc)   

sonuc2= pd.DataFrame(data=Yas, index=range(22),columns=['boy','kilo','yas'])
print(sonuc2) 

sonuc3= pd.DataFrame(data=cinsiyet, index=range(22),columns=['cinsiyet'])
print(sonuc3)

#VERİLERİ BİRLEŞTİRME
s=pd.concat([sonuc,sonuc2],axis=1)#0 olsa alt alta eklerdi
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)
