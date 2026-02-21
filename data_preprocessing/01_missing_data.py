# -*- coding: utf-8 -*-
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt

veriler= pd.read_csv('missingdata.csv')
#print(veriler)
#boy=veriler[['boy']]
#boykilo=veriler[['boy','kilo']]
#print(boykilo)

#eksik veriler
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
Yas= veriler.iloc[:,1:4].values
#print(Yas)
imputer=imputer.fit(Yas[:,1:4])  #veriyi ogrenir(fit)
Yas[:,1:4]=imputer.transform(Yas[:,1:4]) #ogrendiği bilgiyi kullanır (transform)
print(Yas)

#ÖNEMLİİİİ
#scaler.fit(X_train)      # sadece train'i öğren
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)  # test'e sadece transform

#Test verisine fit yaparsan veri sızıntısı (DATA LEAKAGE) olur!
#fit() verinin ortalamasını hesaplar,Standart sapmasını hesaplar
#Min ve max’ını öğrenir Test verisinin ortalamasını öğrenmiş olursun.
#Yani test verisinden bilgi çalmış olursun.bu da data leakage

