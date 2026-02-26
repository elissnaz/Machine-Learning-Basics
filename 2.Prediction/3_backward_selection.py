# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 14:30:29 2026

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#COKLU DEĞİSKEN ICIN VERİ HAZIRLAMA

veriler = pd.read_csv('data.csv')
print(veriler)

ulke = veriler.iloc[:,0:1].values
Yas =veriler.iloc[:,1:4].values

from sklearn import preprocessing
le= preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe=preprocessing.OneHotEncoder()
ulke= ohe.fit_transform(ulke).toarray()
print(ulke)

c = veriler.iloc[:,-1:].values

le= preprocessing.LabelEncoder()
c[:,-1]=le.fit_transform(veriler.iloc[:,-1])
print(c)

ohe=preprocessing.OneHotEncoder()
c= ohe.fit_transform(c).toarray()
print(c)



sonuc= pd.DataFrame(data=ulke, index=range(22),columns=['fr','tr','us'])
print(sonuc)   

sonuc2= pd.DataFrame(data=Yas, index=range(22),columns=['boy','kilo','yas'])
print(sonuc2) 

sonuc3= pd.DataFrame(data=c[:,:1], index=range(22),columns=['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1)#0 olsa alt alta eklerdi
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#COKLU LINEER REGRESYON
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)

y_pred= regressor.predict(x_test)

#veri manipüle s2 de boy sutununu ayırdım
boy= s2.iloc[:,3:4].values
sol= s2.iloc[:,:3]
sag= s2.iloc[:,4:]

veri= pd.concat([sol,sag],axis=1)

x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)

r2= LinearRegression()
r2.fit(x_train, y_train)
y_pred=r2.predict(x_test)




import statsmodels.api as sm

X= np.append(arr= np.ones((22,1)).astype(int), values= veri, axis=1)
#1’lerden oluşan bir kolon üretir.
#Bu kolon sabit terim (intercept / bias) için eklenir.

X_l= veri.iloc[:,[0,1,3,4,5]].values
X_l= np.array(X_l,dtype=float)
model= sm.OLS(boy,X_l).fit()
print(model.summary())

