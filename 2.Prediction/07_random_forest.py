# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 20:05:51 2026

@author: USER
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#veri yukleme
veriler=pd.read_csv('salaries.csv')

#data frame slice
x= veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

#numpy array transformation
X= x.values
Y= y.values


#Linear regression
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(X, lin_reg.predict(X),color='blue')
plt.show()


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=4)
x_poly= poly_reg.fit_transform(X)


lin_reg2= LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()



#veri ölçekleme
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_scaled = sc1.fit_transform(X)
sc2= StandardScaler()
y_scaled= sc2.fit_transform(Y.reshape(-1,1))



#Support vector regression,SVR
from sklearn.svm import SVR
svr_reg=SVR(kernel='rbf')
#eğrinin şeklini öğrenmesini saglar(rbf) kernel fonk.
svr_reg.fit(x_scaled,y_scaled)

plt.scatter(x_scaled, y_scaled,color='red')
plt.plot(x_scaled, svr_reg.predict(x_scaled),color='blue')
plt.show()

print(svr_reg.predict(sc1.transform([[11]])))
print(svr_reg.predict(sc1.transform([[6.6]])))



#karar ağaçlarında ölçeklendirmeye gerek yok 
#fit ederken de numpy Array olan X,Y kullanılır
from sklearn.tree import DecisionTreeRegressor
r_dt= DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

plt.scatter(X, Y, color='red')
plt.plot(X, r_dt.predict(X), color='blue')
plt.show()

print(r_dt.predict(sc1.transform([[11]])))
print(r_dt.predict(sc1.transform([[6.6]])))


#RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
#ravel arrayi tek boyutlu hale getirir
#random forest boyut hatası verebiliyor ondan kullandık
rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.5]]))

plt.scatter(X, Y, color='red')
plt.plot(X,rf_reg.predict(X),color='blue')















