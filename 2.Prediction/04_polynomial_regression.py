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
#It is still a linear regression model,
#but applied to transformed polynomial features

