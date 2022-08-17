"""
Author:Muhammet Ali Varlık
Date:Tuesday,16 August 2022
Version:1.0
Home prices guessing with Linear Regression in Monroe Twp,NJ (USA)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

net_price = np.array([12178, 11085, 10964, 10696, 9793,9208,8597])
year = np.array([2014, 2015, 2016, 2017, 2018,2019,2020]).reshape((-1,1))

# net_price=m*year+b   m:slope  b: intercept

plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')
plt.scatter(year, net_price, color='red',marker='+')

reg=LinearRegression()
model=reg.fit(year,net_price)
# Predict a Response and print it:
y_pred = model.predict(year)
print("In Turkey,predicted net price at 2021 is about",int(model.predict(np.array(2021).reshape((-1,1)))[0]),"₺.")
plt.plot(year,y_pred,color="blue")
plt.show()