"""
Author:Muhammet Ali Varlık
Date:Tuesday,16 August 2022

Home prices guessing with Linear Regression in Monroe Twp,NJ (USA)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

area = np.array([2600, 3000, 3200, 3600, 4000]).reshape((-1,1))
price = np.array([550000, 565000, 610000, 680000, 725000])

# price=m*area+b   m:slope  b: intercept

plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')
plt.scatter(area, price, color='red',marker='+')

reg=LinearRegression()
print(area,price)
model=reg.fit(area,price)

# The following section will get results by interpreting the created instance:

# Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
r_sq = model.score(area, price)
print('coefficient of determination:', r_sq)

# Print the Intercept:
print('intercept:', model.intercept_)

# Print the Slope:
print('slope:', model.coef_)

# Predict a Response and print it:
y_pred = model.predict(area)
print('Predicted response:', y_pred, sep='\n')

plt.plot(area,y_pred,color="blue")
plt.show()