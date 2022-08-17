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
import math

data = {
    'area': [2600, 3000, 3200, 3600, 4000],
    'price': [550000, 565000, 610000, 595000, 760000],
    'bedrooms': [3, 4, None, 3, 5],
    'age': [20, 15, 18, 30, 8],
}

df=pd.DataFrame(data)
print(df)
median_bedrooms=math.floor(df.bedrooms.median())

df.bedrooms=df.bedrooms.fillna(median_bedrooms)
print(df)

# price=m1*area+m2*bedrooms+m3*age+b   m:slope  b: intercept

reg=LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)

prediction=reg.predict([[3000,3,40]])
print(prediction[0])