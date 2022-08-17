import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv("homeprices.csv")
dummies = pd.get_dummies(df.town)
merged = pd.concat([df, dummies], axis='columns')
final = merged.drop(['town', 'west windsor'], axis='columns')
X = final.drop('price', axis='columns')
Y=final.price
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)

print(df)
# price=m1*area+m2*bedrooms+m3*age+b   m:slope  b: intercept

model = LinearRegression()
model.fit(X_train,Y_train)
score=model.score(X_train,Y_train)
prediction = model.predict(X_test)
print("Predictions:",prediction)
print("Real:",Y_test.values.tolist())
print("Score:",score)

