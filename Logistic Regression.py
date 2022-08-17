import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv("insurance_data.csv")
df.bought_insurance=df.bought_insurance.fillna(math.floor(0))
plt.scatter(df.age,df.bought_insurance,marker='+',color="red")
X_train,X_test,Y_train,Y_test=train_test_split(df[['age']],df.bought_insurance,test_size=0.1)

model = LogisticRegression()
model.fit(X_train,Y_train)
score=model.score(X_train,Y_train)
prediction = model.predict(X_test)
plt.plot(X_test,Y_test,color="blue")
print("Predictions:",prediction)
print("Real:",Y_test.values.tolist())
print("Score:",score)
plt.show()