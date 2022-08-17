import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

# import some data to play with
iris = load_iris()
# print(iris.data,iris.target,iris.target_names)
features=pd.DataFrame(iris.data)
classes=pd.DataFrame(iris.target)
df = pd.concat([features,classes], axis='columns')
df = df.set_axis(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'], axis=1, inplace=False)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
pd.crosstab(df['sepal_length'],df['target']).plot(kind='bar',ax=ax1,stacked=True)
ax1.set_xlabel("sepal_length")

pd.crosstab(df['sepal_width'],df['target']).plot(kind='bar',ax=ax2,stacked=True)
ax2.set_xlabel("sepal_width")

pd.crosstab(df['petal_length'],df['target']).plot(kind='bar',ax=ax3,stacked=True)
ax2.set_xlabel("petal_length")

pd.crosstab(df['petal_width'],df['target']).plot(kind='bar',ax=ax4,stacked=True)
ax2.set_xlabel("petal_width")
plt.show()

# As we can see from here, we can classify irises using the petal length and the petal width.
X = df.drop(['sepal_width','sepal_length','target'], axis='columns')

Y=df.target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
model=LogisticRegression()
model.fit(X_train,Y_train)
score=model.score(X_test,Y_test)
prediction=model.predict(X_test)
print("Predictions:",prediction)
print("Real:",Y_test.values.tolist())
print("Score:",score)