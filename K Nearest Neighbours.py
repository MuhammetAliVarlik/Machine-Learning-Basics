import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn

iris = load_iris()
# print(iris.data,iris.target,iris.target_names)
features=pd.DataFrame(iris.data)
classes=pd.DataFrame(iris.target)
df = pd.concat([features,classes], axis='columns')
df = df.set_axis(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'], axis=1, inplace=False)
"""df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color="green", marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color="blue", marker='.')

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color="green", marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color="blue", marker='.')
"""
X = df.drop(['target'], axis='columns')
Y=df.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train, Y_train)
score = model.score(X_test, Y_test)
prediction = model.predict([[4.8, 3.0, 1.5, 0.3]])
print("Predictions:", prediction)
print("Real:", Y_test.values.tolist())
print("Score:", score)


y_pred = model.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()