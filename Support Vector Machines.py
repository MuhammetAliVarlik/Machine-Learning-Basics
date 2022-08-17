import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd

pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

# import some data to play with
digits = load_digits()
# print(iris.data,iris.target,iris.target_names)
features=pd.DataFrame(digits.data)
classes=pd.DataFrame(digits.target)
classes = classes.set_axis(['target'], axis=1, inplace=False)
df = pd.concat([features,classes], axis='columns')
X = features
Y=classes
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
model=SVC()
model.fit(X_train,Y_train)
score=model.score(X_test,Y_test)
prediction=model.predict(X_test)
print("Predictions:",prediction)
print("Real:",Y_test.values.tolist())
print("Score:",score)