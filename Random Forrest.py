from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

digits=load_digits()

plt.gray()

"""for i in range(4):
    plt.matshow(digits.images[i])"""


features=pd.DataFrame(digits.data)
classes=pd.DataFrame(digits.target)
classes = classes.set_axis(['target'], axis=1, inplace=False)
df = pd.concat([features,classes], axis='columns')
X = features
Y=classes
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

model=RandomForestClassifier(n_estimators=5)
model.fit(X_train,Y_train)
score=model.score(X_test,Y_test)
prediction=model.predict(X_test)
print("Predictions:",prediction)
print("Real:",Y_test.values.tolist())
print("Score:",score)


cm=confusion_matrix(Y_test,prediction)
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()