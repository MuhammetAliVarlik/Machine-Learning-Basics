import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier,export_graphviz

pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

#Survived== target Pclass Sex Age Fare
df = pd.read_csv("titanic.csv")
le_pclass=LabelEncoder()
le_sex=LabelEncoder()

df['Class']=le_pclass.fit_transform(df['Pclass'])
df['Gender']=le_sex.fit_transform(df['Sex'])

median_Age=math.floor(df.Age.median())
df.Age=df.Age.fillna(median_Age)

df = df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked','Pclass','Sex'], axis='columns')
X=df.drop('Survived', axis='columns')
Y=df.Survived

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
model=DecisionTreeClassifier(criterion='entrophy',random_state='0')
model.fit(X_train,Y_train)
score=model.score(X_test,Y_test)
prediction=model.predict(X_test)
print("Predictions:",prediction)
print("Real:",Y_test.values.tolist())
print("Score:",score)
