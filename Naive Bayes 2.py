import pandas as pd
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

#Survived== target Pclass Sex Age Fare
df = pd.read_csv("spam_ham_dataset.csv")
df = df.set_axis(['Unnamed', 'label', 'text', 'label_num'], axis=1, inplace=False)
classes=df['label_num']
df = df.drop(['Unnamed','label'], axis='columns')


X_train,X_test,Y_train,Y_test=train_test_split(df.text,df.label_num,test_size=0.25)

vectorizer=CountVectorizer()
X_train_count=vectorizer.fit_transform(X_train)
model=MultinomialNB()
model.fit(X_train_count,Y_train)
emails=["ho ho ho , we ' re around to that most wonderful time of the year - - - neon leaders retreat time !"]
X_test_count=vectorizer.fit_transform(emails)
model.predict(X_test_count)
#score=model.score(X_test_count,Y_test)
"""score=model.score(X_test_count,Y_test)
print("Score:",score)"""
"""prediction=model.predict(X_test_count)
print("Predictions:",prediction)
print("Real:",Y_test.values.tolist())
print("Score:",score)"""

"""le_pclass=LabelEncoder()
le_sex=LabelEncoder()

df['Class']=le_pclass.fit_transform(df['Pclass'])
df['Gender']=le_sex.fit_transform(df['Sex'])"""