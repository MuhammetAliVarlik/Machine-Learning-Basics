from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
digits=load_digits()

X_train,X_test,Y_train,Y_test=train_test_split(digits.data,digits.target,test_size=0.25)
model=LogisticRegression()
model.fit(X_train,Y_train)
score=model.score(X_test,Y_test)
prediction=model.predict(X_test[0].reshape(1,-1))
print(prediction)