from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

digits=load_digits()

plt.gray()

"""for i in range(4):
    plt.matshow(digits.images[i])"""


features=digits.data
classes=digits.target


print(cross_val_score(LogisticRegression(max_iter=8000),features,classes))
print(cross_val_score(SVC(),features,classes))
print(cross_val_score(RandomForestClassifier(n_estimators=50),features,classes))
