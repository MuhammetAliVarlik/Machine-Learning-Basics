import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# import some data to play with
iris = load_iris()
# print(iris.data,iris.target,iris.target_names)
features = pd.DataFrame(iris.data)
classes = pd.DataFrame(iris.target)
df = pd.concat([features, classes], axis='columns')
df = df.set_axis(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'], axis=1, inplace=False)
plt.scatter(df['petal_length'], df['petal_width'])
plt.show()
km = KMeans(n_clusters=3)
predict = km.fit_predict(df[['petal_length', 'petal_width']])
df['cluster'] = predict
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
plt.scatter(df1.petal_length, df1.petal_width, color='green')
plt.scatter(df2.petal_length, df2.petal_width, color='red')
plt.scatter(df3.petal_length, df3.petal_width, color='blue')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='yellow', marker='*', label='centroid')
plt.legend()
plt.show()
# Elbow Technique
k_rng = range(1, 10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['petal_length', 'petal_width']])
    sse.append(km.inertia_)
plt.plot(k_rng, sse)
plt.xlabel('K')
plt.ylabel('Sum of Squared error (SSE)')
plt.show()
