from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import seaborn as sns 
from sklearn.datasets import load_iris
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target 
df['flower_name'] = df['target'].apply(lambda x: iris.target_names[x])
print(df.head())

#Drop unnecessary columns to conduct Unsupervised ML
df = df.drop(['sepal length (cm)', 'sepal width (cm)', 'target', 'flower_name'], axis='columns')


# Plot to detect clusters and patterns
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)')
#plt.show()

# Create three clusters
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['petal length (cm)', 'petal width (cm)']])
print(y_predicted)

df['cluster'] = y_predicted
print(df.head())

print(km.cluster_centers_)

# Plot different clusters
df_cluster1 = df[df['cluster']==0]
df_cluster2 = df[df['cluster']==1]
df_cluster3 = df[df['cluster']==2]

plt.scatter(df_cluster1['petal length (cm)'], df_cluster1['petal width (cm)'], c='red')
plt.scatter(df_cluster2['petal length (cm)'], df_cluster2['petal width (cm)'], c='blue')
plt.scatter(df_cluster3['petal length (cm)'], df_cluster3['petal width (cm)'], c='green')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], marker='*', c='purple', label='Centroids')
plt.legend()
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Unsupervised ML')
plt.show()

# Preprocessing using min max scaler
# scaler = MinMaxScaler()
# scaler.fit(df[['petal length (cm)']])
# df['petal length (cm)'] = scaler.transform(df[['petal length (cm)']])
# scaler.fit(df[['petal width (cm)']])
# df['petal width (cm)'] = scaler.transform(df[['petal width (cm)']])
# # plt.scatter(df['petal length (cm)'],df['petal width (cm)'])
# # plt.show()
# km = KMeans(n_clusters=3)
# y_predicted = km.fit_predict(df[['petal length (cm)', 'petal width (cm)']])
# print(y_predicted)



# Elbow plot - plotting a Loss function of SSE
sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit_predict(df[['petal length (cm)', 'petal width (cm)']])
    sse.append(km.inertia_)

plt.title('Loss Function')    
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()
