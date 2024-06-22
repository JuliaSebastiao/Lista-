import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
import seaborn as sns

# Carregar os dados
iris_data = pd.read_csv('Iris.csv')

print(iris_data.head())

# Identificação de outliers
plt.figure(figsize=(12, 8))
sns.boxplot(data=iris_data.drop('class', axis=1))
plt.title('Boxplots of Iris Dataset Features')
plt.show()

# Normalização dos dados
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_data.drop('class', axis=1))

# Elbow Method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(iris_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()

# KMeans com 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(iris_scaled)

# Avaliação com Silhouette Score e Calinski-Harabasz Score
silhouette_avg = silhouette_score(iris_scaled, clusters)
calinski_harabasz = calinski_harabasz_score(iris_scaled, clusters)
print(f'Silhouette Score: {silhouette_avg:.4f}')
print(f'Calinski-Harabasz Score: {calinski_harabasz:.4f}')

# Visualização dos clusters
plt.figure(figsize=(10, 6))
plt.scatter(iris_scaled[:, 0], iris_scaled[:, 1], c=clusters, cmap='viridis', edgecolor='k', s=50)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
plt.title('Visualization of Clustered Data')
plt.xlabel('Normalized Sepal Length')
plt.ylabel('Normalized Sepal Width')
plt.show()

# Métrica Davies-Bouldin
davies_bouldin = davies_bouldin_score(iris_scaled, clusters)
print(f'Davies-Bouldin Score: {davies_bouldin:.4f}')


# Adicionando os clusters ao dataframe original
iris_data['cluster'] = clusters

# Mapeando as classes para números
class_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
iris_data['class_num'] = iris_data['class'].map(class_mapping)

# Visualizando os agrupamentos incorretos
incorrect = iris_data[iris_data['cluster'] != iris_data['class_num']]

plt.figure(figsize=(10, 6))
sns.scatterplot(x=iris_data['sepallength'], y=iris_data['sepalwidth'], hue=iris_data['class'], style=iris_data['cluster'], palette='viridis')
plt.title('Incorrectly Clustered Instances')
plt.show()