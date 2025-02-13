import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from svd import svd_eig

bbc_data = pd.read_csv('./data/bbc-news-data.csv', sep = '\t')

category = bbc_data.category.tolist()
documents = bbc_data.content.tolist()

vectorizer = TfidfVectorizer(stop_words = 'english')
X = vectorizer.fit_transform(documents)

print(f"Document-term matrix: {X.shape}")

k = 2
U, sigma, V = svd_eig(X.T, k=k)
print(f"U: {U.shape}, sigma: {sigma.shape}, V: {V.shape}")

num_clusters = 5
kmeans = KMeans(n_clusters = num_clusters)
kmeans.fit(V)

plt.figure(figsize = (10,8))
sns.scatterplot(x = V[:,0], y = V[:,1], hue = category)
plt.show()