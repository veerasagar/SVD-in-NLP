import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A dog is a man's best friend",
    "Cats and dogs are common household pets",
    "Cars and trucks contribute to environmental pollution",
    "Electric vehicles reduce air pollution"
]

# Step 1: Create TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)
print("Original TF-IDF Matrix Shape:", tfidf_matrix.shape)

# Step 2: Apply SVD for latent space
n_components = 2
svd = TruncatedSVD(n_components=n_components, random_state=42)
lsa_matrix = svd.fit_transform(tfidf_matrix)

print("\nDocument Vectors in Latent Space:")
print(pd.DataFrame(lsa_matrix, columns=[f"Dim{i+1}" for i in range(n_components)]))

# Step 3: Process a query
query = "automobile pollution causes"  # Note: "automobile" not in original documents

def process_query(query_text):
    # Transform query using same vectorizer and SVD
    query_tfidf = vectorizer.transform([query_text])
    query_lsa = svd.transform(query_tfidf)
    return query_lsa

query_vector = process_query(query)
print("\nQuery Vector in Latent Space:")
print(pd.DataFrame(query_vector, columns=[f"Dim{i+1}" for i in range(n_components)]))

# Step 4: Calculate similarities
cos_similarities = cosine_similarity(query_vector, lsa_matrix).flatten()
results = pd.DataFrame({
    'Document': documents,
    'Similarity': cos_similarities
}).sort_values('Similarity', ascending=False)

print("\nSearch Results:")
print(results)

# Step 5: Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(lsa_matrix[:, 0], lsa_matrix[:, 1], alpha=0.7, label='Documents')
plt.scatter(query_vector[0, 0], query_vector[0, 1], c='red', marker='X', s=200, label='Query')
for i, txt in enumerate([f"Doc{i+1}" for i in range(len(documents))]):
    plt.annotate(txt, (lsa_matrix[i, 0], lsa_matrix[i, 1]))
plt.annotate("Query", (query_vector[0, 0], query_vector[0, 1]))
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("Document-Query Projection in Latent Space")
plt.legend()
plt.show()