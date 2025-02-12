# Import libraries
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
    "Cars and trucks pollute the environment"
]

# Step 1: Create a TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)  # (documents, terms)

# Step 2: Apply Truncated SVD (for sparse matrices)
n_components = 2  # Number of latent topics
svd = TruncatedSVD(n_components=n_components, random_state=42)
lsa_matrix = svd.fit_transform(tfidf_matrix)

# Document-topic matrix (reduced dimensions)
print("\nReduced Document-Topic Matrix (V^T):\n", lsa_matrix.round(3))

# Step 3: Analyze results
# -------------------------
# Document similarity using LSA vectors
cos_sim = cosine_similarity(lsa_matrix)
print("\nDocument Similarity Matrix:\n", pd.DataFrame(cos_sim).round(2))

# Explained variance by each component
explained_variance = svd.explained_variance_ratio_.sum()
print(f"\nExplained Variance (k={n_components}): {explained_variance:.2%}")

# Step 4: Interpret latent topics (terms contributing to components)
terms = vectorizer.get_feature_names_out()
for idx, component in enumerate(svd.components_):
    top_terms = np.argsort(component)[::-1][:3]  # Top 3 terms per topic
    print(f"\nTopic {idx+1}: {', '.join(terms[top_terms])}")


# Output
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(lsa_matrix[:, 0], lsa_matrix[:, 1], alpha=0.7)
for i, txt in enumerate(["Doc1", "Doc2", "Doc3", "Doc4"]):
    plt.annotate(txt, (lsa_matrix[i, 0], lsa_matrix[i, 1]))
plt.xlabel("Topic 1")
plt.ylabel("Topic 2")
plt.title("Document Projection in 2D Latent Space")
plt.show()