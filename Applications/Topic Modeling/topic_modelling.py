import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

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
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)  # (documents, terms)

# Step 2: Apply SVD for topic modeling
n_topics = 2  # Number of latent topics
svd = TruncatedSVD(n_components=n_topics, random_state=42)
topic_matrix = svd.fit_transform(tfidf_matrix)

# Document-topic matrix (each row = document, each column = topic weight)
print("\nDocument-Topic Matrix:")
print(pd.DataFrame(topic_matrix, columns=[f"Topic {i+1}" for i in range(n_topics)]))

# Topic-term matrix (each row = topic, each column = term weight)
topic_term_matrix = svd.components_
print("\nTopic-Term Matrix:")
print(pd.DataFrame(topic_term_matrix, columns=vectorizer.get_feature_names_out()))

# Step 3: Interpret topics
def get_top_terms_per_topic(topic_term_matrix, feature_names, n_terms=3):
    topics = []
    for i, topic in enumerate(topic_term_matrix):
        top_terms_idx = topic.argsort()[-n_terms:][::-1]
        top_terms = [feature_names[idx] for idx in top_terms_idx]
        topics.append(top_terms)
    return topics

feature_names = vectorizer.get_feature_names_out()
top_terms = get_top_terms_per_topic(topic_term_matrix, feature_names)
print("\nTop Terms per Topic:")
for i, terms in enumerate(top_terms):
    print(f"Topic {i+1}: {', '.join(terms)}")

# Step 4: Visualize document-topic relationships
plt.figure(figsize=(8, 6))
plt.scatter(topic_matrix[:, 0], topic_matrix[:, 1], alpha=0.7)
for i, txt in enumerate([f"Doc{i+1}" for i in range(len(documents))]):
    plt.annotate(txt, (topic_matrix[i, 0], topic_matrix[i, 1]))
plt.xlabel("Topic 1")
plt.ylabel("Topic 2")
plt.title("Document-Topic Projection")
plt.show()