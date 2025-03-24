import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_texts(corpus):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer

def apply_svd(matrix, n_components=2):
    svd = TruncatedSVD(n_components=n_components)
    reduced_matrix = svd.fit_transform(matrix)
    return reduced_matrix, svd

def plot_topics(reduced_matrix, documents):
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], marker='o')

    for i, doc in enumerate(documents):
        plt.annotate(f"D{i+1}", (reduced_matrix[i, 0], reduced_matrix[i, 1]))

    plt.title("Documents in Topic Space")
    plt.xlabel("Topic 1")
    plt.ylabel("Topic 2")
    plt.grid(True)
    plt.show()

def display_topics(svd, vectorizer, n_top_words=5):
    topic_words = []
    for i, component in enumerate(svd.components_):
        top_words_idx = component.argsort()[-n_top_words:][::-1]
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
        topic_words.append(top_words)
        print(f"Topic {i+1}: {' '.join(top_words)}")
    return topic_words

# Example usage
documents = [
    "The cat sits on the mat.",
    "Dogs are a man's best friend.",
    "Cats and dogs are popular pets.",
    "I love my pet cat.",
    "The dog chased its tail."
]

# Vectorize the documents
tfidf_matrix, vectorizer = vectorize_texts(documents)

# Print the TF-IDF matrix
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())
print("\n")

# Apply SVD to the TF-IDF matrix
reduced_matrix, svd = apply_svd(tfidf_matrix, n_components=2)

# Print the reduced matrix
print("Reduced Matrix (Topic Space):")
print(reduced_matrix)
print("\n")

# Display the topics
print("Topics:")
topic_words = display_topics(svd, vectorizer, n_top_words=3)

# Plot the documents in the topic space
plot_topics(reduced_matrix, documents)
