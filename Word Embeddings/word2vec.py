import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec

# Sample corpus
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "a dog is a man's best friend",
    "cats and dogs are common household pets",
    "cars and trucks pollute the environment"
]

# Preprocessing
def preprocess(text):
    return [word.lower() for word in text.split()]

processed_corpus = [preprocess(doc) for doc in corpus]

# Build vocabulary and co-occurrence matrix
# ============================================
vocab = list(set(word for doc in processed_corpus for word in doc))
word2idx = {word: idx for idx, word in enumerate(vocab)}
window_size = 2

# Initialize co-occurrence matrix
cooccur_matrix = np.zeros((len(vocab), len(vocab)), dtype=np.float32)

for doc in processed_corpus:
    for i, target_word in enumerate(doc):
        start = max(0, i - window_size)
        end = min(len(doc), i + window_size + 1)
        for j in range(start, end):
            if i != j:
                context_word = doc[j]
                cooccur_matrix[word2idx[target_word], word2idx[context_word]] += 1

# Convert to DataFrame for readability
cooccur_df = pd.DataFrame(
    cooccur_matrix,
    index=vocab,
    columns=vocab
)
print("Co-occurrence Matrix:\n", cooccur_df)

# SVD for Word Embeddings
# ============================================
svd = TruncatedSVD(n_components=2, random_state=42)
word_vectors_svd = svd.fit_transform(cooccur_matrix)
word_vectors_svd = pd.DataFrame(word_vectors_svd, index=vocab, columns=["Dim1", "Dim2"])
print("\nSVD Embeddings:\n", word_vectors_svd)

# Word2Vec for Comparison
# ============================================
model_w2v = Word2Vec(
    sentences=processed_corpus,
    vector_size=2,
    window=2,
    min_count=1,
    workers=4,
    epochs=100
)
word_vectors_w2v = pd.DataFrame(
    [model_w2v.wv[word] for word in vocab],
    index=vocab,
    columns=["Dim1", "Dim2"]
)
print("\nWord2Vec Embeddings:\n", word_vectors_w2v)

# Visualization
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# SVD Embeddings
ax1.scatter(word_vectors_svd["Dim1"], word_vectors_svd["Dim2"], alpha=0.7)
for word in vocab:
    ax1.annotate(word, (word_vectors_svd.loc[word, "Dim1"], word_vectors_svd.loc[word, "Dim2"]))
ax1.set_title("SVD Embeddings from Co-occurrence Matrix")

# Word2Vec Embeddings
ax2.scatter(word_vectors_w2v["Dim1"], word_vectors_w2v["Dim2"], alpha=0.7)
for word in vocab:
    ax2.annotate(word, (word_vectors_w2v.loc[word, "Dim1"], word_vectors_w2v.loc[word, "Dim2"]))
ax2.set_title("Word2Vec Embeddings")

plt.tight_layout()
plt.show()

# Similarity Analysis
# ============================================
def get_similarities(word, vectors, vocab):
    word_vec = vectors.loc[word].values.reshape(1, -1)
    all_vecs = vectors.values
    sims = cosine_similarity(word_vec, all_vecs).flatten()
    return pd.Series(sims, index=vocab).sort_values(ascending=False)

print("\nSVD Similarities for 'dog':")
print(get_similarities("dog", word_vectors_svd, vocab).head(4))  # Top 3 + self

print("\nWord2Vec Similarities for 'dog':")
print(get_similarities("dog", word_vectors_w2v, vocab).head(4))