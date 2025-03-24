# Information Retrieval

This code demonstrates a complete workflow for processing text documents using TF-IDF, reducing dimensionality with Latent Semantic Analysis (LSA), and retrieving the most relevant documents for a given query. Here’s a breakdown of what each part does:

Vectorizing the Texts:
The function vectorize_texts uses the TfidfVectorizer from scikit-learn to convert the list of documents into a TF-IDF matrix, which numerically represents the importance of words in each document while filtering out common English stop words.

Applying LSA:
The apply_lsa function applies Truncated Singular Value Decomposition (SVD) on the TF-IDF matrix to reduce its dimensionality. This process (LSA) captures the underlying topics in the documents and represents them in a lower-dimensional latent space (here, 2 dimensions).

Retrieving Relevant Documents:
The retrieve_documents function transforms the query into the same latent space using the fitted TF-IDF vectorizer and SVD model. It then computes cosine similarities between the query and each document’s latent representation. The documents are ranked by similarity, and the top results are returned.

Plotting the Embeddings:
The plot_embeddings function visualizes the 2D latent space embeddings. It plots the documents as points and the query as a red “x” marker, annotating each document with a label (D1, D2, …). This plot helps illustrate how the query relates to the documents in terms of their semantic content.

Example Usage:
The provided example uses a small set of documents about cats and dogs and a query focused on cats. After vectorization, LSA, and retrieval, it prints out:

The TF-IDF matrix (numeric representation of the documents).

The LSA-transformed matrix (reduced dimensionality representation).

Cosine similarity scores indicating how similar each document is to the query.

The list of top relevant documents based on these scores.

A 2D scatter plot showing the positions of the query and all documents.
