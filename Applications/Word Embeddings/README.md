# Word Embeddings

The code implements a basic approach to learning word embeddings from a text corpus using a co-occurrence matrix and Singular Value Decomposition (SVD). Here’s a step-by-step explanation of its working:

Building the Co-occurrence Matrix:

Function: build_cooccurrence_matrix(corpus, window_size=2)

Process:

The function iterates over each sentence in the corpus.

For every word in a sentence, it considers a context window (neighbors before and after the word, controlled by window_size).

It records how many times each pair of (target_word, context_word) co-occurs across the corpus.

At the same time, it collects all unique words (vocabulary).

Creating the Word-Context Matrix:

Function: build_word_context_matrix(cooccurrence_matrix, vocab)

Process:

The vocabulary is converted into an index mapping (each word gets a unique index).

A square matrix of size (vocab_size x vocab_size) is created, where each entry represents the co-occurrence count between the target word and a context word.

Dimensionality Reduction using SVD:

Function: apply_svd(matrix, n_components=2)

Process:

Truncated SVD is applied to the word-context matrix to reduce its dimensions (in this case, to 2).

The resulting lower-dimensional representation is treated as the word embeddings. These embeddings capture latent semantic relationships between words.

Visualization:

Function: plot_embeddings(word_embeddings, vocab)

Process:

The 2-dimensional word embeddings are plotted on a scatter plot.

Each point is annotated with the corresponding word, allowing for visual inspection of how words are semantically related.

Example Usage:

A small corpus is provided, and the functions are called sequentially:

The co-occurrence matrix is built from the corpus.

The word-context matrix is created from the co-occurrence data.

SVD reduces this matrix to 2 dimensions to obtain word embeddings.

Finally, the word embeddings are printed and visualized on a scatter plot.
