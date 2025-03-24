# Topic Modeling

The code works by taking a set of text documents and performing topic modeling to reveal underlying themes, then visualizing the results. Here’s a step-by-step explanation of its working:

TF-IDF Transformation:

The TfidfVectorizer converts the documents into a TF-IDF matrix.

This matrix represents each document as a vector, where each element reflects the importance of a term in that document, after removing common stop words.

Dimensionality Reduction with SVD:

Truncated Singular Value Decomposition (SVD) is applied to the TF-IDF matrix.

This reduces the high-dimensional data into a lower-dimensional space defined by latent topics.

In this example, n_topics is set to 2, so each document is now represented by two topic weights, forming the document-topic matrix.

The SVD also produces a topic-term matrix, which shows how much each term contributes to each topic.

Topic Interpretation:

The function get_top_terms_per_topic sorts each topic’s term weights and selects the top three terms.

These top terms are intended to provide an intuitive understanding of the themes represented by each topic.

Visualization:

The document-topic matrix is plotted on a 2D scatter plot, with the x-axis representing Topic 1 and the y-axis representing Topic 2.

Each document is plotted as a point, and annotated with its label (e.g., Doc1, Doc2, etc.), allowing you to see how documents are positioned relative to the topics.
