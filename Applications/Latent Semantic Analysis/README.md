# Latent Semantic Analysis

The code performs the following steps:

Data Loading and Preprocessing:
It reads a BBC news dataset, extracting the document categories and their text content. The texts are then converted into a numerical document-term matrix using TF-IDF, which highlights the importance of words in each document while filtering out common stop words.

Custom SVD via Eigen-Decomposition:
It implements a function to compute a singular value decomposition (SVD) by performing an eigen-decomposition on the document-term matrix’s transpose product. This function calculates the top two singular values and corresponding vectors, reducing the high-dimensional data into a 2-dimensional latent space.

Clustering:
The code applies KMeans clustering (with 5 clusters) on the computed singular vectors (from the eigen-decomposition). This groups similar features (or terms) together based on their representation in the reduced space.

Visualization:
Finally, it visualizes the 2-dimensional latent space using a scatter plot where each point is colored according to its original document category. This helps in understanding the distribution and clustering of the data in the latent space.
