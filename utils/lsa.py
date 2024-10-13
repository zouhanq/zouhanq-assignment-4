# utils/lsa.py

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

class LSASearchEngine:
    def __init__(self, n_components=100):
        # Load the dataset
        self.documents = self.load_dataset()
        # Initialize TF-IDF vectorizer and transform documents
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        # Apply Truncated SVD for LSA
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.lsa_matrix = self.svd.fit_transform(self.tfidf_matrix)
        # Normalize the LSA matrix
        self.lsa_matrix_norm = self.normalize_matrix(self.lsa_matrix)

    def load_dataset(self):
        """Fetches the 20 Newsgroups dataset."""
        newsgroups = fetch_20newsgroups(subset='all')
        return newsgroups.data

    def normalize_matrix(self, matrix):
        """Normalizes the rows of a matrix to unit length."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        return matrix / norms

    def process_query(self, query):
        """Processes a query and returns its LSA representation."""
        query_tfidf = self.vectorizer.transform([query])
        query_lsa = self.svd.transform(query_tfidf)
        query_lsa_norm = self.normalize_matrix(query_lsa)
        return query_lsa_norm

    def compute_similarities(self, query_lsa_norm):
        """Computes cosine similarities between the query and all documents."""
        similarities = np.dot(self.lsa_matrix_norm, query_lsa_norm.T).flatten()
        return similarities

    def search(self, query, top_n=5):
        """
        Searches for the top N documents similar to the query.
        Returns:
            - top_documents: list of top N documents.
            - top_similarities: list of cosine similarity scores.
            - top_indices: list of indices of the top N documents.
        """
        query_lsa_norm = self.process_query(query)
        similarities = self.compute_similarities(query_lsa_norm)
        # Get the indices of the top N similar documents
        top_indices = similarities.argsort()[-top_n:][::-1]
        top_documents = [self.documents[i] for i in top_indices]
        top_similarities = [similarities[i] for i in top_indices]
        return top_documents, top_similarities, top_indices.tolist()
