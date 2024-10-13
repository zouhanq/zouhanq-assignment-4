from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


# Fetch the dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Initialize the TF-IDF vectorizer and transform the documents
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Apply Truncated SVD to perform LSA
n_components = 100  # You can adjust this number
svd = TruncatedSVD(n_components=n_components, random_state=42)
lsa_matrix = svd.fit_transform(tfidf_matrix)

# Normalize the LSA matrix for cosine similarity computation
lsa_matrix_norm = lsa_matrix / np.linalg.norm(lsa_matrix, axis=1, keepdims=True)


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # Transform the query using the same TF-IDF vectorizer
    query_tfidf = vectorizer.transform([query])

    # Project the query into the LSA space
    query_lsa = svd.transform(query_tfidf)

    # Normalize the query vector
    query_lsa_norm = query_lsa / np.linalg.norm(query_lsa)

    # Compute cosine similarities
    similarities = np.dot(lsa_matrix_norm, query_lsa_norm.T).flatten()

    # Get the indices of the top 5 similar documents
    top_indices = similarities.argsort()[-5:][::-1]

    # Retrieve the top documents and their similarities
    top_documents = [documents[i] for i in top_indices]
    top_similarities = [similarities[i] for i in top_indices]
    
    return top_documents, top_similarities, top_indices.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(port=3000)