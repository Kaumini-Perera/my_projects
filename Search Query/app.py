from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load and preprocess the data
df = pd.read_csv('documents.csv', header=None, names=['document_name', 'content'])
df['content'] = df['content'].str.lower()

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        results = search_documents(query)
        return render_template('results.html', query=query, results=results)
    return render_template('index.html')

def search_documents(query, top_n=5):
    query = query.lower()
    query_vec = tfidf_vectorizer.transform([query])
    cosine_similarities = np.dot(query_vec, tfidf_matrix.T).toarray()[0]
    relevant_doc_indices = np.argsort(cosine_similarities)[::-1][:top_n]
    
    results = []
    for idx in relevant_doc_indices:
        doc_name = df['document_name'].iloc[idx]
        score = cosine_similarities[idx]
        snippet = get_snippet(df['content'].iloc[idx], query)
        results.append({
            'document_name': doc_name,
            'score': f"{score:.4f}",
            'snippet': snippet
        })
    return results

def get_snippet(doc, query, snippet_length=150):
    query_terms = query.split()
    sentences = doc.split('.')
    for sentence in sentences:
        if any(term in sentence for term in query_terms):
            snippet = sentence.strip()
            if len(snippet) > snippet_length:
                snippet = snippet[:snippet_length] + '...'
            return snippet
    # Fallback if no sentence contains the query
    return doc[:snippet_length] + '...'

if __name__ == '__main__':
    app.run(debug=True)