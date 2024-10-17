import re
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def prepare(data, max_features=140839):
    # Ensure data is a pandas Series
    if not isinstance(data, pd.Series):
        raise ValueError("Input must be a pandas Series.")

    # Convert to lowercase
    data = data.str.lower()

    # Remove non-word characters and extra whitespace
    data = data.str.replace(r'\W', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()

    # Tokenization and stop word removal
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def tokenize_and_remove_stopwords(text):
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stop words and lemmatize
        filtered_lemmas = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(filtered_lemmas)

    # Apply the tokenization and stop word removal
    data = data.apply(tokenize_and_remove_stopwords)

    # Create the TF-IDF vectorizer with max_features
    vectorizer = TfidfVectorizer(max_features=max_features)
    data_vectorized = vectorizer.fit_transform(data)

    # Convert to dense array
    data_dense = data_vectorized.todense()

    # Ensure the output matches the required shape (num_samples, 140839)
    # Use np.array to ensure correct shape
    padded_data = np.zeros((data_dense.shape[0], max_features))  
    padded_data[:, :data_dense.shape[1]] = data_dense  

    return padded_data
