import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def to_txt(text: str) -> str:
    """Replace '.' and '/' with spaces in the text."""
    return text.replace('.', ' ').replace('/', ' ')

def tok(string: str) -> list[str]:
    """Tokenize the string by replacing '/' with '.' and splitting by '.'. """
    return string.replace('/', '.').split('.')

def num_digits(text: str) -> int:
    """Count the number of digits in the text."""
    return len(re.findall(r'\d', text))

def num_dots(text: str) -> int:
    """Count the number of dots in the text."""
    return len(re.findall(r'\.', text))

def num_bar(text: str) -> int:
    """Count the number of slashes in the text."""
    return len(re.findall(r'/', text))

def preprocess_new_data(df: pd.DataFrame, vocabulary: list[str]) -> pd.DataFrame:
    # Create a list of processed texts
    processed_urls = [to_txt(url) for url in df.url]

    # Create a Document-Term Matrix
    vectorizer = CountVectorizer(binary=True, vocabulary=vocabulary)
    doc_term_matrix = vectorizer.fit_transform(processed_urls)

    # Create a DataFrame with the document-term matrix
    matrix = pd.DataFrame(doc_term_matrix.toarray(), columns=vocabulary)
    
    # Add feature columns
    matrix['dots'] = [num_dots(url) for url in df.url]
    matrix['bars'] = [num_bar(url) for url in df.url]
    matrix['length'] = [len(url) for url in processed_urls]
    matrix['digits'] = [num_digits(url) for url in df.url]
    
    return matrix


# Define your vocabulary
VOC = ['http', 'https', 'ftp', 'example', 'com', 'test', 'page', 'query']

