# text_similarity.py

import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('wordnet')
nltk.download('stopwords')

# Preprocessing setup
lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words('english'))

# Load and prepare the data
qa_pairs = pd.read_csv('q&a.csv', on_bad_lines='skip')   # Update the path accordingly
vectorizer = TfidfVectorizer()

def preprocess(text):
    lemmatized = [lemmatizer.lemmatize(word.lower()) for word in text.split() if word.lower() not in english_stopwords]
    return ' '.join(lemmatized)

def find_closest_match(user_input, tfidf_matrix, qa_pairs):
    processed_input = preprocess(user_input)
    input_vector = vectorizer.transform([processed_input])
    similarities = cosine_similarity(input_vector, tfidf_matrix)
    closest = similarities.argmax()
    return qa_pairs.iloc[closest]['Answer']   

# Prepare the data for use in find_closest_match
def setup_similarity():
    qa_pairs['Question'] = qa_pairs['Question'].apply(preprocess)
    tfidf_matrix = vectorizer.fit_transform(qa_pairs['Question'])
    return tfidf_matrix, qa_pairs

if __name__ == "__main__":
    tfidf_matrix, qa_pairs = setup_similarity()