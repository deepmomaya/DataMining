# Importing necessary libraries
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
import os
import io
import math

# Initialize variables
stemmer = PorterStemmer()  # Initialize a Porter stemmer for word stemming
document_file_lengths = Counter()  # Counter to store the length of each document
document_word_counts = {}  # Dictionary to store word counts for each document
document_term_frequencies = Counter()  # Counter to store term frequencies across documents
query_term_frequencies = Counter()  # Counter to store term frequencies in the query
document_vectors = {}  # Dictionary to store TF-IDF weighted vectors for each document
query_vector = {}  # Dictionary to store TF weighted vector for the query

# Function to preprocess a document
def preprocess_document(doc):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(doc)
    stop_words = stopwords.words('english')
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return tokens

# Function to calculate the inverse document frequency (IDF) for a given term
def getidf(term):
    if document_term_frequencies[term] != 0:
        return math.log10(len(document_word_counts) / document_term_frequencies[term])
    else:
        return -1

# Function to calculate the TF-IDF weight for a term in a document
def getweight(filename, term):
    stemmed_term = stemmer.stem(term)
    if document_word_counts[filename][stemmed_term] > 0:
        return (1 + math.log10(document_word_counts[filename][stemmed_term])) * getidf(term)
    else:
        return -1

# Function to calculate the query vector and perform normalization
def calculate_query_vector(query_str):
    query_terms = preprocess_document(query_str)
    for filename in document_word_counts:
        vector_length = 0
        query_vector[filename] = Counter()
        for term in query_terms:
            if document_word_counts[filename][term] == 0:
                tf = 1
            else:
                tf = 1 + math.log10(document_word_counts[filename][term])
            query_vector[filename][term] = tf
            vector_length += tf**2
        document_term_frequencies[filename] = math.sqrt(vector_length)
    for filename in document_word_counts:
        for term in query_terms:
            if document_term_frequencies[filename] == 0:
                query_vector[filename][term] = 0
            else:
                query_vector[filename][term] = query_vector[filename][term] / document_term_frequencies[filename]

# Function to calculate the cosine similarity
def query(query_str):
    cosine_similarities = {}
    query_terms = preprocess_document(query_str)
    calculate_query_vector(query_str)
    for filename in document_word_counts:
        similarity = 0
        for term in query_terms:
            weight = document_vectors[filename][term] * query_vector[filename][term]
            similarity += weight
        cosine_similarities[filename] = similarity
    best_match = max(cosine_similarities, key=cosine_similarities.get)
    return best_match, cosine_similarities[best_match]

# Main code
corpus_root = 'C:/Users/Admin/Desktop/P1/US_Inaugural_Addresses'

# Loop through files in the corpus directory
for filename in os.listdir(corpus_root):
    if filename.startswith('0') or filename.startswith('1'):
        with open(os.path.join(corpus_root, filename), "r", encoding='windows-1252') as file:
            document_text = file.read().lower()
            document_terms = preprocess_document(document_text)
            document_term_frequencies += Counter(set(document_terms))
            document_word_counts[filename] = Counter(document_terms)

# Calculate TF-IDF weighted vectors for each document
for filename in document_word_counts:
    document_vectors[filename] = Counter()
    vector_length = 0
    for term in document_word_counts[filename]:
        tf = 1 + math.log10(document_word_counts[filename][term])
        document_vectors[filename][term] = tf
        vector_length += tf**2
    document_file_lengths[filename] = math.sqrt(vector_length)

# Normalize document vectors
for filename in document_word_counts:
    for term in document_word_counts[filename]:
        weight = document_vectors[filename][term]
        document_vectors[filename][term] = weight / document_file_lengths[filename]

# Test cases
print("%.12f" % getidf('british'))
print("%.12f" % getidf('union'))
print("%.12f" % getidf('war'))
print("%.12f" % getidf('military'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('02_washington_1793.txt', 'arrive'))
print("%.12f" % getweight('07_madison_1813.txt', 'war'))
print("%.12f" % getweight('12_jackson_1833.txt', 'union'))
print("%.12f" % getweight('09_monroe_1821.txt', 'british'))
print("%.12f" % getweight('05_jefferson_1805.txt', 'public'))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("false public"))
print("(%s, %.12f)" % query("people institutions"))
print("(%s, %.12f)" % query("violated willingly"))
