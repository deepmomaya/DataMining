## Inaugural Address Text Analysis

This Python code analyzes the textual content of 15 inaugural addresses by different US presidents. It accomplishes the following tasks:

1. **Data Retrieval:** Reads 15 .txt files, each containing the transcript of an inaugural address. The files are located in a directory specified by the user.

2. **Text Processing:** Tokenizes the content of each file, converts the text to lowercase, removes stopwords using NLTK's built-in corpus, and performs stemming using NLTK's Porter stemmer.

3. **TF-IDF Vectorization:** Calculates the TF-IDF (Term Frequency-Inverse Document Frequency) vector for each document. This involves computing the logarithmic TF (Term Frequency), logarithmic IDF (Inverse Document Frequency), and applying cosine normalization.

4. **Query-Document Similarity:** Implements query-document similarity using the ltc.lnc weighting scheme. Given a query string, the code computes the query vector, calculates the cosine similarity between the query vector and each document vector, and returns the most similar document.
 
