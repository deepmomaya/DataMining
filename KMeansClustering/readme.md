## K-Means Clustering Implementation

This Python code implements the K-means clustering algorithm on UCI datasets without using library functions. The program takes a dataset file as input and performs K-means clustering for a range of K values (2-10). It then calculates and prints the error after 20 iterations for each K value. The error is computed using Euclidean distance between data points and centroids. Additionally, the code visualizes the error values corresponding to different values of K in a graph. It accomplishes the following tasks:

1. **Data Loading:** Read the dataset from a file specified by the user.

2. **Distance Calculation:** Implement the Euclidean distance function to measure the distance between data points and centroids.

3. **K-Means Algorithm:** Implement the K-means clustering algorithm according to the specified steps.

4. **Initialization:** Explore different initialization approaches to mitigate the Initial Centroid Problem.

5. **Error Calculation:** Calculate the error for each K value after 20 iterations and print the results.

6. **Graphical Representation:** Plot a graph showing the Error values against the different values of K.
