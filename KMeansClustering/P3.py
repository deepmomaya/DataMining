import numpy as np
import matplotlib.pyplot as plt
import argparse

# Read the dataset
def read_data(file_path):
    """
    Read data from a file.

    Parameters:
        file_path (str): Path to the data file.

    Returns:
        numpy.ndarray: Array containing the data.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            row = list(map(float, line.strip().split()))
            data.append(row)
    return np.array(data)

# Implement distance calculation
def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
        point1 (numpy.ndarray): Coordinates of the first point.
        point2 (numpy.ndarray): Coordinates of the second point.

    Returns:
        float: Euclidean distance between the points.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Implement K-means algorithm
def kmeans_clustering(data, k=2, max_iterations=20):
    """
    Perform K-means clustering on the given data.

    Parameters:
        data (numpy.ndarray): Data points.
        k (int): Number of clusters.
        max_iterations (int): Maximum number of iterations.

    Returns:
        list: List of errors for each iteration.
    """
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    errors = []
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)
        new_centroids = []
        error = 0
        for i in range(k):
            if len(clusters[i]) > 0:
                new_centroid = np.mean(clusters[i], axis=0)
                new_centroids.append(new_centroid)
                error += np.sum((np.array(clusters[i]) - new_centroid) ** 2)
        centroids = np.array(new_centroids)
        errors.append(error)
    return errors

# Calculate errors for different values of K
def calculate_errors(data_file):
    """
    Calculate errors for different values of K and plot the results.

    Parameters:
        data_file (str): Path to the data file.
    """
    data = read_data(data_file)
    errors_list = []
    for k in range(2, 11):
        errors = kmeans_clustering(data, k)
        final_error = errors[-1]
        errors_list.append(final_error)
        print(f"For k = {k} After 20 iterations: Error = {final_error:.4f}")
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 11), errors_list, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Error')
    plt.title('Error vs. K')
    plt.grid(True)
    plt.show()

# Run the code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-means clustering")
    parser.add_argument("data_file", help="Path to the data file")
    args = parser.parse_args()
    data_file_path = args.data_file
    calculate_errors(data_file_path)
