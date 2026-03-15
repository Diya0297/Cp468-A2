import csv
import math
import matplotlib.pyplot as plt

'''
    A marketing analytics team at a Waterloo-based e-commerce company
    wants to understand customer behaviour by grouping similar customers into clusters. 
    The team has collected a dataset (CustomerProfiles_Q2.csv) containing 30 customer profiles, 
    with two key features representing each customer: 
         Feature 1 (x1): Customer’s average monthly spending
         Feature 2 (x2): Customer’s total number of purchases
    To segment customers into different groups, the team uses K-means clustering
'''
def load_points(filename):
    points = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            x = float(row[0])
            y = float(row[1])
            points.append((x, y))
    return points

def plot_raw_data(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    plt.figure()
    plt.scatter(xs, ys, c='purple')
    plt.title("Customer Data Before Clustering")
    plt.xlabel("Average Monthly Spending(f1)")
    plt.ylabel("Total Number of Purchases(f2)")
    plt.grid(True)
    plt.show()

def euclidean_distance(p, q):
    return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)

def compute_centroid(cluster):
    if len(cluster) == 0:
        return (0, 0)
    x_mean = sum(p[0] for p in cluster) / len(cluster)
    y_mean = sum(p[1] for p in cluster) / len(cluster)
    return (x_mean, y_mean)

def k_means(points, k, max_iters=100):
    # Step 1: first k points as initial centroids
    centroids = [points[i] for i in range(k)]

    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        labels = []

        # Step 2: assign each point to closest centroid
        for p in points:
            distances = [euclidean_distance(p, c) for c in centroids]
            closest = distances.index(min(distances))
            clusters[closest].append(p)
            labels.append(closest)

        # Step 4: compute new centroids
        new_centroids = [compute_centroid(cluster) for cluster in clusters]

        # Step 5: stop if centroids don't change
        if new_centroids == centroids:
            break

        centroids = new_centroids

    return labels, centroids

def plot_clusters(points, labels, centroids, k):
    colors = ['red', 'green', 'blue', 'purple', 'orange']

    plt.figure()
    for i, p in enumerate(points):
        plt.scatter(p[0], p[1], c=colors[labels[i]])

    for i, c in enumerate(centroids):
        plt.scatter(c[0], c[1], c='black', marker='X', s=200)

    plt.title(f"K-Means Clustering (k = {k})")
    plt.xlabel("Average Monthly Spending(f1)")
    plt.ylabel("Total Number of Purchases(f2)")
    plt.grid(True)
    plt.show()

def main():
    points = load_points("data/CustomerProfiles_Q2.csv")

    # Part A
    plot_raw_data(points)

    # Part B + C
    k = 2
    labels, centroids = k_means(points, k)
    plot_clusters(points, labels, centroids, k)

    # Part D
    cluster_counts = [labels.count(i) for i in range(k)]
    print("Cluster sizes:", cluster_counts)

    # Part E
    print("Final centroids:")
    for i, c in enumerate(centroids):
        print(f"Cluster {i}: ({c[0]:.2f}, {c[1]:.2f})")

if __name__ == "__main__":
    main()
