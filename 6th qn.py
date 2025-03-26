import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

class AgglomerativeClustering:
    def __init__(self, cluster_count=2, method='complete'):
        """
        Custom implementation of hierarchical clustering.
        """
        self.cluster_count = cluster_count
        self.method = method
        self.cluster_labels = None
        self.hierarchy_matrix = None
    
    def fit(self, data):
        """
        Perform hierarchical clustering and generate linkage matrix.
        """
        distance_matrix = pdist(data)
        self.hierarchy_matrix = linkage(distance_matrix, method=self.method)
        
        from scipy.cluster.hierarchy import fcluster
        self.cluster_labels = fcluster(self.hierarchy_matrix, t=self.cluster_count, criterion='maxclust') - 1
    
    def visualize_dendrogram(self):
        """
        Generate dendrogram for hierarchical clustering.
        """
        if self.hierarchy_matrix is None:
            raise ValueError("Clustering model has not been trained yet.")
        
        plt.figure(figsize=(10, 6))
        dendrogram(self.hierarchy_matrix)
        plt.title(f'Cluster Dendrogram ({self.method.capitalize()} Linkage)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()

# Generate sample dataset
np.random.seed(42)
data_samples = np.concatenate([
    np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2)),
    np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2)),
    np.random.normal(loc=[0, 5], scale=0.5, size=(50, 2))
])

# Apply hierarchical clustering
cluster_model = AgglomerativeClustering(cluster_count=3, method='complete')
cluster_model.fit(data_samples)

# Plot clustering results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(data_samples[:, 0], data_samples[:, 1], c=cluster_model.cluster_labels, cmap='viridis')
plt.title('Clustering Results')

# Generate dendrogram
plt.subplot(1, 2, 2)
cluster_model.visualize_dendrogram()
plt.tight_layout()
plt.show()