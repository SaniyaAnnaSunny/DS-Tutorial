import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LinearSVM:
    def __init__(self, learning_rate=0.001, regularization=0.01, iterations=1000):
        """
        Linear Support Vector Machine (SVM) using gradient-based optimization.
        """
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.weights = None
        self.offset = None
    
    def train(self, data, labels):
        """
        Train the SVM model using a simple iterative approach.
        """
        num_samples, num_features = data.shape
        processed_labels = np.where(labels <= 0, -1, 1)
        
        self.weights = np.zeros(num_features)
        self.offset = 0
        
        for _ in range(self.iterations):
            for idx, sample in enumerate(data):
                condition = processed_labels[idx] * (np.dot(sample, self.weights) - self.offset) >= 1
                
                if condition:
                    self.weights -= self.learning_rate * (2 * self.regularization * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.regularization * self.weights - np.dot(sample, processed_labels[idx]))
                    self.offset -= self.learning_rate * processed_labels[idx]
    
    def classify(self, data):
        """
        Classify input data based on trained model.
        """
        return np.sign(np.dot(data, self.weights) - self.offset)
    
    def compute_boundary(self, data):
        """
        Compute margin-based decision boundary.
        """
        return np.dot(data, self.weights) - self.offset

# Function to display the decision boundary
def plot_boundary(model, data, labels):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.Paired)
    ax = plt.gca()
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    
    x_vals = np.linspace(x_range[0], x_range[1], 30)
    y_vals = np.linspace(y_range[0], y_range[1], 30)
    grid_y, grid_x = np.meshgrid(y_vals, x_vals)
    points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    Z = model.compute_boundary(points).reshape(grid_x.shape)
    
    ax.contour(grid_x, grid_y, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.show()

# Example usage
if __name__ == "__main__":
    data, labels = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                       n_informative=2, random_state=1, 
                                       n_clusters_per_class=1)
    labels = np.where(labels == 0, -1, 1)
    
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    svm_model = LinearSVM(learning_rate=0.001, regularization=0.01, iterations=1000)
    svm_model.train(data_train, labels_train)
    
    label_predictions = svm_model.classify(data_test)
    model_accuracy = np.mean(label_predictions == labels_test)
    print(f"SVM Classification Accuracy: {model_accuracy:.4f}")
    
    plot_boundary(svm_model, data_train, labels_train)
