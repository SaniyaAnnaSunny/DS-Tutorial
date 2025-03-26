import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class CustomRandomForest:
    def __init__(self, n_trees=100, depth_limit=None, feature_subset='sqrt', seed=None):
        """
        Initialize the Custom Random Forest model.
        
        Parameters:
        - n_trees: Number of trees in the forest
        - depth_limit: Maximum depth of each tree
        - feature_subset: Number of features considered at each split ('sqrt' for square root of total features)
        - seed: Random seed for reproducibility
        """
        self.n_trees = n_trees
        self.depth_limit = depth_limit
        self.feature_subset = feature_subset
        self.seed = seed
        self.forest = []  # Stores trained trees
        self.selected_features = []  # Stores feature indices used per tree
    
    def fit(self, X_train, y_train):
        """
        Train the Custom Random Forest using bootstrapped samples.
        
        Parameters:
        - X_train: Feature matrix (num_samples, num_features)
        - y_train: Target labels (num_samples,)
        """
        np.random.seed(self.seed)
        num_samples, num_features = X_train.shape
        
        # Determine the number of features to use per tree
        if self.feature_subset == 'sqrt':
            num_selected_features = int(np.sqrt(num_features))
        elif isinstance(self.feature_subset, int):
            num_selected_features = self.feature_subset
        else:
            num_selected_features = num_features
        
        self.forest = []
        self.selected_features = []
        
        for _ in range(self.n_trees):
            # Generate a bootstrap sample
            X_resampled, y_resampled = resample(X_train, y_train, random_state=self.seed)
            
            # Randomly choose feature subset
            chosen_features = np.random.choice(num_features, num_selected_features, replace=False)
            X_resampled = X_resampled[:, chosen_features]
            
            # Train individual decision tree
            tree = DecisionTreeClassifier(max_depth=self.depth_limit, random_state=self.seed)
            tree.fit(X_resampled, y_resampled)
            
            # Store trained tree and selected feature indices
            self.forest.append(tree)
            self.selected_features.append(chosen_features)
    
    def predict_proba(self, X_test):
        """
        Compute class probabilities by averaging predictions from all trees.
        
        Parameters:
        - X_test: Feature matrix (num_samples, num_features)
        
        Returns:
        - avg_probabilities: Array (num_samples, num_classes) with class probabilities
        """
        num_samples = X_test.shape[0]
        probabilities = []
        
        for tree, features in zip(self.forest, self.selected_features):
            X_subset = X_test[:, features]
            probabilities.append(tree.predict_proba(X_subset))
        
        # Average the probabilities from all trees
        avg_probabilities = np.mean(probabilities, axis=0)
        return avg_probabilities
    
    def predict(self, X_test):
        """
        Predict class labels using majority voting from all trees.
        
        Parameters:
        - X_test: Feature matrix (num_samples, num_features)
        
        Returns:
        - predicted_labels: Array (num_samples,) with class predictions
        """
        avg_probabilities = self.predict_proba(X_test)
        return np.argmax(avg_probabilities, axis=1)

# Example usage
if __name__ == "__main__":
    # Load sample dataset
    dataset = load_iris()
    X, y = dataset.data, dataset.target
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Custom Random Forest
    model = CustomRandomForest(n_trees=100, depth_limit=3, seed=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate model accuracy
    model_accuracy = accuracy_score(y_test, predictions)
    print(f"Custom Random Forest Accuracy: {model_accuracy:.4f}")
