import numpy as np

# Given 2×2 data matrix (each row is a sample, each column is a feature)
X = np.array([[2, 3], [6, 8]])  # Feature matrix
y = np.array([0, 1])  # Class labels

# Step 1: Compute mean vectors
mean_0 = np.mean(X[y == 0], axis=0)  # Mean of class 0
mean_1 = np.mean(X[y == 1], axis=0)  # Mean of class 1

# Step 2: Compute within-class scatter matrix Sw
S_W = np.zeros((2, 2))
for i, mean in zip([0, 1], [mean_0, mean_1]):
    class_scatter = (X[y == i] - mean).T @ (X[y == i] - mean)
    S_W += class_scatter

# Step 3: Compute between-class scatter matrix Sb
mean_diff = (mean_1 - mean_0).reshape(2, 1)
S_B = mean_diff @ mean_diff.T

# Step 4: Solve for eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Step 5: Select the eigenvector corresponding to the largest eigenvalue
top_eigenvector = eig_vecs[:, np.argmax(eig_vals)]

# Step 6: Transform the data
X_lda_manual = X @ top_eigenvector

print("Eigenvector used for LDA transformation:\n", top_eigenvector)
print("Transformed Data (LDA):\n", X_lda_manual)
