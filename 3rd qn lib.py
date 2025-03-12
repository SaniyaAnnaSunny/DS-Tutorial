import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Sample 2×2 matrix (2 samples, 2 features)
X = np.array([[2, 3], [6, 8]])  # Feature matrix
y = np.array([0, 1])  # Class labels

# Applying LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)

print("Transformed Data (LDA):\n", X_lda)
