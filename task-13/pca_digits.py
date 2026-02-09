import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load digits dataset
digits = load_digits()
X = digits.data       # already flattened
y = digits.target

print("Original shape:", X.shape)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Logistic Regression on original data
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
original_acc = accuracy_score(y_test, y_pred)

print("Accuracy without PCA:", original_acc)

# Apply PCA with different components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative explained variance
plt.plot(explained_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance vs Components")
plt.show()

# PCA with 30 components
pca_30 = PCA(n_components=30)
X_pca_30 = pca_30.fit_transform(X_scaled)

# Train-test split after PCA
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_pca_30, y, test_size=0.2, random_state=42
)

# Logistic Regression after PCA
lr_pca = LogisticRegression(max_iter=2000)
lr_pca.fit(X_train_p, y_train_p)
y_pred_pca = lr_pca.predict(X_test_p)
pca_acc = accuracy_score(y_test_p, y_pred_pca)

print("Accuracy with PCA (30 components):", pca_acc)

# PCA 2D visualization
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=y, cmap="tab10", s=10)
plt.colorbar()
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("PCA 2D Visualization")
plt.show()
