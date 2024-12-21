import numpy as np

A = np.random.randint(-3, 4, (2, 2))
cov_vector = A @ A.T
while np.any(np.linalg.eigvals(cov_vector) <= 0):
    A = np.random.randint(-3, 4, (2, 2))
    cov_vector = A @ A.T

mean_vector = np.random.randint(-3, 4, (2,))

print("Sigma Matrix = ", cov_vector)
print("\n Mean Mu is = ", mean_vector)

##-----------------------------------------------------------------------------------------------------------##
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

eigen_val_matrix, corresponding_eigen_vectors = np.linalg.eigh(cov_vector)

sample_size = int(1e3)
data_points = np.random.multivariate_normal(mean_vector, cov_vector, sample_size)

plt.figure(figsize=(8, 8))
plt.scatter(data_points[:, 0], data_points[:, 1], alpha=0.2, label="Data Points")

plt.plot(mean_vector[0], mean_vector[1], 'ro', label="Mean ($\mu$)")

for i in range(len(eigen_val_matrix)):
    eigval = eigen_val_matrix[i]
    eigvec = corresponding_eigen_vectors[:, i]
    vector = np.sqrt(eigval) * eigvec * 2
    plt.quiver(mean_vector[0], mean_vector[1], vector[0], vector[1], angles='xy', scale_units='xy', scale=1, width=0.005,
               label=f"Eigenvector {i+1} (scaled)")

teeta = np.linspace(0, 2 * np.pi, 100)
ellipsoid_model = np.array([np.cos(teeta), np.sin(teeta)])
transformed_ellipsoid_model = corresponding_eigen_vectors @ np.diag(np.sqrt(eigen_val_matrix) * 2) @ ellipsoid_model
plt.plot(transformed_ellipsoid_model[0, :] + mean_vector[0], transformed_ellipsoid_model[1, :] + mean_vector[1], label="Ellipsoid")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axhline(0, color="grey", linewidth=0.5)
plt.axvline(0, color="grey", linewidth=0.5)
plt.legend()
plt.title("2-Dimensional Multivariate Gaussian Data Distribution, Eigenvectors, and Ellipsoid")
plt.grid(True)
plt.axis("equal")

plt.show()