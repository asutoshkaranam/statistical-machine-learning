import numpy as np

def get_matrix():
    A = np.random.uniform(-1, 1, (3, 3))
    
    # Ensure full rank by adding a small value to the diagonal
    A += np.eye(3) * 0.01
    
    sigma_matrix = np.dot(A.T, A)
    
    return sigma_matrix

def result_parta_partb(matrix):
    # symmetry
    symmetry_check = np.allclose(matrix, matrix.T)
    
    # positive semi-definiteness
    eigenvalues = np.linalg.eigvals(matrix)
    positive_definite_check = np.all(eigenvalues >= -1e-10)  # Prevent FloatingPointError
    
    return symmetry_check, positive_definite_check

sigma_matrix = get_matrix()
print("Matrix rank:", np.linalg.matrix_rank(sigma_matrix))
print("\n")

symmetry_check, positive_definite_check = result_parta_partb(sigma_matrix)

print("Covariance Matrix:")
print(sigma_matrix)
print("\n")
print("Symmetry status:", symmetry_check)
print("Positive semi-definiteness status:", positive_definite_check)