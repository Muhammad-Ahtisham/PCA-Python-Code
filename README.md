# PCA-Python-Code
This code implements Principal Component Analysis (PCA) from scratch using Python. Let’s go step by step to understand the logic behind it.

## Source File:
PCA.ipynb

# Understanding PCA

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms data into a new coordinate system where the greatest variance lies on the first principal component, the second greatest variance on the second principal component, and so on.

## Steps in PCA:

1. Compute the mean vector of the dataset.
2. Center the dataset by subtracting the mean vector.
3. Compute the covariance matrix of the centered data.
4. Find the eigenvectors (principal components) of the covariance matrix.
5. Project the original data onto the principal components.

---

## Function Breakdown

Now, let's analyze each function in the code.

### 2.1 Computing the Mean Vector

```python
def mean_vector(data):
    """Computes the mean vector for a dataset"""
    n = len(data)  # Number of samples (rows)
    m = len(data[0])  # Number of features (columns)
    means = [sum(data[i][j] for i in range(n)) / n for j in range(m)]
    return means
```

- Computes the mean of each column (feature).
- The output is a list containing the mean of each feature.

✅ Example:

```python
mean_vector([[1,2], [3,4], [5,6]])  # Output: [3.0, 4.0]
```

### 2.2 Centering the Data

```python
def subtract_mean(data, mean):
    """Centers the dataset by subtracting the mean"""
    return [[data[i][j] - mean[j] for j in range(len(mean))] for i in range(len(data))]
```

- This function subtracts the mean from each feature, making the dataset centered at zero.

✅ Example:

```python
subtract_mean([[1,2], [3,4], [5,6]], [3,4])  
# Output: [[-2, -2], [0, 0], [2, 2]]
```

### 2.3 Transposing a Matrix

```python
def transpose(matrix):
    """Computes the transpose of a matrix"""
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
```

- Transposes the given matrix (rows become columns, columns become rows).

✅ Example:

```python
transpose([[1, 2, 3], [4, 5, 6]])  
# Output: [[1, 4], [2, 5], [3, 6]]
```

### 2.4 Matrix Multiplication

```python
def matrix_multiply(A, B):
    """Multiplies two matrices A and B"""
    result = [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
    return result
```

- Multiplies two matrices using the dot product formula.

✅ Example:

```python
matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
# Output: [[19, 22], [43, 50]]
```

### 2.5 Computing the Covariance Matrix

```python
def covariance_matrix(data):
    """Computes the covariance matrix"""
    n = len(data)
    data_T = transpose(data)
    cov_matrix = [[sum(data_T[i][k] * data_T[j][k] for k in range(n)) / (n - 1) for j in range(len(data_T))] for i in range(len(data_T))]
    return cov_matrix
```

- Computes the covariance matrix, which tells how much features vary together.

✅ Example:

```python
covariance_matrix([[-2, -2], [0, 0], [2, 2]])  
# Output: [[4.0, 4.0], [4.0, 4.0]]
```

### 2.6 Finding the Dominant Eigenvector using Power Iteration

```python
import math

def power_iteration(A, num_simulations=100):
    """Finds the dominant eigenvector using power iteration"""
    n = len(A)
    b_k = [1] * n  # Initial guess
    
    for _ in range(num_simulations):
        b_k1 = [sum(A[i][j] * b_k[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(x ** 2 for x in b_k1))
        b_k = [x / norm for x in b_k1]
    
    return b_k
```

- Uses Power Iteration to find the dominant eigenvector (corresponding to the largest eigenvalue).
- The eigenvector represents the principal direction of variance in the dataset.

### 2.7 PCA Function

```python
def pca(data, num_components=2):
    """Performs PCA on a dataset and reduces to num_components dimensions"""
    mean = mean_vector(data)  # Step 1: Compute mean
    centered_data = subtract_mean(data, mean)  # Step 2: Center the data
    cov_matrix = covariance_matrix(centered_data)  # Step 3: Compute covariance matrix
    
    eigenvectors = []
    for _ in range(num_components):
        ev = power_iteration(cov_matrix)  # Step 4: Compute eigenvectors
        eigenvectors.append(ev)
    
    transformed_data = matrix_multiply(centered_data, transpose(eigenvectors))  # Step 5: Project data
    return transformed_data
```

- Calls all the previous functions in sequence to reduce dimensionality to `num_components`.

---

## Example Usage

```python
data = [
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2, 1.6],
    [1, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
]

reduced_data = pca(data, num_components=2)
print("Reduced Data:", reduced_data)
```

- This applies PCA to a 2D dataset and reduces it to 2D principal components.

---

## Summary of Steps

1. Compute mean of the dataset.
2. Center the dataset by subtracting the mean.
3. Compute the covariance matrix of the centered data.
4. Find the principal components (eigenvectors) using power iteration.
5. Project data onto the principal components to get the reduced dataset.

## Improvements

- The Power Iteration method finds only the dominant eigenvector, but for multiple components, we should deflate the matrix after each iteration.
- The matrix multiplication can be optimized using NumPy.
- The covariance matrix computation can be simplified using NumPy’s `np.cov()` function.

