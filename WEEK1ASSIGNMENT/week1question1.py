import numpy as np

array = np.random.randint(1, 51, size=(5, 4))
print("Original Array:\n", array)


anti_diagonal = [array[i, -i - 1] for i in range(min(array.shape))]
print("Anti-diagonal Elements:", anti_diagonal)


row_max = np.max(array, axis=1)
print("Maximum Value in Each Row:", row_max)


overall_mean = np.mean(array)
filtered_array = array[array <= overall_mean]
print("Filtered Array :", filtered_array)


def numpy_boundary_traversal(matrix):
    top = matrix[0, :]
    right = matrix[1:-1, -1]
    bottom = matrix[-1, ::-1]
    left = matrix[1:-1, 0][::-1]
    return list(top) + list(right) + list(bottom) + list(left)

boundary_elements = numpy_boundary_traversal(array)
print("Boundary Traversal (Clockwise):", boundary_elements)
