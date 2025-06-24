
import numpy as np
array = np.random.uniform(0, 10, 20)
print("Original Array:", array)


array_rounded = np.round(array, 2)
print("Rounded Array:", array_rounded)


min_val = np.min(array_rounded)
max_val = np.max(array_rounded)
median_val = np.median(array_rounded)
print(f"Min: {min_val}, Max: {max_val}, Median: {median_val}")


array_modified = np.where(array_rounded < 5, array_rounded**2, array_rounded)
print("Modified Array:", array_modified)


def numpy_alternate_sort(array):
    array_sorted = np.sort(array)
    result = np.empty_like(array_sorted)
    result[0::2] = array_sorted[:len(array)//2]
    result[1::2] = array_sorted[len(array)//2:][::-1]
    return result

alt_sorted = numpy_alternate_sort(array_rounded)
print("Alternately Sorted Array:", alt_sorted)
