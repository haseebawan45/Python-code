import numpy as np

# Sample dataset
data = [10, 12, 23, 23, 16, 23, 21, 16]

# Manual calculation of variance
mean = sum(data) / len(data)
squared_deviations = [(x - mean) ** 2 for x in data]
variance = sum(squared_deviations) / len(data)
standard_deviation = variance ** 0.5

print("Mean:", mean)
print("Variance (manual):", variance)
print("Standard Deviation (manual):", standard_deviation)

# Using NumPy for verification
np_variance = np.var(data)
np_std_dev = np.std(data)

print("Variance (NumPy):", np_variance)
print("Standard Deviation (NumPy):", np_std_dev)
