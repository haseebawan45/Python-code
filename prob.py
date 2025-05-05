import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.stats import levene, bartlett

# Sample dataset
data = [10, 12, 23, 23, 16, 23, 21, 16]

# 1. Basic Computations of Variance and Standard Deviation
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

# 2. Visualizing Data Dispersion
plt.hist(data, bins=5, color='blue', alpha=0.7, rwidth=0.85)
plt.title('Histogram of Dataset')
plt.xlabel('Data Values')
plt.ylabel('Frequency')
plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.2f}')
plt.legend()
plt.show()

# 3. Comparing Variance and Mean Absolute Deviation
mad = sum(abs(x - mean) for x in data) / len(data)
print("Mean Absolute Deviation (MAD):", mad)
print("Variance vs. MAD ratio:", variance / mad)

# 4. Variance and Standard Deviation in Real-World Contexts
# Simulating stock returns
returns = [0.02, 0.03, -0.01, 0.04, -0.02, 0.01]
mean_return = np.mean(returns)
variance_returns = np.var(returns)
std_dev_returns = np.std(returns)

print("Mean Return:", mean_return)
print("Variance of Returns:", variance_returns)
print("Standard Deviation of Returns:", std_dev_returns)

# Interpreting risk
if std_dev_returns > 0.02:
    print("High risk stock based on standard deviation.")
else:
    print("Low risk stock based on standard deviation.")

# 5. Large Dataset Analysis
# Generating a large dataset
large_data = np.random.normal(50, 10, 100000)

# Computing variance and standard deviation
large_variance = np.var(large_data)
large_std_dev = np.std(large_data)

print("Variance of Large Dataset:", large_variance)
print("Standard Deviation of Large Dataset:", large_std_dev)

# Visualizing the large dataset
plt.figure(figsize=(10, 6))
plt.hist(large_data, bins=50, color='green', alpha=0.7)
plt.title('Histogram of Large Dataset')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# 6. Advanced Applications
# Feature Scaling in Machine Learning
data_ml = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_ml)

print("Original Data:\n", data_ml)
print("Scaled Data:\n", scaled_data)

# Ridge Regression Example
X = np.random.rand(100, 3)
y = X @ np.array([1.5, -2.0, 3.0]) + np.random.normal(0, 0.5, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
print("Ridge Coefficients:", model.coef_)

# 7. Statistical Tests for Variance
# Testing equality of variances between two groups
group1 = [10, 20, 30, 40, 50]
group2 = [15, 25, 35, 45, 55]

levene_stat, levene_p = levene(group1, group2)
print("Levene's Test Statistic:", levene_stat, "P-value:", levene_p)

bartlett_stat, bartlett_p = bartlett(group1, group2)
print("Bartlett's Test Statistic:", bartlett_stat, "P-value:", bartlett_p)