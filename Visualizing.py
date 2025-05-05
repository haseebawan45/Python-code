import matplotlib.pyplot as plt

# Visualizing the dataset
plt.hist(data, bins=5, color='blue', alpha=0.7, rwidth=0.85)
plt.title('Histogram of Dataset')
plt.xlabel('Data Values')
plt.ylabel('Frequency')
plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.2f}')
plt.legend()
plt.show()
