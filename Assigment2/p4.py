# batch size 2077
import matplotlib.pyplot as plt

# Your data
epochs = list(range(1, 101))
accuracy = [
    23.50, 38.27, 46.74, 51.34, 58.33, 64.26, 66.51, 71.40, 74.28, 75.19,
    78.17, 80.56, 81.55, 83.23, 84.22, 85.57, 85.92, 88.73, 88.08, 88.93,
    89.94, 90.10, 91.79, 92.66, 92.56, 93.93, 92.19, 93.85, 92.79, 93.92,
    94.96, 95.71, 96.55, 96.72, 94.63, 95.44, 95.44, 96.26, 97.63, 96.47,
    96.86, 96.08, 96.62, 96.44, 95.21, 96.76, 96.98, 96.82, 97.27, 98.45,
    98.47, 98.55, 97.48, 97.51, 97.81, 95.02, 97.21, 97.15, 97.76, 97.48,
    98.00, 98.33, 98.42, 97.80, 96.32, 96.27, 96.90, 98.83, 98.88, 99.13,
    98.66, 96.91, 97.81, 97.99, 98.96, 99.61, 99.48, 99.28, 97.25, 97.09,
    98.38, 96.31, 94.86, 97.89, 98.80, 97.13, 97.33, 98.09, 97.34, 98.74,
    99.43, 99.42, 98.32, 97.99, 96.61, 95.49, 95.33, 97.77, 98.86, 98.77,
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy, marker='o', linestyle='-')
plt.title('Epoch vs. Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)

# Show the plot
plt.show()