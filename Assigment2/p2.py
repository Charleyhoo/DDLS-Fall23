import matplotlib.pyplot as plt

# 将Epoch和Accuracy数据提取出来
epochs = range(1, 101)  # 从1到100的Epoch
accuracy = [
    34.70, 52.18, 64.04, 71.55, 75.88, 80.00, 81.88, 84.01, 85.42, 86.78,
    88.14, 88.79, 89.66, 90.85, 91.25, 92.01, 92.83, 92.52, 92.96, 93.60,
    93.77, 94.22, 94.16, 94.53, 94.94, 94.46, 95.02, 95.11, 95.19, 94.79,
    95.31, 95.25, 95.45, 95.44, 95.49, 95.98, 95.62, 95.84, 95.84, 95.88,
    96.18, 96.18, 96.09, 96.03, 95.59, 96.07, 96.04, 96.37, 96.25, 96.61,
    95.42, 96.68, 96.38, 96.16, 96.42, 96.09, 96.06, 96.38, 96.23, 96.26,
    96.58, 96.17, 96.55, 96.80, 96.07, 96.06, 96.82, 96.11, 96.29, 96.98,
    96.74, 96.45, 96.57, 96.65, 96.52, 96.65, 96.43, 96.09, 96.85, 96.87,
    96.88, 96.41, 96.52, 97.04, 96.54, 96.51, 97.04, 96.94, 96.11, 96.40,
    96.98, 91.05, 96.75, 96.29, 96.79, 97.09, 96.87, 97.04, 96.55, 96.79
]

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy, marker='o', linestyle='-')
plt.title('Epoch vs Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.xticks(range(0, 101, 10))
plt.yticks(range(0, 101, 10))
plt.ylim(0, 100)
plt.xlim(0, 101)
plt.show()
