import matplotlib.pyplot as plt

# val/accuracy 的 SimpleValue 值
val_accuracy_values = [
    0.2233283519744873,
    0.35783547163009644,
    0.4763959050178528,
    0.5847426652908325,
    0.6196001768112183,
    0.725781261920929,
    0.6931927800178528,
    0.4579388499259949,
    0.7005974054336548,
    0.7051470875740051
]

# 对应的 epoch 值
epochs = list(range(1, len(val_accuracy_values) + 1))

# 设置图表样式
plt.style.use('seaborn-whitegrid')

# 创建颜色映射
palette = plt.get_cmap('Set1')

# 绘制 val/accuracy 曲线
plt.plot(epochs, val_accuracy_values, marker='o', color=palette(1), linewidth=2.5, alpha=0.9, label='val/accuracy')

# 添加标题和轴标签
plt.title('Validation Accuracy over Epochs', loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')

# 显示图例
plt.legend()

# 仅显示横向网格线
plt.gca().yaxis.grid(True)
plt.gca().xaxis.grid(False)

# 显示图表
plt.show()
