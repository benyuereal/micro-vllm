import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 定义 Sigmoid 函数 =====================
def sigmoid(x):
    """
    Sigmoid 函数实现
    :param x: 输入（可以是单个数值/一维数组）
    :return: Sigmoid 输出（范围 0~1）
    """
    return 1 / (1 + np.exp(-x))

# ===================== 2. 生成多数据点 =====================
# 生成从 -5 到 5 的均匀数据点（步长 0.1）
x_points = np.arange(-5, 5.1, 0.1)
y_points = sigmoid(x_points)

# 打印关键数据点（和之前给的数值表一致）
key_x = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
key_y = sigmoid(np.array(key_x))
print("===== Sigmoid 关键数值表 =====")
for x, y in zip(key_x, key_y):
    print(f"x = {x:2d}, sigmoid(x) = {y:.4f}")

# ===================== 3. 可视化 Sigmoid 曲线 =====================
plt.figure(figsize=(10, 6))
# 绘制完整曲线
plt.plot(x_points, y_points, color='blue', linewidth=2, label='Sigmoid(x)')
# 标注关键数据点
plt.scatter(key_x, key_y, color='red', s=50, zorder=5, label='关键数据点')
# 标注坐标轴
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='y=0.5 (决策边界)')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
# 图表样式
plt.title('Sigmoid 函数曲线 (σ(x) = 1/(1+e⁻ˣ))', fontsize=14)
plt.xlabel('输入 x', fontsize=12)
plt.ylabel('输出 σ(x)', fontsize=12)
plt.xlim(-5.5, 5.5)
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
# 保存/显示图片
plt.savefig('sigmoid_curve.png', dpi=150)
plt.show()

# ===================== 4. 补充说明 =====================
print("\n===== Sigmoid 几何意义 =====")
print("1. 把整个实数轴 (-∞,+∞) 平滑压缩到 (0,1) 区间")
print("2. x=0 时输出 0.5，是二分类的决策边界")
print("3. x 越大越接近 1，x 越小越接近 0")