import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 定义自然指数函数 =====================
def exp(x):
    """计算自然指数 e^x（numpy 实现，精度更高）"""
    return np.exp(x)

# ===================== 2. 生成关键数据点 =====================
# 覆盖 Softmax 常用的输入范围（-5 到 5）
x_values = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
exp_values = exp(x_values)

# 打印数值表（保留4位小数）
print("===== 自然指数 e^x 数值表 =====")
for x, ex in zip(x_values, exp_values):
    print(f"e^({x}) = {ex:.4f}")

# ===================== 3. 可视化 e^x 曲线 =====================
# 生成密集数据点（平滑曲线）
x_dense = np.arange(-5, 5.1, 0.1)
exp_dense = exp(x_dense)

plt.figure(figsize=(10, 6))
# 绘制平滑曲线
plt.plot(x_dense, exp_dense, color='blue', linewidth=2, label='$e^x$ 曲线')
# 标注关键数据点
plt.scatter(x_values, exp_values, color='red', s=60, zorder=5, label='关键数据点')
# 辅助线
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='$e^0=1$')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
# 图表样式
plt.title('自然指数函数 $y=e^x$（Softmax 核心基础）', fontsize=14)
plt.xlabel('输入 x', fontsize=12)
plt.ylabel('输出 $e^x$', fontsize=12)
plt.xlim(-5.5, 5.5)
plt.ylim(-0.5, np.max(exp_dense)+1)  # 留出顶部空间
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
# 保存图片
plt.savefig('exp_function.png', dpi=150)
plt.show()

# ===================== 4. e^x 在 Softmax 中的核心作用 =====================
print("\n===== e^x 在 Softmax 中的关键作用 =====")
print("1. 放大输入差异：大的 x 会让 e^x 急剧增大，小的 x 会让 e^x 趋近于 0")
print("2. 非负性：e^x 永远大于 0，保证 Softmax 输出为正概率")
print("3. 单调性：x 越大，e^x 越大，保证 Softmax 输出和输入正相关")
print("示例：输入 [2,1] → e^2=7.3891, e^1=2.7183 → 比值约 2.72:1 → Softmax 输出 [0.7311, 0.2689]")