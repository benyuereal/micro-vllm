import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===================== 1. 分步实现 Softmax（显式展示 e^x 步骤） =====================
def softmax_step_by_step(z):
    """
    分步计算 Softmax，展示每一步（重点突出 e^x）
    :param z: 输入向量
    :return: (e^z 结果, 归一化后的 Softmax 结果)
    """
    z = np.array(z)
    # 步骤1：计算 e^z（核心）
    exp_z = np.exp(z)
    # 步骤2：计算 e^z 的和
    sum_exp_z = np.sum(exp_z)
    # 步骤3：归一化（除以和）
    softmax_res = exp_z / sum_exp_z
    return exp_z, softmax_res

# ===================== 2. 二维 Softmax 计算（关联 e^x） =====================
print("===== 二维 Softmax 分步计算（重点看 e^z） =====")
z_2d = [2, 1]  # 经典示例
exp_z_2d, softmax_2d = softmax_step_by_step(z_2d)
print(f"输入向量 z = {z_2d}")
print(f"步骤1：计算 e^z = {exp_z_2d.round(4)}")
print(f"步骤2：e^z 求和 = {np.sum(exp_z_2d).round(4)}")
print(f"步骤3：Softmax = e^z / sum(e^z) = {softmax_2d.round(4)}")
print(f"验证：Softmax 和 = {np.sum(softmax_2d).round(4)}")

# 额外测试多组输入
z_2d_list = [[3, -1], [0, 0], [-2, 5]]
print("\n===== 更多二维 Softmax 示例 =====")
for z in z_2d_list:
    exp_z, sm = softmax_step_by_step(z)
    print(f"z={z} → e^z={exp_z.round(4)} → Softmax={sm.round(4)} (和={np.sum(sm).round(4)})")

# ===================== 3. 三维 Softmax 计算 =====================
print("\n===== 三维 Softmax 分步计算 =====")
z_3d = [1, 2, 3]
exp_z_3d, softmax_3d = softmax_step_by_step(z_3d)
print(f"输入向量 z = {z_3d}")
print(f"步骤1：计算 e^z = {exp_z_3d.round(4)}")
print(f"步骤2：e^z 求和 = {np.sum(exp_z_3d).round(4)}")
print(f"步骤3：Softmax = {softmax_3d.round(4)}")
print(f"验证：Softmax 和 = {np.sum(softmax_3d).round(4)}")

# ===================== 4. 可视化：二维 Softmax（关联 e^x 与概率） =====================
plt.figure(figsize=(10, 5))

# 子图1：e^z 对比
plt.subplot(1, 2, 1)
x_labels = ['z1', 'z2']
plt.bar(x_labels, z_2d, color='lightblue', label='原始输入 z')
plt.bar(x_labels, exp_z_2d, color='orange', alpha=0.7, label='$e^z$')
plt.title(f'输入 {z_2d} → $e^z$ 放大差异', fontsize=12)
plt.ylabel('数值大小')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：Softmax 概率分布
plt.subplot(1, 2, 2)
plt.bar(x_labels, softmax_2d, color='green', alpha=0.7, label='Softmax 概率')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
plt.title('Softmax 归一化（和为1）', fontsize=12)
plt.ylabel('概率')
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('softmax_vs_exp.png', dpi=150)
plt.show()

# ===================== 5. 可视化：三维 Softmax 概率分布 =====================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制概率分布
x = [0, 1, 2]
y = [0, 0, 0]
z = softmax_3d
ax.bar3d(x, y, 0, 0.5, 0.5, z, color=['red', 'green', 'blue'], alpha=0.7)

# 标注
ax.set_xticks(x)
ax.set_xticklabels(['类别1', '类别2', '类别3'])
ax.set_ylabel('')
ax.set_zlabel('概率')
ax.set_title(f'三维 Softmax 概率分布（输入 {z_3d}）', fontsize=12)
plt.tight_layout()
plt.savefig('softmax_3d_dist.png', dpi=150)
plt.show()

# ===================== 核心总结 =====================
print("\n===== Softmax 核心逻辑 =====")
print("1. 核心：用 e^z 放大输入向量的差异（大的数更大，小的数更小）")
print("2. 归一化：将 e^z 除以总和，得到和为1的概率分布")
print("3. 关键：e^z 保证了输出非负，且和输入的大小正相关")