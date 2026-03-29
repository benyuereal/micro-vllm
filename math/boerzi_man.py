import numpy as np
import matplotlib.pyplot as plt
import platform

# ====================== 核心修复：自动适配中文字体，解决乱码 ======================
sys_name = platform.system()
if sys_name == "Windows":
    # Windows系统用黑体
    plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
elif sys_name == "Darwin":
    # macOS系统用苹方
    plt.rcParams["font.family"] = ["PingFang SC", "Arial Unicode MS", "sans-serif"]
elif sys_name == "Linux":
    # Linux系统用文泉驿微米黑
    plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "sans-serif"]
# 解决负号显示异常问题
plt.rcParams["axes.unicode_minus"] = False

# 全局设置：图片清晰度与字体大小
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 8

# 创建1行3列的子图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ====================== 图1：本源玻尔兹曼分布（能量为自变量）======================
ax1 = axes[0]
# 能量轴：0到10 kBT
E = np.linspace(0, 10, 500)
# 不同温度下的分布（相对温度，T=1为基准，T=3为高温）
p_T1 = np.exp(-E)  # T=1，kBT=1
p_T3 = np.exp(-E/3) # T=3，kBT=3

# 归一化处理
p_T1 = p_T1 / np.sum(p_T1)
p_T3 = p_T3 / np.sum(p_T3)

ax1.plot(E, p_T1, label='低温 T=1', color='#1f77b4', linewidth=2)
ax1.plot(E, p_T3, label='高温 T=3', color='#ff7f0e', linewidth=2)

ax1.set_title('图1：本源玻尔兹曼分布（能量为自变量）', fontweight='bold')
ax1.set_xlabel('能量 E (单位: $k_B T$)')
ax1.set_ylabel('归一化占据概率 p(E)')
ax1.legend()
ax1.grid(alpha=0.3)

# ====================== 图2：麦克斯韦-玻尔兹曼速率分布 ======================
ax2 = axes[1]
# 速率轴：0到5倍最概然速率vp
v = np.linspace(0, 5, 500)
# 麦克斯韦速率分布核心公式
def maxwell_dist(v, T):
    return (v**2) * np.exp(-v**2 / T)

p_v_T1 = maxwell_dist(v, T=1)
p_v_T3 = maxwell_dist(v, T=3)

# 归一化处理
p_v_T1 = p_v_T1 / np.sum(p_v_T1)
p_v_T3 = p_v_T3 / np.sum(p_v_T3)

ax2.plot(v, p_v_T1, label='低温 T=1', color='#1f77b4', linewidth=2)
ax2.plot(v, p_v_T3, label='高温 T=3', color='#ff7f0e', linewidth=2)

ax2.set_title('图2：麦克斯韦-玻尔兹曼速率分布', fontweight='bold')
ax2.set_xlabel('速率 v (单位: 最概然速率 $v_p$)')
ax2.set_ylabel('归一化概率密度 p(v)')
ax2.legend()
ax2.grid(alpha=0.3)

# ====================== 图3：双势阱下的位置空间玻尔兹曼分布 ======================
ax3 = axes[2]
# 位置轴
x = np.linspace(-2, 2, 500)
# 双势阱势能函数
E_x = x**4 - 2 * x**2
# 位置空间玻尔兹曼分布计算
def position_dist(x, T):
    E_x = x**4 - 2 * x**2
    return np.exp(-E_x / T)

p_x_T1 = position_dist(x, T=0.2) # 低温，粒子集中在势阱
p_x_T3 = position_dist(x, T=1)   # 高温，粒子可越过势垒

# 归一化处理
p_x_T1 = p_x_T1 / np.sum(p_x_T1)
p_x_T3 = p_x_T3 / np.sum(p_x_T3)

# 双Y轴：同步绘制势能曲线
ax3_twin = ax3.twinx()
ax3_twin.plot(x, E_x, color='gray', linestyle='--', label='势能 E(x)', alpha=0.7)
ax3_twin.set_ylabel('势能 E(x)', color='gray')

# 绘制概率分布曲线
ax3.plot(x, p_x_T1, label='低温 T=0.2', color='#1f77b4', linewidth=2)
ax3.plot(x, p_x_T3, label='高温 T=1', color='#ff7f0e', linewidth=2)

ax3.set_title('图3：双势阱位置空间分布（U形误解来源）', fontweight='bold')
ax3.set_xlabel('位置 x')
ax3.set_ylabel('归一化位置概率 p(x)')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')
ax3.grid(alpha=0.3)

# 自动调整布局，避免标签重叠
plt.tight_layout()
plt.show()