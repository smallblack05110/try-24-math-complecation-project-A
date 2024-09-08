import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas
np.set_printoptions(precision=6)
#定义常量
p = 0.55 #螺距
v_dra_head = 1.0 #龙头速度
r_0 = 8.8 #径向矢量
total_num = 223 #板凳节数
l_dra_head = 3.41 #龙头长度
l_dra_body = 2.2 #龙身长度、
delta_r1 = 3.41 - 0.275 * 2
delta_r2 = 2.2 - 0.275 * 2
alpha1 = 0.55/ (2 * np.pi) #定义常量alpha1
v = 1.0 #定义龙头速度
t = 0  #定义时间
beta = 8.8 - np.sqrt(77.44 - (2 * v * t) / np.sqrt((1 + alpha1) ** 2)) #龙头转过的弧度

# 定义方程
def equation(r_1, r_0):
    return r_0 ** 2 + r_1 ** 2 - 2 * r_0 * r_1 * np.cos((1 / alpha1) * (r_1 - r_0)) - delta_r1 ** 2
# 初始猜测值
initial_guess = 9  # 初始r_1猜测

# 保存所有r_1的解
solutions = []
solutions2 = []
# 循环求解r_1
for i in range(224):
    if i == 1:
        delta_r1 = delta_r2
    # 使用fsolve求解r_1
    r_1_solution = fsolve(equation, initial_guess, args=(r_0))

    # 将解得的 r_1 保存到 solutions 列表
    solutions.append(r_1_solution[0])
    solutions2.append(((r_1_solution[0] - 8.8) / (-alpha1)))

    # 将当前解得的 r_1 作为新的 r_0
    r_0 = r_1_solution[0]

    # 更新初始猜测值，以便下次求解
    initial_guess = r_0 + 0.01

#定义数组表示各节点位置（极坐标）
r = []
theta = []
dra_note_site = np.zeros(shape=(224,3))
dra_note_site[0][1] = 8.80
dra_note_site[0][2] = 0
dra_note_site[223][0] = 224
for i in range(223):
    dra_note_site[i][0] = i+1
    dra_note_site[i+1][1] = solutions[i]
    dra_note_site[i+1][2] = solutions2[i]
print(dra_note_site)
#定义数组表示各节点位置（x，y）
dra_note_site2 = np.zeros(shape=(224,3))
for i in range(224):
    dra_note_site2[i][0] = dra_note_site[i][0]
    dra_note_site2[i][1] = dra_note_site[i][1] * np.cos(dra_note_site[i][2])
    r.append(dra_note_site[i][1])
    dra_note_site2[i][2] = - dra_note_site[i][1] * np.sin(dra_note_site[i][2])
    theta.append(-dra_note_site[i][2])
print(dra_note_site2)

#画出各节点初始位置
fig,axe = plt.subplots(subplot_kw={'projection': 'polar'})
axe.grid(True)
plt.plot(theta,r,linestyle='-',marker='.',markersize=5)
axe.set_title("Dragon_note_Site", va='bottom')
plt.show()