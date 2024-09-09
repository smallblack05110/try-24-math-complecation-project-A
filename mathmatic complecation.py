import numpy as np
import pprint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd
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

#定义变量
rou = []  # 存储极径
radians = []  # 存储弧度
dra_note_site = np.zeros(shape=(224,3)) #存储节点位置
global r_initial
global theta_initial
beta = 8.8 - np.sqrt(77.44 - (2 * alpha1 * v * t) / np.sqrt((1 + alpha1) ** 2))  # 龙头转过的弧度
r_initial = -alpha1 * beta + 8.8
theta_initial = (r_initial - 8.8) / (-alpha1)
dra_note_x = [] #记录每节点x位置
dra_note_y = [] #记录每节点y位置
dra_note_v = [] #记录每节点速度

# 定义方程
def equation(r_1, r_0,delta):
    return r_0 ** 2 + r_1 ** 2 - 2 * r_0 * r_1 * np.cos((1 / alpha1) * (r_1 - r_0)) - delta ** 2
def equation2(r_0,r_1,v_0,v_1):
    return (r_0 - r_1 * np.cos((r_1 - r_0) / alpha1) - (r_1 * r_0) / alpha1 * np.sin((r_1 - r_0) / alpha1)) * (v_0 / np.sqrt(1 + (r_0 ** 2) / alpha1 **2))
    + (r_1 - r_0 * np.cos((r_1 - r_0) / alpha1) + (r_1 * r_0) / alpha1 * np.sin((r_1 - r_0) / alpha1)) * (v_1 / np.sqrt(1 + (r_1 ** 2 / alpha1 ** 2)))

#计算龙头及每节节点实时位置
def cal_dra_note_site(r_0,theta_intial):
    rou.append(r_0)
    radians.append(theta_intial)
    # 循环求解r_1
    delta = delta_r1
    for i in range(223):
        initial_guess = r_0 + 0.01
        if i == 1:
           delta = delta_r2
        # 使用fsolve求解r_1
        r_1_solution = fsolve(equation, initial_guess, args=(r_0,delta))
        # 将解得的 r_1 保存到 solutions 列表
        rou.append(r_1_solution[0])
        radians.append(-((r_1_solution[0] - 8.8) / (-alpha1)))
        # 将当前解得的 r_1 作为新的 r_0
        r_0= r_1_solution[0]
def cal_dra_note_v(r_0,r_1,v_0):
        initial_guess = v_0 * 0.99999
        v_1_solution = fsolve(equation2,initial_guess,args=(r_0,r_1,v_0))
        dra_note_v.append(abs(v_1_solution[0]))
        v_0 =abs( v_1_solution[0])

for i in range(0,2,60):
    t = i
    beta = (8.8 - np.sqrt(77.44 - (2 * alpha1 * v * t) / np.sqrt(1 + alpha1 ** 2)))/alpha1  # 龙头转过的弧度
    r_initial = -alpha1 * beta + 8.8
    theta_initial = (r_initial - 8.8) / (-alpha1)
    cal_dra_note_site(r_initial,theta_initial)
dra_note_v.append(1.0)
for i in range(0,len(rou) - 1):
    cal_dra_note_v(rou[i],rou[i+1],1)

for r,theta in zip(rou,radians):
    dra_note_x.append(r * np.cos(theta))
    dra_note_y.append(r * np.sin(theta))
df = pd.DataFrame({
        'x':dra_note_x,
        'y':dra_note_y,
        '速度':dra_note_v,
    })
df.to_excel('dra_note_site.xlsx', index=False)

# 画出各节点初始位置
fig,axe = plt.subplots(subplot_kw={'projection': 'polar'})
axe.grid(True)
plt.plot(radians,rou,linestyle='-',marker='.',markersize=5)
axe.set_title("Dragon_note_Site", va='bottom')
plt.show()
# fig,axe = plt.subplots()
# axe.grid(True)
# plt.axis('equal')
# plt.plot(dra_note_x,dra_note_y)
# plt.show()