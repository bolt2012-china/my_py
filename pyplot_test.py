import matplotlib.pyplot as plt
import numpy as np
import math
x = np.linspace(-2*math.pi, 2*math.pi, 100)
y = np.sin(x)/x
plt.plot(x,y,color='blue',label='sin(x)/x',linestyle='--',linewidth=1)
# 设置 x，y 轴的范围以及 label 标注
plt.xlim(-2*math.pi,2*math.pi)
plt.ylim(-0.5,1.1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin(x)/x')
# 设置坐标轴 gca() 获取坐标轴信息
ax=plt.gca()

# __接下来通过移动边框来让边框作为坐标轴__
# 使用.spines设置边框：x轴；将右边颜色设置为 none。
# 使用.set_position设置边框位置：y=0的位置；（位置所有属性：outward，axes，data）
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# 移动坐标轴
# 将 bottom 即是 x 坐标轴设置到 y=0 的位置。
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))

# 将 left 即是 y 坐标轴设置到 x=0 的位置。
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

# 设置坐标轴刻度线
# Tick X 范围 (-1，2) Tick Label(-1，-0.25，0.5，1.25，2) 刻度数量 5 个
new_ticks=np.linspace(-2*math.pi,2*math.pi,5)
plt.xticks(new_ticks)

# Tick Y 范围(-2.2,-1,1,1.5,2.4) ，Tick Label (-2.2, -1, 1, 1.5, 2.4) 别名(下面的英文)
#plt.yticks([-2.2,-1,1,1.5,2.4],
          #[r'$really\ bad$',r'$bad$',r'$normal$',r'$good$',r'$really\ good$'])
plt.yticks(np.linspace(-0.5,1.1,5))

plt.legend()    # 显示图例
plt.show()      # 显示图像






