import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.arange(0, 9, 1) # 横坐标数据为从0到9之间，步长为0.1的等差数组
# 纵坐标数据为 x 对应的 y 值
y=[0.114,
0.070,
0.013,
0.029,
0.015,
0.017,
0.017,
0.012,
0.011]

plt.ylabel('training_loss') # 横坐标轴的标题
plt.xlabel('Epoch') # 纵坐标轴的标题
plt.legend() # 显示图例, 图例中内容由 label 定义
plt.title('training loss/Epoch')
# 生成图形
plt.plot(x, y)

# 显示图形
#plt.show()

plt.savefig("training_loss.jpg")
