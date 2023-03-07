from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# a, b = 1,2
# mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
# fig, ax = plt.subplots(1, 1)
#
# x = np.linspace(beta.ppf(0.01, a, b),beta.ppf(0.6, a, b), 100)
# ax.plot(x, beta.pdf(x, a, b),'r-', lw=5, alpha=0.6, label='beta pdf')
# plt.show()
#
a = 1
b = 2
# X = np.arange(0, 1, 0.001)
# y = stats.beta.pdf(X,1,2)
# #绘图
# plt.plot(X,y)
# #x轴文本
# plt.xlabel('随机变量：x')
# #y轴文本
# plt.ylabel('概率：y')
#
# #网格
# plt.grid()
# #显示图形
# plt.show()

r = beta.rvs(a, b, size=1000)
group = [0,0.1,0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]


plt.hist(r, group, histtype='bar', rwidth=0.8)

plt.legend()

plt.xlabel('salary-group')
plt.ylabel('salary')



plt.show()
