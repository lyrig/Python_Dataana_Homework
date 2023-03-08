'''
Date: 2023-02-27 09:43:07
LastEditTime: 2023-02-28 20:16:20
'''
import math
import matplotlib.pyplot as plt


def cacluator1(n):
    ans = []
    for x in range(0, 10, 0.1):
        tmp = 0
        for i in range(n):
            tmp += (((-1)**i) * (x**i)) / (math.factorial(i))
            #print(tmp)
        ans.append(tmp)
    return ans



def cacluator2(n):
    ans = []
    for x in range(0, 10, 0.1):
        memory = [1]
        tmp = 1
        for i in range(1, n):
            memory.append((-1) * memory[i - 1] * x / i)
            tmp += (-1) * memory[i - 1] * x / i
        ans.append(tmp)
    return ans


def cacluator3(n):
    ans = []
    for x in range(0, 10, 0.1):
        memory = [1]
        tmp = 1
        for i in range(1, n):
            memory.append(memory[i - 1] * x / i)
            tmp += memory[i - 1] * x / i
        ans.append(1/tmp)
    return ans

print(math.sqrt(30))
plt.figure(figsize=(20, 10), dpi=100)
x_arix = [x for x in range(0, 100, 1)]
y1 = cacluator1(100)
y2 = cacluator2(100)
y3 = cacluator3(100)
y4 = [x for x in range(0, 10, 0.1)]
y4 = list(map(lambda x : math.exp(-x), y4))
print(y1)
plt.plot(x_arix, y1, c='red', label="y1")
plt.plot(x_arix, y2, c='green', linestyle='--', label="y2")
plt.plot(x_arix, y3, c='blue', linestyle='-.', label="y3")
plt.plot(x_arix, y4, c='yellow', linestyle=':', label="y4")
plt.scatter(x_arix, y1, c='red')
plt.scatter(x_arix, y2, c='green')
plt.scatter(x_arix, y3, c='blue')
plt.scatter(x_arix, y4, c='yellow')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("x取值", fontdict={'size': 16})
plt.ylabel("计算结果", fontdict={'size': 16})
plt.title("计算结果对比", fontdict={'size': 20})
plt.show()