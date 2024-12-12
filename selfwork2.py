import time
import math
import random
from itertools import count


def Brute_force(n):
    time_start = time.time()
    count = 0
    for i in range(2,n+1):
        prime = True    #假设所有的都是素数
        for j in range(2,i):
            if i % j == 0:
                prime = False
                break
        if prime:   #将两层判断相关联，保证count可以正确自加
            count += 1
    time_end = time.time()
    return time_end - time_start,count

def Optimized_Brute_force(n):
    time_start = time.time()
    count = 0
    for j in range(2,n+1):
        for i in range(2,math.floor(math.sqrt(j+1))+1):
            if j % i == 0:
                break
            count += 1
    time_end = time.time()
    return time_end - time_start,count

def Optimized_factor(n):
    time_start = time.time()
    count = 2   #2,3不是对应形式但仍是质数，提前加进来
    lst = []    #按要求缩小排查范围
    for i in range(1,n+1):
        if (i+1)%6==0 or (i-1)%6==0:
            lst.append(i)
        else:
            continue
    for j in lst:   #使用方法2进一步压缩
        for i in range(2,math.floor(math.sqrt(j+1))+1):
            if j % i == 0:
                break
        else:
            count += 1
    time_end = time.time()
    return time_end - time_start,count-1    #多加了一个

def Seive_Eratosthene(n):
    time_start = time.time()
    p = 2   #第一个质数是2
    numbers = [True]*(n+1)    #先将0-n的每个数都标记为质数
    numbers[0], numbers[1] = False, False   #0,1不是质数
    while p*p <= n:
        if numbers[p] == True:  #p是质数
            for i in range(p*p, n+1, p):
                numbers[i] = False
        p += 1
    time_end = time.time()
    return time_end - time_start,sum(numbers)

def Miller_Rabin(n):
    time_start = time.time()
    numbers = [True] * (n + 1)  #先将0-n的每个数都标记为质数
    numbers[0], numbers[1] = False, False #0,1不是质数
    for i in range(4, n+1): #2,3都是素数，所以直接从4开始
        if i%2 == 0:    #偶数肯定不是素数
            numbers[i] = False
        if numbers[i] == True:
            t, u = 0, i-1   #将i-1表示为2^t*u
            while u%2 == 0:
                u = u//2
                t += 1   #不断乘2
            for j in range(5):  #测试5次
                a = random.randint(1, i-1)
                m = pow(a,u,i)
                if m == 1 or m == i-1:
                    continue
                for _ in range(t-1):
                    m = (m*m)%i
                    if m == i-1:
                        break
                else:
                    numbers[i] = False
                    break
    time_end = time.time()
    return time_end - time_start,sum(numbers)

print(Brute_force(500000))
print(Optimized_Brute_force(500000))
print(Optimized_factor(500000))
print(Seive_Eratosthene(500000))
print(Miller_Rabin(500000))


