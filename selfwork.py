#1. Hanoi Problem
def Hanoi_plus(n,x,y,z):
    sources = [1]*n #source rod, 1表示放满
    count = 0
    if len(sources) == 1:  #当source rob只有一个，直接放到C
        print(f"{x}->{y}")
        print(f"{y}->{z}")
        count += 2
    else:
        count += Hanoi_plus(n-1,x,y,z)   #先将n-1个放到C
        count += 1
        print(f"{x}->{y}")   #把A剩下的第n个放到B
        count += Hanoi_plus(n-1,z,y,x)   #把C上的n-1个放到A
        count += 1   #把第n个从B放到C
        print(f"{y}->{z}")
        count += Hanoi_plus(n-1,x,y,z)    #把剩下的n-1个从A放到C，递归

    return count
print(Hanoi_plus(4,"A","B","C"))


#2. Josephus Circle
def problem1_circle(n):
    people = list(range(1,n+1))   #给人编号
    i = 0    #人的序号
    count = 1   #人报数
    while len(people) > 1:
        if count%2 == 0:
            people.pop(i)    #移除报偶数的
            count += 1
        else:
            i = (i+1) % len(people)
            count += 1
            continue
    return people[0]    #返回这个人一开始的报数
print(problem1_circle(4))

def problem2_circle(n):
    if n == 1:
        return 1
    else :
        if n%2 == 0:
            return 2*problem2_circle(n/2)-1   #套公式即可
        else :
            return 2*problem2_circle(n//2)+1
print(problem2_circle(5))

import math
def problem3_circle(n):
    m = math.floor(math.log2(n))      #套公式
    l = n-2**m
    return 2*l+1
print(problem3_circle(5))