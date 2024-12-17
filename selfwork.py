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


# 3.chess board problem
import numpy as np


def grid_cover_iterative(k: int, i: int, j: int):
    dic = {(1, 1): [[0, 4], [4, 4]], (1, 2): [[3, 0], [3, 3]], (2, 1): [[2, 2], [0, 2]], (2, 2): [[1, 1], [1, 0]]}
    position = {1: 4, 2: 3, 3: 2, 4: 1}

    stack = [(k, i, j)]
    grids = {}

    while stack:
        k, i, j = stack.pop()

        if k == 1:
            grids[(k, i, j)] = dic[(i, j)]
        else:
            if (k - 1, i, j) not in grids:
                stack.append((k, i, j))
                stack.append((k - 1, 2 ** (k - 1), 1))
                stack.append((k - 1, 1, 1))
                stack.append((k - 1, 1, 2 ** (k - 1)))
                continue

            if i <= 2 ** (k - 1) and j <= 2 ** (k - 1):  # 1
                grid = np.block([[grids[(k - 1, i, j)], grids[(k - 1, 2 ** (k - 1), 1)]],
                                 [grids[(k - 1, 1, 1)], grids[(k - 1, 1, 2 ** (k - 1))]]])
                grid[2 ** (k - 1)][2 ** (k - 1)-1], grid[2 ** (k - 1)][2 ** (k - 1)], grid[2 ** (k - 1)-1][
                    2 ** (k - 1)] = position[1], position[1], position[1]
            elif i <= 2 ** (k - 1) and j > 2 ** (k - 1):  # 2
                grid = np.block([[grids[(k - 1, 2 ** (k - 1), 2 ** (k - 1))], grids[(k - 1, i, j - 2 ** (k - 1))]],
                                 [grids[(k - 1, 1, 2 ** (k - 1))], grids[(k - 1, 1, 1)]]])
                grid[2 ** (k - 1)-1][2 ** (k - 1)-1], grid[2 ** (k - 1)][2 ** (k - 1)-1], grid[2 ** (k - 1)][
                    2 ** (k - 1)] = position[2], position[2], position[2]
            elif i > 2 ** (k - 1) and j <= 2 ** (k - 1):  # 3
                grid = np.block([[grids[(k - 1, 2 ** (k - 1), 2 ** (k - 1))], grids[(k - 1, 2 ** (k - 1), 1)]],
                                 [grids[(k - 1, i - 2 ** (k - 1), j)], grids[(k - 1, 1, 1)]]])
                grid[2 ** (k - 1)-1][2 ** (k - 1)-1], grid[2 ** (k - 1)-1][2 ** (k - 1)], grid[2 ** (k - 1)][
                    2 ** (k - 1)] = position[3], position[3], position[3]
            elif i > 2 ** (k - 1) and j > 2 ** (k - 1):  # 4
                grid = np.block([[grids[(k - 1, 2 ** (k - 1), 2 ** (k - 1))], grids[(k - 1, 2 ** (k - 1), 1)]],
                                 [grids[(k - 1, 1, 2 ** (k - 1))], grids[(k - 1, i - 2 ** (k - 1), j - 2 ** (k - 1))]]])
                grid[2 ** (k - 1)-1][2 ** (k - 1)-1], grid[2 ** (k - 1)-1][2 ** (k - 1)], grid[2 ** (k - 1)][
                    2 ** (k - 1)-1] = position[4], position[4], position[4]

            grids[(k, i, j)] = grid

    return grids[(k, i, j)]


print(grid_cover_iterative(2, 1, 2))
