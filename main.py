# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 21:48:21 2025

@author: 王颖颖， 夏雨杨， 施丹砚
"""

import minimatrix as mm

def test_matrix_operations():
    # 定义矩阵并输出
    mat1 = mm.Matrix(dim=(2, 3), init_value=0)
    print(mat1)
    
    mat2 = mm.Matrix(data=[[0, 1], [1, 2], [2, 3]])
    print(mat2)
    
    # 矩阵乘法
    A = mm.Matrix(data=[[1, 2], [3, 4]])
    print(A.dot(A))
    
    # 矩阵转置
    A = mm.Matrix(data=[[1, 2], [3, 4]])
    print(A.T())
    
    B = mm.Matrix(data=[[1, 2, 3], [4, 5, 6]])
    print(B.T())
    
    # 矩阵指定元素求和
    A = mm.Matrix(data=[[1, 2, 3], [4, 5, 6]])

    print(A.matrix_sum(axis=0))
    print(A.matrix_sum(axis=1))
    #这组运行不下去
    
    # 切片
    x = mm.Matrix(data=[
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 0, 1]
        ])
    
    print(x[1, 2])
    print(x[0:2, 1:4])
    print(x[:, :2])
    
    # reshape变形函数
    m24 = mm.arange(0, 24, 1)
    print("Original m24:")
    print(m24)
    
    reshaped_m24_3x8 = m24.reshape((3, 8))
    print("Reshaped m24 (3x8):")
    print(reshaped_m24_3x8)

    reshaped_m24_24x1 = m24.reshape((24, 1))
    print("Reshaped m24 (24x1):")
    print(reshaped_m24_24x1)

    reshaped_m24_4x6 = m24.reshape((4, 6))
    print("Reshaped m24 (4x6):")
    print(reshaped_m24_4x6)
    
    # 赋值
    x[1, 2] = 0
    print(x)
    x[1:, 2:] = mm.Matrix(data=[[1, 2], [3, 4]])
    print(x)
    
    # 矩阵元素乘法
    print(mm.Matrix(data=[[1, 2]]) * mm.Matrix(data=[[3, 4]]))
    
    # zero相关函数
    zeros_matrix = mm.zeros([3, 3])
    print(zeros_matrix)

    A = mm.Matrix(data=[[1, 2, 3], [2, 3, 4]])
    zeros_like_A = mm.zeros_like(A)
    print(zeros_like_A)
    
    # one相关函数
    one_matrix = mm.ones([3, 3])
    print(one_matrix)

    B = mm.Matrix(data=[[1, 2, 3], [2, 3, 4]])
    ones_like_B = mm.ones_like(B)
    print(ones_like_B)
    #这里的报错是Index must be a 2-tuple
    
    # 矩阵拼接
    A, B = mm.Matrix([[0, 1, 2]]), mm.Matrix([[3, 4, 5]])
    print(A.concatenate(B))
    A.concatenate(B, A, axis=1)
    #报错是 if len(A.data[1]) != len(self.data[1]):IndexError: list index out of range
    
    # 测试 nrandom() 和 nrandom_like()
    nrandom_matrix = mm.nrandom([3, 3])
    print("Random matrix (3x3):")
    print(nrandom_matrix)
    #得是维数以内的随机数, 这个也不对,没有参数, 而且输出结果是list
    

    nrandom_like_m24 = mm.nrandom_like(m24)
    print("Random_like(m24):")
    print(nrandom_like_m24)
    

import random

def test_least_squares():
    m, n = 2, 2

    X = mm.nrandom((m, n))
    w = mm.Matrix(data=[[random.random() for _ in range(n)] for _ in range(m)])
    e = mm.Matrix(data=[[random.gauss(0, 1) for _ in range(m)] for _ in range(n)])

    Y = X.dot(w).__add__(e)

    X_transpose = X.T()
    X_transpose_X = X_transpose.dot(X)
    X_transpose_X_inv = X_transpose_X.inverse()
    w1 = X_transpose_X_inv.dot(X_transpose).dot(Y)

    print("Original w:")
    print(w)
    print("Estimated w1:")
    print(w1)

if __name__ == "__main__":
    test_matrix_operations()
    test_least_squares()
