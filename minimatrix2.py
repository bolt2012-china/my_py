# Framework for IEEE course final project
# Fan Cheng, 2022
import copy
import random

'''
矩阵类核心功能（由一个人负责）
函数：__init__, shape, reshape, dot, T, sum, copy, Kronecker_product, det, inverse, rank
这些函数是矩阵类的核心功能，实现矩阵的基本操作（比如矩阵乘法、转置、求和、逆矩阵、秩等）。一个人可以负责矩阵类的设计和实现，确保矩阵的基本运算正确无误。
'''
class Matrix:
    r"""
    自定义的二维矩阵类

    Args:
        data: 一个二维的嵌套列表，表示矩阵的数据。即 data[i][j] 表示矩阵第 i+1 行第 j+1 列处的元素。
              当参数 data 不为 None 时，应根据参数 data 确定矩阵的形状。默认值: None
        dim: 一个元组 (n, m) 表示矩阵是 n 行 m 列, 当参数 data 为 None 时，根据该参数确定矩阵的形状；
             当参数 data 不为 None 时，忽略该参数。如果 data 和 dim 同时为 None, 应抛出异常。默认值: None
        init_value: 当提供的 data 参数为 None 时，使用该 init_value 初始化一个 n 行 m 列的矩阵，
                    即矩阵各元素均为 init_value. 当参数 data 不为 None 时，忽略该参数。 默认值: 0

    Attributes:
        dim: 一个元组 (n, m) 表示矩阵的形状
        data: 一个二维的嵌套列表，表示矩阵的数据

    Examples:
        >>> mat1 = Matrix(dim=(2, 3), init_value=0)
        >>> print(mat1)
        >>> [[0 0 0]
             [0 0 0]]
        >>> mat2 = Matrix(data=[[0, 1], [1, 2], [2, 3]])
        >>> print(mat2)
        >>> [[0 1]
             [1 2]
             [2 3]]
    """

    def __init__(self, data=None, dim=None, init_value=None):
        if data is None and dim is None:
            raise ValueError("Both data and dim cannot be None at the same time.")
        if data is None:
            self.dim = dim
            self.data = [[init_value for _ in range(dim[1])] for _ in range(dim[0])]
        else:
            if isinstance(data[0], list):
                self.dim = (len(data), len(data[0]))
            else:
                self.dim = (1, len(data))
            self.data = data

    

    def shape(self):
        return self.dim

    def reshape(self, newdim):
        r"""
        将矩阵从(m,n)维拉伸为newdim=(m1,n1)
        该函数不改变 self

        Args:
            newdim: 一个元组 (m1, n1) 表示拉伸后的矩阵形状。如果 m1 * n1 不等于 self.dim[0] * self.dim[1],
                    应抛出异常

        Returns:
            Matrix: 一个 Matrix 类型的返回结果, 表示 reshape 得到的结果
        """
        m,n = self.dim
        m1,n1 = newdim
        assert m*n == m1*n1, "Cannot reshape matrix with different dimensions."     # 其实不应该用断言，只是我想用
        remat = []                                                                  # 应该用raise ValueError更好
        mat = []
        for i in range(m):
            for j in range(n):
                remat.append(self.data[i][j])
        for i in range(m1):
            row = []
            mat.append(row)
            for j in range(n1):
                row.append(remat[i*n1+j])
        return Matrix(data=mat)


    def dot(self, other):
        r"""
        矩阵乘法：矩阵乘以矩阵
        按照公式 A[i, j] = \sum_k B[i, k] * C[k, j] 计算 A = B.dot(C)

        Args:
            other: 参与运算的另一个 Matrix 实例

        Returns:
            Matrix: 计算结果

        Examples:
            >>> A = Matrix(data=[[1, 2], [3, 4]])
            >>> A.dot(A)
            >>> [[ 7 10]
                 [15 22]]
        """
        dim1 = self.dim
        dim2 = other.dim
        assert dim1[1] == dim2[0], "Cannot dot matrix with different dimensions."
        result = [[0 for _ in range(dim1[0])] for _ in range(dim2[1])]
        i = 0
        while i < dim1[0]:
            for j in range(dim2[1]):
                k = 0
                while k < dim1[1]:
                    result[i][j] += self.data[i][k] * other.data[k][j]
                    k += 1
            i += 1

        return Matrix(data=result)

    def T(self):
        r"""
        矩阵的转置

        Returns:
            Matrix: 矩阵的转置

        Examples:
            >>> A = Matrix(data=[[1, 2], [3, 4]])
            >>> A.T()
            >>> [[1 3]
                 [2 4]]
            >>> B = Matrix(data=[[1, 2, 3], [4, 5, 6]])
            >>> B.T()
            >>> [[1 4]
                 [2 5]
                 [3 6]]
        """
        trans = []
        for i in range(self.dim[1]):
            row = []
            trans.append(row)
            for j in range(self.dim[0]):
                row.append(self.data[j][i])
        return Matrix(data=trans)


    def mul(self,num):
        comat = self.data.copy()
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                comat[i][j] = comat[i][j] * num
        return Matrix(data=comat)



    def sum(self, axis=None):
        r"""
        根据指定的坐标轴对矩阵元素进行求和

        Args:
            axis: 一个整数，或者 None. 默认值: None
                  axis = 0 表示对矩阵进行按列求和，得到形状为 (1, self.dim[1]) 的矩阵
                  axis = 1 表示对矩阵进行按行求和，得到形状为 (self.dim[0], 1) 的矩阵
                  axis = None 表示对矩阵全部元素进行求和，得到形状为 (1, 1) 的矩阵

        Returns:
            Matrix: 一个 Matrix 类的实例，表示求和结果

        Examples:
            >>> A = Matrix(data=[[1, 2, 3], [4, 5, 6]])
            >>> A.sum()
            >>> [[21]]
            >>> A.sum(axis=0)
            >>> [[5 7 9]]
            >>> A.sum(axis=1)
            >>> [[6]
                 [15]]
        """
        if axis == None:
            sum = 0
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    sum += self.data[i][j]

        if axis == 0:
            sum = [0]*self.dim[1]
            for j in range(self.dim[1]):
                for i in range(self.dim[0]):
                    sum[j] += self.data[i][j]

        elif axis == 1:
            sum = [0]*self.dim[0]
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    sum[i] += self.data[i][j]
#axis ==1和0都跑不动
#报错信息是   if isinstance(data[0], list):TypeError: 'int' object is not subscriptable

        return Matrix(data=sum)


    def copy(self):
        r"""
        返回matrix的一个备份

        Returns:
            Matrix: 一个self的备份
        """
        cmat = copy.deepcopy(self.data)
        return Matrix(data=cmat)


    def concatenate(self, *items, axis=0):
        r"""
        将若干矩阵按照指定的方向拼接起来
        若给定的输入在形状上不对应，应抛出异常
        该函数应当不改变 items 中的元素

        Args:
            items: 一个可迭代的对象，其中的元素为 Matrix 类型。
            axis: 一个取值为 0 或 1 的整数，表示拼接方向，默认值 0.
                  0 表示在第0维即行上进行拼接
                  1 表示在第1维即列上进行拼接

        Returns:
            Matrix: 一个 Matrix 类型的拼接结果

        Examples:
            >>> A, B = Matrix([[0, 1, 2]]), Matrix([[3, 4, 5]])
            >>> concatenate((A, B))
            >>> [[0 1 2]
                 [3 4 5]]
            >>> concatenate((A, B, A), axis=1)
            >>> [[0 1 2 3 4 5 0 1 2]]
        """
        while axis == 0:
            for A in items:
                if len(A.data[0]) != len(self.data[0]):
                    raise ValueError("The number of columns must be the same for vertical concatenation.")
                else:
                    result = self.data + A.data
                    return Matrix(data=result)
        while axis == 1:
            for A in items:
                if len(A.data[1]) != len(self.data[1]):
                    raise ValueError("The number of rows must be the same for horizontal concatenation.")
                else:
                    result = []
                    for i in range(self.dim[0]):
                        result.append(self.data[i] + A.data[i])
                    return Matrix(data=result)



    def Kronecker_product(self, other):
        r"""
        计算两个矩阵的Kronecker积，具体定义可以搜索，https://baike.baidu.com/item/克罗内克积/6282573

        Args:
            other: 参与运算的另一个 Matrix

        Returns:
            Matrix: Kronecke product 的计算结果
        """
        if not isinstance(other, Matrix):
            raise TypeError("The argument must be an instance of Matrix")
        row_a, col_a = self.dim
        row_b, col_b = other.dim
        result_data = [[0 for _ in range(col_a * col_b)] for _ in range(row_a * row_b)]

        for i in range(row_a):
            for j in range(col_a):
                for k in range(row_b):
                    for l in range(col_b):
                        result_data[i * row_b + k][j * col_b + l] += self.data[i][j] * other.data[k][l]

        return Matrix(result_data)


    def __str__(self):
        """将矩阵格式化为带换行的字符串，按照指定格式输出"""
        # 使用列表推导式格式化矩阵每一行，并对齐元素
        rows = ["[" + " ".join(f"{elem:3}" for elem in row) + "]" for row in self.data]
        return "[" + "\n ".join(rows) + "]"


    def _get_slice_indices(self, indices, dim_size):
        """辅助函数获得索引切片"""
        if isinstance(indices, slice):
            return indices.indices(dim_size)
        return indices, indices + 1, 1
    
    
    def __getitem__(self, key):
        r"""
        实现 Matrix 的索引功能，即 Matrix 实例可以通过 [] 获取矩阵中的元素（或子矩阵）

        x[key] 具备以下基本特性：
        1. 单值索引
            x[a, b] 返回 Matrix 实例 x 的第 a 行, 第 b 列处的元素 (从 0 开始编号)
        2. 矩阵切片
            x[a:b, c:d] 返回 Matrix 实例 x 的一个由 第 a, a+1, ..., b-1 行, 第 c, c+1, ..., d-1 列元素构成的子矩阵
            特别地, 需要支持省略切片左(右)端点参数的写法, 如 x 是一个 n 行 m 列矩阵, 那么
            x[:b, c:] 的语义等价于 x[0:b, c:m]
            x[:, :] 的语义等价于 x[0:n, 0:m]

        Args:
            key: 一个元组，表示索引

        Returns:
            索引结果，单个元素或者矩阵切片
        """

        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("Index must be a 2-tuple")

        row_indices, col_indices = key

        # 获取行和列的切片索引
        start_row, stop_row, step_row = self._get_slice_indices(row_indices, self.dim[0])
        start_col, stop_col, step_col = self._get_slice_indices(col_indices, self.dim[1])

        # 检查索引是否有效
        if start_row >= stop_row or start_col >= stop_col:
            raise IndexError("Invalid slice indices")
        if start_row < 0 or start_col < 0 or stop_row > self.dim[0] or stop_col > self.dim[1]:
            raise IndexError("Slice indices out of range")

        # 返回单个元素或切片矩阵
        if isinstance(row_indices, int) and isinstance(col_indices, int):
            return self.data[start_row][start_col]
        else:
            sliced_data = [row[start_col:stop_col] for row in self.data[start_row:stop_row]]
            return Matrix(data=sliced_data)
        

    def __setitem__(self, key, value):
        """实现 Matrix 的赋值功能, 通过 x[key] = value 进行赋值的功能"""
        # 合并的条件判断，检查 key 是否是 2-tuple 类型
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("Index must be a 2-tuple")

        row_indices, col_indices = key

        # 处理单元素赋值的情况
        if isinstance(row_indices, int) and isinstance(col_indices, int):
            self.data[row_indices][col_indices] = value

        # 处理切片赋值的情况
        else:
            # 如果是切片，检查 value 是否为一个 Matrix 实例，并且其维度和切片的维度匹配
            if not isinstance(value, Matrix):
                raise ValueError("Assigned value must be a Matrix instance")

            # 获取切片范围
            start_row, stop_row, _ = self._get_slice_indices(row_indices, self.dim[0])
            start_col, stop_col, _ = self._get_slice_indices(col_indices, self.dim[1])

            # 检查赋值矩阵的形状是否匹配
            if value.dim[0] != stop_row - start_row or value.dim[1] != stop_col - start_col:
                raise ValueError(f"Shape of value {value.dim} does not match slice shape {(stop_row - start_row, stop_col - start_col)}")

            # 执行赋值
            for i in range(start_row, stop_row):
                for j in range(start_col, stop_col):
                    self.data[i][j] = value.data[i - start_row][j - start_col]

    
    def __pow__(self, n):
        """实现矩阵的 n 次幂，n 为自然数，使用快速幂算法进行优化."""
        if not isinstance(n, int) or n < 0:
            raise ValueError("Exponent must be a non-negative integer.")
    
        if self.dim[0] != self.dim[1]:
            raise ValueError("Matrix must be square to compute power.")
    
        # 如果 n 为 0，返回单位矩阵
        if n == 0:
            return Matrix(data=[[1 if i == j else 0 for j in range(self.dim[1])] for i in range(self.dim[0])])
    
        # 如果 n 为 1，直接返回矩阵本身
        if n == 1:
            return self
    
        # 快速幂算法
        def matrix_pow(A, n):
            result = Matrix(data=[[1 if i == j else 0 for j in range(A.dim[1])] for i in range(A.dim[0])])  # 单位矩阵
            base = A
        
            while n > 0:
                if n % 2 == 1:  # 如果 n 是奇数
                    result = result.dot(base)
                base = base.dot(base)  # base = base^2
                n //= 2  # 递归处理，n 除以 2
        
            return result
    
        return matrix_pow(self, n)

    def __add__(self, other):
        """两个矩阵相加"""
        # 检查两个矩阵的维度是否相同
        if self.dim != other.dim:
            raise ValueError("Matrices must have the same dimensions to be added.")
    
        # 逐元素相加，创建新矩阵
        result_data = [
            [self.data[i][j] + other.data[i][j] for j in range(self.dim[1])]
            for i in range(self.dim[0])
            ]
    
        return Matrix(data=result_data)
    
    def __sub__(self, other):
        """两个矩阵相减"""
        # 检查两个矩阵的维度是否相同
        if self.dim != other.dim:
            raise ValueError("Matrices must have the same dimensions to be subtracted.")
    
        # 逐元素相减，创建新矩阵
        result_data = [
            [self.data[i][j] - other.data[i][j] for j in range(self.dim[1])]
            for i in range(self.dim[0])
            ]
    
        return Matrix(data=result_data)
    
    def __mul__(self, other):
        """两个矩阵对应位置元素相乘"""
        # 检查两个矩阵的维度是否相同
        if self.dim != other.dim:
            raise ValueError("Matrices must have the same dimensions to perform element-wise multiplication.")
    
        # 逐元素相乘，创建新矩阵
        result_data = [
            [self.data[i][j] * other.data[i][j] for j in range(self.dim[1])]
            for i in range(self.dim[0])
            ]
    
        return Matrix(data=result_data)
    
    def __len__(self):
        '''返回矩阵元素的数目'''
        return self.dim[0] * self.dim[1]

    
    def det(self):
        """计算方阵的行列式，使用高斯消元法"""
        if self.dim[0] != self.dim[1]:
            raise ValueError("Matrix must be square to compute determinant.")

        n = self.dim[0]
        mat = [row[:] for row in self.data]  # 深拷贝矩阵数据，避免修改原矩阵
        determinant = 1

        for i in range(n):
            # 1. 选取主元并交换行
            pivot_row = self._select_pivot_row(mat, i)
            if mat[pivot_row][i] == 0:
                return 0  # 如果主元为 0，则行列式为 0

            # 如果主元行不是当前行，交换行并调整行列式符号
            if pivot_row != i:
                self._swap_rows(mat, i, pivot_row)
                determinant *= -1

            # 2. 消元过程，将当前列下方的所有元素变为 0
            self._eliminate(mat, i)

            # 累乘对角线元素
            determinant *= mat[i][i]

        return determinant


    def _select_pivot_row(self, mat, col):
        """辅助函数，选择主元行，返回绝对值最大的元素所在的行"""
        pivot_row = max(range(col, len(mat)), key=lambda x: abs(mat[x][col]))
        return pivot_row


    def _swap_rows(self, mat, row1, row2):
        """辅助函数，交换矩阵的两行"""
        mat[row1], mat[row2] = mat[row2], mat[row1]


    def _eliminate(self, mat, col):
        """辅助函数，对矩阵进行消元操作，将当前列下方的元素消去"""
        n = len(mat)
        for row in range(col + 1, n):
            factor = mat[row][col] / mat[col][col]
            for j in range(col, n):
                mat[row][j] -= factor * mat[col][j]

    def _eliminate_column(self, mat, identity, col):
        """辅助函数，消去指定列中除主元外的所有元素。"""
        n = len(mat)
        for i in range(n):
            if i != col:
                factor = mat[i][col]
                mat[i] = [mat_i - factor * mat_col for mat_i, mat_col in zip(mat[i], mat[col])]
                identity[i] = [id_i - factor * id_col for id_i, id_col in zip(identity[i], identity[col])]
            
    
    def _normalize_row(self, mat, identity, row):
        """辅助函数将指定行归一化，使得主元为 1。"""
        pivot = mat[row][row]
        mat[row] = [x / pivot for x in mat[row]]
        identity[row] = [x / pivot for x in identity[row]]

    def inverse(self):
        """计算非奇异方阵的逆矩阵。"""
        if self.dim[0] != self.dim[1]:
            raise ValueError("Matrix must be square to compute the inverse.")

        n = self.dim[0]
        mat = [row[:] for row in self.data]  # 深拷贝矩阵数据
        identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]  # 单位矩阵

        for i in range(n):
            # 1. 主元选择并交换行
            pivot_row = self._select_pivot_row(mat, i)
            if mat[pivot_row][i] == 0:
                raise ValueError("Matrix is singular and cannot be inverted.")
            if pivot_row != i:
                self._swap_rows(mat, i, pivot_row)
                self._swap_rows(identity, i, pivot_row)

            # 2. 归一化主元行
            self._normalize_row(mat, identity, i)

            # 3. 消去其他行
            self._eliminate_column(mat, identity, i)

        return Matrix(data=identity)
    
    def rank(self):
        m, n = self.dim
        r = 0
        for i in range(m - 1):
            pivot = -1
            for j in range(i + 1, m):
                if self.data[j][i] != 0:
                    pivot = j
                    break  # 找到第一个非零元
            if pivot == -1:
                continue
            self.data[i], self.data[pivot] = self.data[pivot], self.data[i]

            for k in range(i + 1, m):
                scale = self.data[k][i] / self.data[i][i]
                for l in range(i, n):
                    self.data[k][l] -= scale * self.data[k][l]

            if any(self.data[i]):
                r += 1

        return r
    
def I(n):
    '''return an n*n unit matrix'''
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1
    return matrix
    
def narray(dim, init_value=1):  # dim (,,,,,), init为矩阵元素初始值
    matrix = [[init_value for _ in range(dim[0])] for _ in range(dim[1])]
    return matrix
    # return Matrix(dim, None, init_value)

def arange(start, end, step):
    return Matrix([[x for x in range(start, end, step)]])
    
def zeros(dim):
    matrix = [[0 for _ in range(dim[0])] for _ in range(dim[1])]
    return matrix
    
def zeros_like(matrix):
    raw = len(matrix)
    col = len(matrix[0])
    return [[0 for _ in range(col)] for _ in range(raw)]
    
def ones(dim):
    matrix = [[1 for _ in range(dim[0])] for _ in range(dim[1])]
    return matrix
    
def ones_like(matrix):
    r"""
    返回一个维数和matrix一样 的全1 narray
    类同 zeros_like
    """
    raw = len(matrix)
    col = len(matrix[0])
    return [[1 for _ in range(col)] for _ in range(raw)]

def nrandom(dim):
    r"""
    返回一个维数为dim 的随机 narray
    参数与返回值类型同 zeros
    """
    init_value = random.randint(0, 10) #这里报错
    return narray(dim, init_value)

def nrandom_like(matrix):
    """返回一个维数和 matrix 一样的随机 narray"""
    rows, cols = matrix.dim
    random_data = [[random.random() for _ in range(cols)] for _ in range(rows)]
    return Matrix(data=random_data)

def vectorize(func):
    """将给定函数进行向量化"""
    def vectorized_function(matrix):
        if not isinstance(matrix, Matrix):
            raise TypeError("Input must be an instance of Matrix.")
        # 对矩阵中的每个元素应用函数 func
        result_data = [[func(value) for value in row] for row in matrix.data]
        return Matrix(data=result_data)

    return vectorized_function


if __name__ == "__main__":
    print("test here")

  

