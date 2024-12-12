def multiplication(m):
    m = int(input("输入一个数"))
    row = 1
    while row <= m:
        col = 1
        while col <= row:
                print("%dx%d=%d\t"%(row,col,row*col),end="") #生成金字塔型
                col += 1
        row += 1
        print("")    #换行
    return

multiplication("请输入一个数")



