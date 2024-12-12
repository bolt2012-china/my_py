def banker(x):
    y = float(input("请输入一个需要处理的数（三位小数）"))
    x = 1000*y
    a = x//10
    b = x-10*a
    if b<5:
        x=x-b
        y=x/1000
        print(y)
    elif b>5:
        x=(a+1)*10
        y = x / 1000
        print(y)
    elif b==5:
        if a%2==0:
            x=x-b
            y = x / 1000
            print(y)
        elif a%2==1:
            x = (a + 1) * 10
            y = x / 1000
            print(y)
banker("请输入一个需要处理的数（三位数）")


