#! /usr/bin/python
# -*- coding: utf-8 -*-


# str, list, dict
import math

PI = math.pi



def circleArea(r):
    area = PI * r ** 2
    return area


def noReturnValue():
    print('noReturnValue')


def complexFuncWithMoreParams(a, b, c, d):
    return 1, 2, 3


print(circleArea(2))
a = noReturnValue()
print(a)


def add(a,b):
    c=a+b
    return c
def sub(a,b):
    c=a-b
    return c


print(add(1,2),sub(1,2))

# 任意多个参数的函数

def add2(*args):
    s = 0
    for arg in args:
        s += arg
    return s
print(add2(1,2,3,4))



# 有默认参数的函数
def sayHi(words, name='xiaohong', age=18):
    pass


sayHi('你好呀，傻蛋', 'shadan')


# 多个参数和默认参数
def sayHi(words_1, words_2, *args, name='xiaohong',age=18):
    pass


sayHi('我们','在', '一起','吧',age=19,name='xiaogang')










