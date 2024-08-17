#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# Copyright 2017 izhangxm@gmail.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# class_mates = ['xiaoming', 12, 12.4]
# class_mates.append('xiaohong')
# print(class_mates)
#
# print(class_mates[0])
# print(class_mates[-1])
# print(class_mates[-2])
#
# print(class_mates[1])
# class_mates.append('xiaohong')
#
# print(class_mates)
# class_mates.pop()
# print(class_mates)
#
# print(len(class_mates))
# print(len("123"))
# # print(len(12)) XXXX
#
#
# arr_2d_ = ["d", 2, ['xiaohong','xiaoming'] ]
# print(arr_2d_[2])
# print(arr_2d_[2][0])
#
# if 2 > 3 or '1' < '3':
#     print('ssss')
#
#
# if not 1 > 3:
#     print('1111')
#
# a = 'a'
# b = 'b'
#
# if a > b:
#     print('a>b')
# else:
#     print('else')
#
#
# if a > b:
#     print('a>b')
#     print('ss')
#     print('ss')
# elif a == b :
#     print('ab')
#     if 2>1:
#         print('2>1')
#
# elif a<b:
#     print('a<b')


names = ['Michael', 'Bob', 'Tracy']

for ele in names:
    print(ele)


n = 10
while n>0:
    n -= 1
    if n >= 7:
        continue


    if n > 5:
        break

    print(n)

    for ele in names:
        print(ele)
        break

import time
from datetime import datetime

while True:
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    time.sleep(1)



















































