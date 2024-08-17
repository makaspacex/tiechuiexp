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

"""
1x1=1
2x1=2 2x2=4
3x1=3 3x2=6 3x3=9
...............................
"""

#
"""
   *
  ***
 *****
******* # 个数是n
 *****
  ***
   *

n=20
n=30
"""
# n = 3
# m = (n - 1) / 2
# while m >= 0:
#     i = 1
#     while i <= m:
#         print(' ', end='')
#         i = i + 1
#     while i > m:
#         b = n - m - 1
#         a = 1
#         while a <= b:
#             print("*", end='')
#             a = a + 1
#         break
#     print()
#     m = m - 1

def exe_xing():
    n=7
    row=1
    half_n=int((n-1)/2)
    while row<half_n+2:
        space_n=half_n-row+1
        for i in range(space_n):
            print(" ",end="")
        for i in range(2*row-1):
            print("*",end="")
        print()
        row=row+1
    while row<=n:
        space_n = row-half_n - 1
        for i in range(space_n):
            print(" ", end="")
        for i in range(n-2*space_n):
            print("*", end="")
        print()
        row = row + 1

exe_xing()

def exe_xing_v2():
    n = 13
    m = 1
    a = int((n - 1) / 2)

    while m < a + 2:
        print("%s%s" % (' ' * (a - m + 1), "*" * (2 * m - 1)))
        m = m + 1
    while m <= n:
        print("%s%s" % (' ' * (m - a - 1), "*" * (n - 2 * (m - a - 1))))
        m = m + 1

exe_xing_v2()

"""
   *
  ***
 *****
******* # 个数是n
   *
   *
   *
   *

n=15
n=22
n=...
"""
