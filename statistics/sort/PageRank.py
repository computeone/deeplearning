import sys
import os

import numpy as np
import math


#pagerank simple algorithnm
#       A   B   C   D
#   A   0   1   0   0
#   B   1   0   0   1
#   C   1   0   1   1
#   D   1   0   0   0
class PageRank:


    def __init__(self):
        print("init pagerank")

#初始化简单PR数据值,得到迁移概率矩阵

    def init_simple_example(self):
        migrate  = np.matrix([[0,1/2,0,0],[1/3,0,0,1/2],[1/3,0,1,1/2],[1/3,1/2,0,0]])
        return migrate

#进行PR值的迭代计算

    def eval(self,migrate,d,max_iteraition):
        n = migrate.shape[0]
        bias = np.matrix([(1-d)/n] * n).transpose()
        rank = np.matrix([1] * n).transpose()

        for i in range(max_iteraition):
            rank = bias + d * migrate * rank;

        return rank



pr = PageRank()
rank = pr.init_simple_example()
rank = pr.eval(rank,0.85,20)
print("rank:")
print(rank)
