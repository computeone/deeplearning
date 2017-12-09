import sys
import os

import numpy as np
import math

from scipy.special import comb,perm

class EM:

    def __init__(self):
        print("init em")


    def simple_train_set(self):

        train = np.array(
            [[5,5],
            [9,1],
            [8,2],
            [4,6],
            [7,3]]
        )
        return train


    #m步来进行概率计算，二项式分布公式来进行计算
    def m_step(self,a,b,count):

        train_a = [0] * 5
        train_b = [0] * 5

        for i in range(5):
            pa = np.sum(count[i] * np.log((a,1 - a)))
            pb = np.sum(count[i] * np.log((b,1 - b)))

            pc = np.exp(pa) + np.exp(pb)
            pa = np.exp(pa) / pc
            pb = np.exp(pb) / pc

            [pa_count_head,pa_count_tail] = (pa,pa) * np.array(count[i])
            [pb_count_head,pb_count_tail] = (pb,pb) * np.array(count[i])

            train_a[i] = [pa_count_head,pa_count_tail]
            train_b[i] = [pb_count_head,pb_count_tail]

        return [train_a,train_b]


    #e步来进行重新计算概率值
    def e_step(self,train_a,train_b):

        count_head = 0
        count_tail = 0
        for i in train_a:
            count_head += i[0]
            count_tail += i[1]

        pa = count_head / (count_head + count_tail)

        count_head = 0
        count_tail = 0
        for i in train_b:
            count_head += i[0]
            count_tail += i[1]

        pb = count_head / (count_head + count_tail)

        return [pa,pb]


    def eval(self,pa,pb,train,step):

        for i in range(step):
            [train_a,train_b] = self.m_step(pa,pb,train)
            [pa,pb] = self.e_step(train_a,train_b)

            print((pa,pb))

        return [pa,pb]



if __name__ == '__main__':

    em = EM()

    train = em.simple_train_set()

    print(train)

    em.eval(0.6,0.5,train,10)







