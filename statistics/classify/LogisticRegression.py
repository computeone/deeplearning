import sys
import os

import numpy as np
import math


class LogisticRegression:

    def __init__(self):
        print("init LogisticRegression")

    #第一行表示学生花费在学习上的时间，第二行为能否通过考试，研究学习时间和考试逻辑回归
    def simple_train_set(self):
        train = np.array(
            [[0,50,0],
            [0.75,0],
            [1.00,0],
            [1.25,0],
            [1,50,0],
            [1.75,0],
            [2.00,0],
            [2.25,1],
            [2.50,0],
            [2.75,1],
            [3.00,0],
            [3.25,1],
            [3.50,0],
            [4.00,1],
            [4.25,1],
            [4.50,1],
            [4.75,1],
            [5.00,1],
            [5.50,1]]
        )

        return train

    def caculate_coefficient(self,train):
        #默认用EM算法来估计他的参数，此处省略，直接得出参数

        #           coefficient  std.error   z-value    p-value
        # intercept -4.0777       1.7610    -2.316      0.0206
        #  hours    1.5046          0.6287  2.393       0.0167
        return [(-4.0777,1.5046),(1.7610,0.6287),(-2.316,2.393),(0.0206,0.0167)]

    def evel(self,train):

        [(coefficient,c_hours),(error,e_hours),(z,z_hours),(p,p_hours)] = self.caculate_coefficient(train)

        #线性输出

        for i in range(5):
            p = 1 / (1 + math.exp(-(z_hours * i + coefficient)))

            print("hours of study  probability of passing exam")
            print(i.__str__() + "               " + p.__str__())



if __name__ == '__main__':

    lr = LogisticRegression()

    train = lr.simple_train_set()

    print(train)

    lr.evel(train)




