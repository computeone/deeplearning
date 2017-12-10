import sys
import os

import numpy as np
import math

class NaiveBayes:

    def __init__(self):
        print("init naivebayes")

    # simple train set 第一行类别分别为，age，income,student,credit_rating,buys_computer
    # age:0-youth 1-middle_aged, 2 - senior,income: 0-low,1-medium,2-high credit_rating:0-fair,1-excellent
    # buyes_compute:0-no,1-yes
    def simple_train_set(self):
        train = np.array(
                          [[0, 2, 0, 0, 0],
                          [0, 2, 0, 1, 0],
                          [1, 2, 0, 0, 1],
                          [2, 1, 0, 0, 1],
                          [2, 0, 1, 0, 1],
                          [2, 0, 1, 1, 0],
                          [1, 0, 1, 1, 1],
                          [0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 1],
                          [2, 1, 1, 0, 1],
                          [0, 1, 1, 1, 1],
                          [1, 1, 0, 1, 1],
                          [1, 2, 1, 0, 1],
                          [2, 1, 0, 1, 0]])

        return train

    def evel(self,train):

        #计算类别的出现次数
        y = {}
        size = train.shape[1]
        for item in train:

            if(item[size - 1] not in y):
                y[item[size - 1]] = 1
            else:
                y[item[size - 1]] += 1

        #计算X各个分量的次数
        x = {}
        for item in train:

            for i in range(size - 1):
                tmp = [-1] * size
                tmp[i] = item[i]
                tmp[size - 1] = item[size - 1]

                if(tuple(tmp) not in x):
                    x[tuple(tmp)] = 1
                else:
                    x[tuple(tmp)] += 1

        #计算p(X|yi)概率
        x_y = {}
        for xi in x:
            x_y[xi] = x[xi] / y[xi[size - 1]]

        #计算p(yi)概率
        py = {}
        for yi in y:
            py[yi] = y[yi] / train.shape[0]

        return [x_y,py]



if __name__ == '__main__':

    nb = NaiveBayes()

    train = nb.simple_train_set()

    print(train)

    [x_y,y] = nb.evel(train)

    test = [0,1,1,0]

    y0 = x_y[0,-1,-1,-1,0] * x_y[-1,1,-1,-1,0] * x_y[-1,-1,1,-1,0] * x_y[-1,-1,-1,0,0]
    y0 = y0 * y[0]

    y1 = x_y[0,-1,-1,-1,1] * x_y[-1,1,-1,-1,1] * x_y[-1,-1,1,-1,1] * x_y[-1,-1,-1,0,1]
    y1 = y1 * y[1]



    print("分类结果:")
    print(y1)
    print(y0)
    if y1 > y0:
        print("yes")
    else:
        print("no")

