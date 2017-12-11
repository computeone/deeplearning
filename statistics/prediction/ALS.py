import sys
import os

import numpy as np
import math

class ALS:

    def __init__(self):
        print("init als")

    #原始评分表，m * n的评分表
    def simple_train_set(self):
        train = np.array(
            [[3, 4, 5,2],
             [0, 1, 1,3],
             [0, 0, 1,2]]
        )

        return train

    #计算差错
    def evel_error(self,q,x,y,w):
        return np.sum((w * (q - np.dot(x,y))) ** 2)


    def eval(self, train, max_iteration):

        w = train.copy()

        for i in range(train.shape[0]):
            for j in range(train.shape[1]):

                if(train[i][j] > 0.5):
                    w[i][j] = 1
                else:
                    w[i][j] = 0

        m,n = train.shape
        lambda_  = 0.1
        n_factors = 2
        error = 0

        x = 3 * np.random.rand(m,n_factors)
        y = 3 * np.random.rand(n_factors,n)


        for i in range(max_iteration):

            x = np.linalg.solve(np.dot(y,y.T) + lambda_ * np.eye(n_factors),np.dot(y,train.T)).T

            y = np.linalg.solve(np.dot(x.T,x) + lambda_ * np.eye(n_factors),np.dot(x.T,train))

            error = self.evel_error(train,x,y,w)

        return x,y,error




if __name__ == '__main__':

    als = ALS()

    train = als.simple_train_set()

    print(train)

    x,y,error = als.eval(train,20)

    print("X矩阵:")
    print(x)

    print("Y矩阵:")
    print(y)

    print("差错值:")
    print(error)



