import sys
import os

import numpy as np
import math

class Pca:

    def __init__(self):
        print("init pca")

    def simple_train_set(self):
        train = np.array(
            [[2,2,3,5,3],
            [1,3,3,4,4]]
        )

        return train

    def means(self,train):

        for i in range(train.shape[0]):
            sum = 0
            for j in range(train.shape[1]):
                sum += train[i][j]

            average = sum / train.shape[1]
            for j in range(train.shape[1]):
                train[i][j] -= average

        return train


    def eval(self,train):

        #均值化
        train = self.means(train)

        #求解协方差矩阵
        cov  = np.cov(train)

        #求解特征值和特征向量
        eig = np.linalg.eig(cov)

        #降维
        result = np.matrix(eig[1][1]) * np.matrix(train)

        return result


if __name__ == '__main__':

    pca = Pca()
    train = pca.simple_train_set()
    print("train")
    print(train)
    result = pca.eval(train)
    print("result")
    print(result)
