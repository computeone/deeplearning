import sys
import os

import numpy as np
import math

class LinearRegression:

    def __init__(self):
        print("init LinearRegression")


    def simple_train_set(self):
        train = np.array(
            [[1,1],
            [2,3],
            [4,3],
            [3,2],
            [5,5]]
        )

        return train

    def evel_coefficient(self,train):

        sum = [0,0]
        for item in train:
            sum  += item

        average = sum / train.shape[0]


        sum = [0,0]
        stdev = [0,0]

        for item in train:

            sum[0] += math.pow(item[0] - average[0],2)
            sum[1] += math.pow(item[1] - average[1],2)

        stdev[0] = math.sqrt(sum[0] / (train.shape[0] - 1))
        stdev[1] = math.sqrt(sum[1] / (train.shape[0] - 1))

        correlation = 0

        for item in train:
            correlation += (item[0] - average[0]) * (item[1] - average[1])

        correlation = (correlation / (train.shape[0] - 1)) / (stdev[0] * stdev[1])

        b1 = correlation * stdev[1] / stdev[0]

        b0 = average[1] - b1 * average[0]


        predict = [0] * train.shape[0]

        for i in range(train.shape[0]):
            predict[i] = b0 + b1 * train[i][0]

        rmse = np.sum(np.power(np.array(predict) - train.transpose()[1],2)) / train.shape[0]

        rmse = math.sqrt(rmse)

        return [b0,b1,rmse]



if __name__ == '__main__':

    lr = LinearRegression()

    train = lr.simple_train_set()

    print(train)

    [b0,b1,rmse] = lr.evel_coefficient(train)

    print("系数:")
    print(b0)
    print(b1)

    print("误差:")
    print(rmse)