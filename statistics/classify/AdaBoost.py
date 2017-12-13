import sys
import os

import numpy as np
import math

class AdaBoost:

    def __init__(self):
        print("init adaboost")

    def simple_train_set(self):
        train = np.array(
            [(1,1,1),
            (1,2,1),
            (1.5,0.5,-1),
            (1.8,2.5,-1),
            (1.9,4,1),
            (2,3,-1),
            (2.1,4,1),
            (3.5,3.5,1),
            (5.5,1.5,-1),
            (5.6,4.5,-1)]
        )

        return train

    def classify_x(self,train,min_value,max_value):

        while(True):

            rand = np.random.randint(min_value,max_value) * np.random.rand()

            m,n = train.shape
            a = [0,0]
            b = [0,0]
            for i in range(m):

                if(train[i][0] == rand):
                    break

                if(train[i][0] < rand):
                    if(train[i][2] == 1):
                        a[0] += 1
                    else:
                        a[1] += 1
                else:
                    if(train[i][2] == 1):
                        b[0] += 1
                    else:
                        b[1] += 1

            error1 = 0
            error2 = 0

            error1 = (a[0] + b[1]) / m
            error2 = (a[1] + b[0]) / m

            if(min(error1,error2) > 0.1 and min(error1,error2) < 0.4):

                if(error1 == min(error1,error2)):
                    return (rand,-1)
                else:
                    return (rand,1)


    def classify_y(self,train,min_value,max_value):

        while (True):

            rand = np.random.randint(min_value, max_value) * np.random.rand()

            m,n = train.shape
            a = [0, 0]
            b = [0, 0]
            for i in range(m):

                if(train[i][1] == rand):
                    break

                if (train[i][1] < rand):
                    if (train[i][2] == 1):
                        a[0] += 1
                    else:
                        a[1] += 1
                else:
                    if (train[i][2] == 1):
                        b[0] += 1
                    else:
                        b[1] += 1

            error1 = 0
            error2 = 0

            error1 = (a[0] + b[1]) / m
            error2 = (a[1] + b[0]) / m

            if (min(error1, error2) > 0.1 and min(error1, error2) < 0.4):

                if (error1 == min(error1, error2)):
                    return (rand, -1)
                else:
                    return (rand, 1)


    def eval(self,train):

        c = [0] * 3

        c[0] = (0,self.classify_x(train,0,8))
        c[1] = (0,self.classify_x(train,0,8))
        c[2] = (1,self.classify_y(train,0,5))

        c[0] = (c[0][0],c[0][1][1],c[0][1][0])
        c[1] = (c[1][0],c[1][1][1],c[1][1][0])
        c[2] = (c[2][0],c[2][1][1],c[2][1][0])

        m,n = train.shape
        w = [0] * m
        for i in range(m):
            w[i] = 1 / m


        alpha = [0] * len(c)
        for i in range(3):

            error = []
            for j in range(m):

                #X垂线分类器
                if(c[i][0] == 0):

                    #左边为+1
                    if(c[i][1] == 1):
                        if(train[j][0] > c[i][2]):
                            if(train[j][2] == 1):
                                error.append(j)
                        else:
                            if(train[j][2] == -1):
                                error.append(j)

                    else:
                        if(train[j][0] > c[i][2]):
                            if(train[j][2] == -1):
                                error.append(j)
                        else:
                            if(train[j][2] == 1):
                                error.append(j)
                #Y水平分类器
                else:

                    #下边为+1
                    if(c[i][1] == 1):
                        if(train[j][1] > c[i][2]):
                            if(train[j][2] == 1):
                                error.append(j)
                        else:
                            if(train[j][2] == -1):
                                error.append(j)

                    else:
                        if(train[j][1] > c[i][2]):
                            if(train[j][2] == -1):
                                error.append(j)
                        else:
                            if(train[j][2] == 1):
                                error.append(j)


            #计算误差
            err = 0
            for j in range(len(error)):
                err += w[j]

            #计算alpha值
            alpha[i] =  math.log((1 - err) / err,math.e) / 2

            #更新权值
            for j in range(len(w)):
                if(j not in error):
                    w[j] = w[j] * math.exp(alpha[i])
                else:
                    w[j] = w[j] * math.exp(-alpha[i])

        #adaboost分类函数


        return [alpha,c]


if __name__ == '__main__':

    adaboost = AdaBoost()

    train = adaboost.simple_train_set()

    print(train)

    [alpha,c] =adaboost.eval(train)

    print([alpha,c])




