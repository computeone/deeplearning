import sys
import os

import numpy as np
import math

import random
#simple dataset
# A1(2,10) A2(2,5) A3(8,4) A4(5,8) A5(7,5) A6(6,4) A7(1,2) A8(4,9)

class Kmeans:

    def __init__(self):
        print("init kmeans")

    #simple train dataset
    def simple_train_set(self):
        train = [[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]]
        return train

    #进行k个聚类，聚类中心点随机，选k个
    def eval(self,train,k,max_iteration):
        cluster = [0] * k

        #init cluster
        for i in range(k):
            rand =random.randint(0,len(train) - 1)
            cluster[i] = [train[rand][0],train[rand][1]]

        #iterator
        for i in range(max_iteration):

            for j in range(len(train)):


                #计算度量值，得到相应的类别
                min_distance = sys.maxsize
                index = -1
                for n in range(k):
                    distance = pow(cluster[n][0] - train[j][0],2) + pow(cluster[n][0] - train[j][0],2)
                    if(min_distance > distance):
                        min_distance = distance
                        index = n
                train[j] = [train[j][0],train[j][1],index]

            #重新计算cluster的中心
            weight = np.zeros((3,k))

            for m in range(len(train)):
                category = train[m][2]
                weight[category][0] += train[m][0]
                weight[category][1] += train[m][1]
                weight[category][2] += 1
            for n in range(k):
                cluster[n] = [weight[n][0] / weight[n][2],weight[n][1] / weight[n][2]]

        return train

if __name__ == '__main__':

    kmeans = Kmeans()
    train = kmeans.simple_train_set()
    category = kmeans.eval(train,3,10)
    print(category)