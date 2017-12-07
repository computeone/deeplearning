import sys
import os

import numpy as np
import math

#KNN algorithnm example
# dataset
#7          7       bad
#7          4       bad
#3          4       good
#1          4       good
class KNN:

    def __init__(self):
        print("init knn")

#生成默认数据，这是简单的测试数据
    def get_simple_train_data(self):
        data = [[7,7,0],[7,4,0],[3,4,1],[1,4,1]]
        return data

#计算度量值，默认计算欧式距离
    def metrics(self,datas,target):

        distances = [None] * len(datas)
        for index in range(len(datas)):
            distances[index] = [math.pow(datas[index][0] - target[0],2) +
                                math.pow(datas[index][1] - target[1],2),datas[index][2]]
        return distances

#按照度量值，来进行排序，默认是按照数值升序排序
    def sort(self,distancs):
        distances_sorted = sorted(distancs,key= lambda s:s[0])
        return distances_sorted

#收集前k个值，得到最终结果
    def gather(self,distances_sorted,k):
        categorys = {}

        for index in range(k):
            if(distances_sorted[index][1] in categorys.keys()):
                categorys[distances_sorted[index][1]] += 1
            else:
                categorys[distances_sorted[index][1]] = 1


        sum = [None,-1]

        for key in categorys.keys():
            if(sum[1] < categorys[key]):
                sum[0] = key
                sum[1] = categorys[key]

        return sum[0]


knn = KNN()

datas = knn.get_simple_train_data()
print("datas:")
print(datas)

distances = knn.metrics(datas,[3,7])
print("distance:")
print(distances)

distances_sorted = knn.sort(distances)
print("sort:")
print(distances_sorted)

category = knn.gather(distances_sorted,3)
print("result:")
print(category)
