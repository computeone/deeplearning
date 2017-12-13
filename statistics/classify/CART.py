import sys
import os

import numpy as np
import math

import operator

class CART:

    def __init__(self):
        print("init cart")


    def simple_train_set(self):

        train = np.array(
            [[1,1,125,0],
            [0,2,100,0],
            [0,1,70,0],
            [1,2,120,0],
            [0,0,95,1],
            [0,2,60,0],
            [1,0,220,0],
            [0,1,85,1],
            [0,2,75,0],
            [0,1,90,1]]
        )

        return train


    def select_node(self,train,type):

        node_type_count = {}
        m,n = train.shape
        node_type = {}
        node = []


        # 抽取各个属性出现的次数
        for i in range(m):

            for j in range(n):

                if(type[j] == 1):
                    continue

                if(j not in node_type):
                    node_type[j] = [train[i][j]]
                else:
                    if(train[i][j] not in node_type[j]):
                        node_type[j].append(train[i][j])

                if ((j, train[i][j], train[i][n - 1]) not in node_type_count):
                    node_type_count[(j, train[i][j], train[i][n - 1])] = 1
                else:
                    node_type_count[(j, train[i][j], train[i][n - 1])] += 1

        #计算总的GINI因子
        category = [0] * len(node_type[n - 1])

        for i in range(train.shape[0]):
            category[train[i][n - 1]] += 1

        base = 1

        for i in range(len(category)):
            base -= math.pow(category[i] / m,2)


        gini = []
        #计算GINI因子
        for i in range(len(node_type)):

            if(i not in node_type):
                continue

            for j in range(len(node_type[i])):

                if(len(node_type[i]) == 2 and j == 1):
                    continue

                sum = 0
                count = [0] * len(node_type[n - 1])

                for k in range(len(node_type[n - 1])):

                    if((i,node_type[i][j],node_type[n - 1][k]) not in node_type_count):
                        count[k] = 0
                    else:
                        count[k] = node_type_count[(i,node_type[i][j],node_type[n - 1][k])]

                    sum += count[k]

                p1 = 1 - math.pow(count[0] / sum,2) - math.pow(count[1] / sum,2)

                count_opposite = self.eval_opposite(node_type[i][j],node_type,node_type_count,[n -1 ,i])
                sum = count_opposite[0] + count_opposite[1]

                if(sum == 0):
                    continue

                p2 = 1 - math.pow(count_opposite[0] / sum,2) - math.pow(count_opposite[1] / sum,2)

                g = base - (count[0] + count[1]) / m * p1 - (count_opposite[0] + count_opposite[1]) / m * p2
                gini.append((i,node_type[i][j],g))

        for i in range(len(type)):
            if(type[i] == 1):
                p = self.eval_regression_node_gini(train.transpose()[i],train.transpose()[n - 1],node_type[n - 1],[m,base,i])
                gini.append(p)

        node = gini[0]
        max_gini = gini[0][2]
        for i in range(len(gini)):
            if(gini[i][2] > max_gini):
                max_gini = gini[i][2]
                node = gini[i]
        print("GINI:")
        print(gini)
        return node



    def eval_opposite(self,positive,type,type_count,info):

        count = [0] * len(type[info[0]])
        for i in range(len(type[info[1]])):
            if(i != positive):
                for j in range(len(type[3])):
                    if((info[1],i,j)  in type_count):
                        count[j] += type_count[(info[1],i,j)]
        return count


    def eval_regression_node_gini(self,train,classify,type,info):

        train_sort = sorted(train)

        split = [0] * (len(train) - 1)

        for i in range(len(train_sort) - 1):
            split[i] = (train_sort[i] + train_sort[i - 1]) / 2

        type_count = {}

        for i in range(len(train)):

            if((train[i],classify[i]) not in type_count):
                type_count[(train[i],classify[i])] = 1
            else:
                type_count[(train[i],classify[i])] += 1

        gini = [0] * len(split)

        for i in range(len(split)):

            positive = [0] * len(type)
            opposite = [0] * len(type)
            for j in range(len(train)):

                if(split[i] > train[j]):

                    for k in range(len(type)):
                        if((train[j],type[k]) in type_count):
                            positive[k] += type_count[(train[j],type[k])]
                else:
                    for k in range(len(type)):
                        if((train[j],type[k]) in type_count):
                            opposite[k] += type_count[(train[j],type[k])]

            p1 = 1 - math.pow(positive[0] / (positive[0] + positive[1]),2) - math.pow(positive[1] / (positive[0] + positive[1]),2)
            p2 = 1 - math.pow(opposite[0] / (opposite[0] + opposite[1]),2) - math.pow(opposite[1] / (opposite[0] + opposite[1]),2)

            gini[i] = info[1] -  (positive[0] + positive[1]) / info[0] * p1 - (opposite[0] + opposite[1]) / info[0] * p2

        return info[2],split[gini.index(min(gini))],min(gini)



    #得到选择node的全部train数据
    def get_divide_train(self,train,node,node_type):

        left_node = []
        right_node = []

        for i in range(train.shape[0]):
            if(node_type == 1):
                if(train[i][node[0]] > node[1]):
                    right_node.append(train[i])
                else:
                    left_node.append(train[i])
            else:

                if(train[i][node[0]] != node[1]):
                    right_node.append(train[i])
                else:
                    left_node.append(train[i])

        return [np.array(left_node),np.array(right_node)]


    def create_tree(self,train,type):

        # 判断是否终止产生子节点
        category = train[0][train[0].size - 1]
        m,n = train.shape
        terminate = True
        for i in range(m):
            if (train[i][n - 1] != category):
                terminate = False
                break

        if(n == 1):
            terminate = True

        if (terminate):
            print("leaf node:")
            print(train)
            return

        #选择一个属性作为接下来的划分点
        node = self.select_node(train,type)
        print("selected node:")
        print(node)
        node_train = self.get_divide_train(train,node,type[node[0]])
        print("left node:")
        print(node_train[0])
        print("right node:")
        print(node_train[1])
        self.create_tree(node_train[0],type)
        self.create_tree(node_train[1],type)




if __name__ == '__main__':

    cart = CART()

    train = cart.simple_train_set()

    print(train)

    cart.create_tree(train,[0,0,1,0])