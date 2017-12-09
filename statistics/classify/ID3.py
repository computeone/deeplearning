import sys
import os

import numpy as np
import math

class ID3:

    def __init__(self):
        print("init id3")

    #simple train set 第一行类别分别为，age，income,student,credit_rating,buys_computer
    #age:0-youth 1-middle_aged, 2 - senior,income: 0-low,1-medium,2-high credit_rating:0-fair,1-excellent
    #buyes_compute:0-no,1-yes

    def simple_train_set(self):
        train = np.array([[0,2,0,0,0],
                [0,2,0,1,0],
                [1,2,0,0,1],
                [2,1,0,0,1],
                [2,0,1,0,1],
                [2,0,1,1,0],
                [1,0,1,1,1],
                [0,1,0,0,0],
                [0,0,1,0,1],
                [2,1,1,0,1],
                [0,1,1,1,1],
                [1,1,0,1,1],
                [1,2,1,0,1],
                [2,1,0,1,0]])

        return train

    def select_node(self,train):
        node_attr = {}


        #抽取各个属性出现的次数
        for item in train:
            for attr in range(item.size - 1):
                if((attr,item[attr],item[item.size - 1]) not in node_attr):
                    node_attr[(attr,item[attr],item[item.size - 1])] = 1
                else:
                    node_attr[(attr,item[attr],item[item.size - 1])] += 1

        #计算各个属性选项的信息熵值
        entropy = {}
        for item in node_attr:
            if((item[0],item[1]) not in entropy):
                no = 0
                yes = 0
                if((item[0],item[1],0) in node_attr.keys()):
                    no = node_attr[(item[0],item[1],0)]

                if((item[0],item[1],1) in node_attr.keys()):
                    yes = node_attr[(item[0],item[1],1)]

                ep = 0
                if(no !=0 and yes != 0):
                    ep = -1 * no/(yes + no) * math.log2(no /(yes + no)) - yes/(yes + no) * math.log2(yes / (yes + no))
                entropy[(item[0],item[1])] = ep

        #计算每一个属性的信息熵及其参考信息熵

        refer_entropy = 0
        attr_entropy = {}
        for item in entropy:

            no = 0
            yes = 0

            if((item[0],item[1],0) in node_attr):
                no = node_attr[(item[0],item[1],0)]

            if((item[0],item[1],1) in node_attr):
                yes = node_attr[(item[0],item[1],1)]

            if(item[0] not in attr_entropy):
                count = no + yes
                attr_entropy[item[0]] = entropy[(item[0],item[1])] * count / train.shape[0]

                refer_entropy += no
            else:
                count = no + yes
                attr_entropy[item[0]] += entropy[(item[0],item[1])] * count / train.shape[0]

                refer_entropy += no

        refer_entropy = refer_entropy /(train.shape[1] - 1)
        refer_entropy = -refer_entropy / train.shape[0] * math.log2(refer_entropy /train.shape[0]) - \
                        (train.shape[0] - refer_entropy) / train.shape[0] *  \
                        math.log2((train.shape[0] - refer_entropy) / train.shape[0])

        #print(attr_entropy)
        #print(refer_entropy)

        #计算信息熵增益，并且选择信息熵增益最高的

        max_gain_entropy = -1
        selected_attr = -1
        for item in attr_entropy:
            gain_entropy = refer_entropy - attr_entropy[item]

            if(gain_entropy > max_gain_entropy):
                max_gain_entropy = gain_entropy
                selected_attr = item

        return selected_attr


    #得到选择node的全部train数据
    def get_attr_train(self,train,attr,node):

        attr_train = []

        for item in train:
            if(item[attr] == node):
                tmp = []
                for i in range(len(item)):
                    if(i != attr):
                        tmp.append(item[i])

                attr_train.append(tmp)

        return np.array(attr_train)


    #得到node节点的所有孩子节点
    def get_childs_node(self,train,node):
        childs_node = []

        for item in train:
            if(item[node] not in childs_node):
                childs_node.append(item[node])

        return childs_node

#创建id3 tree，输出构建的node

    def create_tree(self,train):

    #判断是否终止产生子节点
        category = train[0][train[0].size - 1]
        terminate  = True
        for item in train:
            if(item[item.size - 1] != category):
                terminate = False
                break

        if(terminate):
            print("leaf node:")
            print(train)
            return

#选择一个属性作为接下来的划分点
        node = self.select_node(train)
        childs_node = self.get_childs_node(train,node)
        print("selected node:")
        print(node)
        for child in childs_node:
            child_train = self.get_attr_train(train,node,child)
            self.create_tree(child_train)


if __name__ == '__main__':

    id3 = ID3()
    train = id3.simple_train_set()
    id3.create_tree(train)











