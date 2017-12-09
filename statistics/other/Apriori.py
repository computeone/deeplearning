import sys
import os

import numpy as np
import math

class Apriori:

    def __init__(self):
        print("init apriori")

    def simple_train_set(self):
        train = np.array(
            [['A','C','D'],
            ['B','C','E'],
            ['A','B','C','E'],
            ['B','E']]
        )

        return train

    #制作k项集
    def make_frequently_item(self,k_1_train):
        sigle_item_set = set()

        for items in k_1_train:
            for item in items:
                sigle_item_set.add(item)

        k_frequently_item = set()
        for item in k_1_train:

            for sigle in sigle_item_set:

                if(sigle  in "".join(item)):
                    continue

                k_item = "".join(item) + sigle
                k_item = "".join(sorted(k_item))
                k_frequently_item.add(k_item)

        return k_frequently_item

    #计算支持度和置信度
    def support(self,train,k_train):

        k_frequently_item = {}
        for item in train:

            for k_item in k_train:

                if(k_item not in k_frequently_item):
                    k_frequently_item[k_item] = 0

                is_continue = True

                for c in k_item:
                    if(c not in item):
                        is_continue = False
                        break

                if(is_continue):
                    k_frequently_item[k_item] += 1


        return k_frequently_item


    def remove_min_support(self,k_frequently_item,support):

        # 移除支持度小于support的项，
        remove_not_frequently_item = []
        for key in k_frequently_item:
            if (k_frequently_item[key] < support):
                remove_not_frequently_item.append(key)

        for key in remove_not_frequently_item:
            k_frequently_item.pop(key)

        return k_frequently_item


    def eval(self,train,support):

        #求得第一个频繁项
        k_frequently_item = {}
        for item in train:
            for i in item:
                if(i not in k_frequently_item):
                    k_frequently_item[i] = 1
                else:
                    k_frequently_item[i] += 1

        k_min_support_item = self.remove_min_support(k_frequently_item,support)

        k = 2
        while(len(k_min_support_item) > 1):

            k_frequently_item = self.make_frequently_item(k_min_support_item)
            k_min_support_item = self.support(train,k_frequently_item)

            k_min_support_item = self.remove_min_support(k_min_support_item,support)

            k += 1

        return k_min_support_item


if __name__ == '__main__':

    apriori = Apriori()
    train = apriori.simple_train_set()

    print(train)

    result = apriori.eval(train,2)

    print(result)


