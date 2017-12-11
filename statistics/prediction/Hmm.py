import sys
import os

import numpy as np
import math

class Hmm:

    def __init__(self):
        print("init hmm")


    #simple train date
    #   转移矩阵
    #   'healthy':['healthy':0.7,'fever':0.3]
    #   'fever':['healthy':0.4,'fever':0.6]
    #
    #   状态转换概率矩阵
    #   'healthy':['normal':0.5,'cold':0.4,'dizzy':0.1]
    #   'fever':['normal':0.1,'cold':0.3,'dizzy':0.6]
    #
    #
    def simple_train_set(self):

        transition_probability = np.array(
            [[0.7,0.3],
            [0.4,0.6]]
        )

        emission_probability = np.array(
            [[0.5,0.4,0.1],
            [0.1,0.3,0.6]]
        )

        return [transition_probability,emission_probability]


    def evel_next_observation(self,start,transition,emission,state):

        vp = [0] * len(start)

        for i in range(len(start)):

            p = [0] * len(start)

            for j in range(len(start)):

                p[j] = transition[j][i] * emission[i][state] * start[j]

            vp[i] = np.max(p)

        return vp


    def get_max_index(self,array):

        index = 0;
        max = array[0]

        for i in range(len(array)):
            if(array[i] > max):
                max = array[i]
                index = i

        return index

    #viterbi aglorithm
    #给序列，求状态列表

    def eval(self,train,start,observation):

        max_state = [0] * len(observation)

        for i in range(len(start)):

            start[i] = start[i] * train[1][i][observation[0]]

        max_state[0] = self.get_max_index(start)

        print(start)

        for i in range(len(observation) - 1):

            start = self.evel_next_observation(start,train[0],train[1],observation[i + 1])

            max_state[i + 1] = self.get_max_index(start)

        return max_state


if __name__ == '__main__':

    hmm = Hmm()

    train = hmm.simple_train_set()

    print(train)

    start = [0.6,0.4]
    observation = [0,1,2]

    state = hmm.eval(train,start,observation)

    print(state)

