import numpy as np
from circuit.circuit import *


class JumpFrog:
    def __init__(self, population, meme_size, mprm):
        self.population = population
        self.meme_size = meme_size
        self.group_num = int(population / meme_size)
        self.dim = mprm.in_num
        self.global_best = None
        self.mprm = mprm
        self.dic = {}

    def init(self):
        self.frogs = np.random.randint(0, 3, size=(int(self.population / 3), self.dim))
        pair_frogs_1 = (self.frogs + 1) % 3
        pair_frogs_2 = (self.frogs + 2) % 3
        self.frogs = np.vstack((self.frogs, pair_frogs_1, pair_frogs_2))
        if self.frogs.shape[0] < self.population:
            rand_frog = np.random.randint(0, 3, size=(self.population - self.frogs.shape[0], self.dim))
            self.frogs = np.vstack((self.frogs, rand_frog))
        self.fitness = self.get_fitness(self.frogs)

    def get_fitness(self, frogs):
        res = []
        frogs = frogs.reshape(-1, self.dim).astype(int)
        # print(frogs.shape)
        for i in range(frogs.shape[0]):
            res.append(self.get_one_fitness(frogs[i]))
        res = np.array(res).reshape((frogs.shape[0], -1))
        return -res

    def get_one_fitness(self, frog):
        num = get_polarity_num(frog)
        if num in self.dic:
            return self.dic.get(num)
        self.mprm.turnTo(frog)
        self.dic[num] = self.mprm.get_area()
        print("one", num, self.dic[num])
        return self.dic[num]

    def sort(self):
        self.fitness_sub = np.argsort(-self.fitness.squeeze())
        self.fitness_sub = np.array(self.fitness_sub)

    def grouping(self):
        self.groups = self.fitness_sub.reshape((self.meme_size, self.group_num)).T

    def evolve(self, iterator_times):
        global_best = self.groups[0][0]
        for i in range(self.group_num):
            print("e", i)
            for iter in range(iterator_times):
                local_best = self.groups[i][0]
                local_worst = self.groups[i][self.meme_size - 1]
                for k in range(self.meme_size):
                    if self.fitness[self.groups[i][k]] > self.fitness[local_best]:
                        local_best = self.groups[i][k]
                    if self.fitness[self.groups[i][k]] < self.fitness[local_worst]:
                        local_worst = self.groups[i][k]

                dis = (np.random.rand(1, self.frogs[local_best].shape[0]) *
                       (self.frogs[local_best] - self.frogs[local_worst]))
                for it in range(dis.shape[1]):
                    if dis[0][it] >= 1.5:
                        dis[0][it] = 2
                    elif 0.5 <= dis[0][it] < 1.5:
                        dis[0][it] = 1
                    elif dis[0][it] < 0.5:
                        dis[0][it] = 0
                temp = self.frogs[local_worst] + dis
                # temp[temp < 0] = 0
                temp_f = self.get_fitness(temp)
                if temp_f > self.fitness[local_worst]:
                    self.frogs[local_worst] = temp
                    self.fitness[local_worst] = temp_f
                else:
                    dis = (np.random.rand(1, self.frogs[global_best].shape[0]) *
                           (self.frogs[global_best] - self.frogs[local_worst]))
                    for it in range(dis.shape[1]):
                        if dis[0][it] >= 1.5:
                            dis[0][it] = 2
                        elif 0.5 <= dis[0][it] < 1.5:
                            dis[0][it] = 1
                        elif dis[0][it] < 0.5:
                            dis[0][it] = 0
                    temp = self.frogs[local_worst] + dis
                    # temp[temp < 0] = 0
                    temp_f = self.get_fitness(temp)
                    if temp_f > self.fitness[local_worst]:
                        self.frogs[local_worst] = temp
                        self.fitness[local_worst] = temp_f
                    else:
                        self.frogs[local_worst] = np.random.randint(0, 3, size=(1, self.dim))
                        self.fitness[local_worst] = self.get_fitness(self.frogs[local_worst])
            for j in range(self.meme_size):
                if self.fitness[global_best] < self.fitness[self.groups[i][j]]:
                    global_best = self.groups[i][j]
        return global_best

    def train(self, times):
        max_t = max(5, int(times * 0.5))
        cur_t = 0
        global_best = []
        for i in range(times):
            print("t", i)
            self.fitness = self.get_fitness(self.frogs)
            self.sort()
            self.grouping()
            global_best = self.frogs[self.groups[0][0]]
            new_best = self.evolve(5)
            # print(i, ":", self.fitness[new_best])
            if self.get_fitness(global_best) >= self.fitness[new_best]:
                cur_t += 1
                if cur_t >= max_t:
                    return global_best, self.get_fitness(global_best)
            else:
                cur_t = 0
                global_best = self.frogs[new_best]
            print("t", -self.get_fitness(global_best))
        # print(test(self.frogs[global_best], self.weights, self.limitation))
        return global_best, self.get_fitness(global_best)
