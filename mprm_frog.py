import numpy as np
from circuit.circuit import *
import matplotlib.pyplot as plt


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
        # print("one", num, self.dic[num])
        return self.dic[num]

    def sort(self):
        self.fitness_sub = np.argsort(-self.fitness.squeeze())
        self.fitness_sub = np.array(self.fitness_sub)

    def grouping(self):
        self.groups = self.fitness_sub.reshape((self.meme_size, self.group_num)).T

    def evolve(self, iterator_times):
        global_best = self.groups[0][0]
        for i in range(self.group_num):
            # print("e", i)
            for iter in range(iterator_times):
                local_best = self.groups[i][0]
                local_worst = self.groups[i][self.meme_size - 1]
                for k in range(self.meme_size):
                    if self.fitness[self.groups[i][k]] > self.fitness[local_best]:
                        local_best = self.groups[i][k]
                    if self.fitness[self.groups[i][k]] < self.fitness[local_worst]:
                        local_worst = self.groups[i][k]
                """
                tr = np.random.randint(0, self.meme_size, size=(1, 2))
                dis = (1.5 * np.random.rand(1, self.frogs[local_best].shape[0]) *
                       ((self.frogs[local_best] - self.frogs[tr[0][0]])
                        + (self.frogs[local_worst] - self.frogs[tr[0][1]])))
                """

                dis = (1.5 * np.random.rand(1, self.frogs[local_best].shape[0]) *
                       (self.frogs[local_best] - self.frogs[local_worst]))
                """
                dis = (np.sin((np.pi/ 2) * (1 / (iterator_times - iter + 1)))
                       *(self.frogs[local_best] - self.frogs[local_worst])).reshape(1, -1)
                """
                dis = np.round(dis)
                dis[dis > 2] = 2
                dis[dis < 0] = 0
                """
                for it in range(dis.shape[1]):
                    if dis[0][it] >= 1.5:
                        dis[0][it] = 2
                    elif 0.5 <= dis[0][it] < 1.5:
                        dis[0][it] = 1
                    elif dis[0][it] < 0.5:
                        dis[0][it] = 0
                """
                temp = self.frogs[local_worst] + dis
                # temp[temp < 0] = 0
                temp_f = self.get_fitness(temp)
                if temp_f > self.fitness[local_worst]:
                    self.frogs[local_worst] = temp
                    self.fitness[local_worst] = temp_f
                else:

                    dis = (1.5 * np.random.rand(1, self.frogs[global_best].shape[0]) *
                           (self.frogs[global_best] - self.frogs[local_worst]))
                    """
                    dis = (np.sin((np.pi / 2) * (1 / (iterator_times - iter + 1)))
                           * (self.frogs[local_best] - self.frogs[local_worst])).reshape(1, -1)
                    """
                    dis = np.round(dis)
                    dis[dis > 2] = 2
                    dis[dis < 0] = 0
                    """
                    for it in range(dis.shape[1]):
                        if dis[0][it] >= 1.5:
                            dis[0][it] = 2
                        elif 0.5 <= dis[0][it] < 1.5:
                            dis[0][it] = 1
                        elif dis[0][it] < 0.5:
                            dis[0][it] = 0
                    """
                    temp = self.frogs[local_worst] + dis
                    # temp[temp < 0] = 0
                    temp_f = self.get_fitness(temp)
                    if temp_f > self.fitness[local_worst]:
                        self.frogs[local_worst] = temp
                        self.fitness[local_worst] = temp_f
                    else:
                        new_1 = np.random.randint(0, 3, size=(1, self.dim))
                        new_2 = 2 - new_1
                        if self.get_fitness(new_2) > self.get_fitness(new_1):
                            new_1 = new_2
                        self.frogs[local_worst] = new_1
                        # self.frogs[local_worst] = 2 - self.frogs[local_worst]
                        self.fitness[local_worst] = self.get_fitness(self.frogs[local_worst])
                    global_best = self.elite(i, global_best)
            for j in range(self.meme_size):
                if self.fitness[global_best] < self.fitness[self.groups[i][j]]:
                    global_best = self.groups[i][j]
        return global_best

    def train(self, times):  # 全局迭代
        x = []
        y = []
        max_t = max(5, int(times * 0.5))
        cur_t = 0
        global_best = []
        for i in range(times):
            x.append(i)
            # print("t", i)
            self.fitness = self.get_fitness(self.frogs)
            self.sort()
            self.grouping()
            global_best = self.frogs[self.groups[0][0]]
            new_best = self.evolve(5)
            # print(i, ":", self.fitness[new_best])
            if self.get_fitness(global_best) >= self.fitness[new_best]:
                y.append(self.get_fitness(global_best)[0])
                cur_t += 1
                if cur_t >= max_t:
                    """
                    plt.plot(np.array(x), np.array(y))
                    plt.show()
                    """
                    return global_best, self.get_fitness(global_best)
            else:
                y.append(self.fitness[new_best])
                cur_t = 0
                global_best = self.frogs[new_best]
            # print("t", -self.get_fitness(global_best))
        """
        plt.plot(np.array(x), np.array(y))
        plt.show()
        """
        # print(test(self.frogs[global_best], self.weights, self.limitation))
        return global_best, self.get_fitness(global_best)

    def elite(self,group_index,global_best):

        local_worst = self.groups[group_index][self.meme_size - 1]
        local_best = self.groups[group_index][0]
        for i in range(int(0.4*self.meme_size)):
            local_worst = self.groups[group_index][self.meme_size - 1]
            local_best = self.groups[group_index][0]
            for k in range(self.meme_size):
                if self.fitness[self.groups[group_index][k]] > self.fitness[local_best]:
                    local_best = self.groups[group_index][k]
                if self.fitness[self.groups[group_index][k]] < self.fitness[local_worst]:
                    local_worst = self.groups[group_index][k]
            new=self.frogs[local_worst].copy()
            for j in range(new.shape[0]):
                new[j]=abs(np.random.normal(0, 1))*new[j]
            new[new > 1] = 2
            if self.fitness[local_worst]<self.get_fitness(new):
                self.frogs[local_worst]=new
                self.fitness[local_worst] = self.get_fitness(new)
                if self.fitness[local_best]<self.fitness[local_worst]:
                    self.frogs[local_best]=self.frogs[local_worst]
                    if self.fitness[global_best]<self.fitness[local_worst]:
                        global_best=local_worst

        for i in range(3):
            new=self.frogs[local_best].copy()
            for j in range(new.shape[0]):
                new[j]=(1+0.1*abs(np.random.normal(0,1)))*new[j]
            new[new>1]=2
            if self.fitness[local_best]<self.get_fitness(new):
                self.frogs[local_best]=new
                if self.fitness[global_best]<self.fitness[local_best]:
                    global_best=local_best
        return global_best

    def explosion(self, local_best):
        r = np.random.randint(4, self.dim)
        poi = np.random.randint(0, self.dim - r)
        new = self.frogs[local_best].copy()
        for i in range(poi, poi + r):
            new[i] = np.random.randint(0, 2)
        return new

