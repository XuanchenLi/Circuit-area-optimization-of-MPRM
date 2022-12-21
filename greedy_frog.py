import numpy as np
import matplotlib.pyplot as plt


def test(inputs, w, limits):
    return np.mat(inputs) * np.mat(w).T <= limits


class GreedyFrog:
    def __init__(self, population, meme_size, weights, values, limitation):
        self.population = population
        self.meme_size = meme_size
        self.group_num = int(population / meme_size)
        self.weights = weights.flatten()
        self.values = values.flatten()
        self.limitation = limitation
        assert weights.shape == values.shape
        self.dim = weights.shape[1]
        self.global_best = None
        self.density = np.array(values).astype(np.float) / (np.array(weights).astype(np.float) + 1e-5)
        self.sorted_density_idx = np.argsort(-self.density).flatten()
        self.sorted_weight_idx = np.argsort(-self.weights).flatten()

    def init(self):
        self.frogs = np.random.randint(0, 2, size=(int(self.population / 2), self.dim))
        pair_frogs = 1 - self.frogs
        self.frogs = np.vstack((self.frogs, pair_frogs))
        self.fitness, self.frogs = self.get_fitness_with_limit(self.frogs)

    def get_fitness(self, frogs):
        return np.mat(frogs) * np.mat(self.values).T

    def get_fitness_with_limit(self, frogs):
        frogs = frogs.reshape(-1, self.dim)
        for i in range(frogs.shape[0]):
            if self.sum_weight(frogs[i]) > self.limitation:
                frogs[i] = self.drop(frogs[i])
                frogs[i] = self.add(frogs[i])
        return np.mat(frogs) * np.mat(self.values).T, frogs

    def drop(self, frogs):
        w = self.sum_weight(frogs)[0]
        idx = self.dim - 1
        while w > self.limitation:
            while frogs[self.sorted_density_idx[idx]] == 0 and idx > 0:
                idx = idx - 1
            frogs[self.sorted_density_idx[idx]] = 0
            w -= self.weights[self.sorted_density_idx[idx]]
        # print(test(frogs, self.weights, self.limitation))
        return frogs

    def add(self, frogs):
        w = self.sum_weight(frogs)[0]
        idx = 0
        while idx < self.dim and w < self.limitation:
            if frogs[self.sorted_weight_idx[idx]] == 0 and \
                    w + self.weights[self.sorted_weight_idx[idx]] < self.limitation:
                w += self.weights[self.sorted_weight_idx[idx]]
                frogs[self.sorted_weight_idx[idx]] = 1
            idx += 1
        return frogs

    def sort(self):
        self.fitness_sub = np.argsort(-self.fitness.squeeze())
        self.fitness_sub = np.array(self.fitness_sub)

    def sum_weight(self, frogs):
        return np.mat(frogs) * np.mat(self.weights).T

    def grouping(self):
        self.groups = self.fitness_sub.reshape((self.meme_size, self.group_num)).T

    def evolve(self, iterator_times):
        global_best = self.groups[0][0]
        for i in range(self.group_num):
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
                dis[dis >= 0.5] = 1
                dis[dis < 0.5] = 0
                temp = self.frogs[local_worst] + dis
                # temp[temp < 0] = 0
                temp_f, temp = self.get_fitness_with_limit(temp)
                if temp_f > self.fitness[local_worst]:
                    self.frogs[local_worst] = temp
                    self.fitness[local_worst] = temp_f
                else:
                    dis = (np.random.rand(1, self.frogs[global_best].shape[0]) *
                           (self.frogs[global_best] - self.frogs[local_worst]))
                    dis[dis >= 0.5] = 1
                    dis[dis < 0.5] = 0
                    temp = self.frogs[local_worst] + dis
                    # temp[temp < 0] = 0
                    temp_f, temp = self.get_fitness_with_limit(temp)
                    if temp_f > self.fitness[local_worst]:
                        self.frogs[local_worst] = temp
                        self.fitness[local_worst] = temp_f
                    else:
                        self.frogs[local_worst] = np.random.randint(0, 2, size=(1, self.dim))
                        self.fitness[local_worst], self.frogs[local_worst] = \
                            self.get_fitness_with_limit(self.frogs[local_worst])
            for j in range(self.meme_size):
                if self.fitness[global_best] < self.fitness[self.groups[i][j]]:
                    global_best = self.groups[i][j]
        return global_best

    def train(self, times):
        x = []
        y = []
        max_t = max(5, int(times * 0.5))
        cur_t = 0
        global_best = []
        for i in range(times):
            x.append(i)
            self.fitness, self.frogs = self.get_fitness_with_limit(self.frogs)
            self.sort()
            self.grouping()
            global_best = self.frogs[self.groups[0][0]]
            new_best = self.evolve(15)
            # print(i, ":", self.fitness[new_best])
            if self.get_fitness(global_best) >= self.fitness[new_best]:
                y.append(self.get_fitness(global_best)[0])
                cur_t += 1
                if cur_t >= max_t:
                    plt.plot(np.array(x), np.array(y).squeeze(2))
                    plt.show()
                    return global_best, self.get_fitness(global_best)
            else:
                y.append(self.fitness[new_best])
                cur_t = 0
                global_best = self.frogs[new_best]
        # print(test(self.frogs[global_best], self.weights, self.limitation))
        plt.plot(np.array(x), np.array(y).squeeze(0))
        plt.show()
        return global_best, self.get_fitness(global_best)

