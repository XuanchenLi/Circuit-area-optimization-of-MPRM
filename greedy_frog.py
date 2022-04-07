import numpy as np


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


    def init(self):
        self.frogs = np.random.randint(0, 2, size=(self.population, self.dim))
        self.fitness, self.frogs = self.get_fitness_with_limit(self.frogs)

    def get_fitness(self, frogs):
        return np.mat(frogs) * np.mat(self.values).T

    def get_fitness_with_limit(self, frogs):
        frogs = frogs.reshape(-1, self.dim)
        for i in range(frogs.shape[0]):
            if self.sum_weight(frogs[i]) > self.limitation:
                frogs[i] = self.add(self.drop(frogs[i]))

        return np.mat(frogs) * np.mat(self.values).T, frogs

    def drop(self, frogs):
        w = self.sum_weight(frogs)[0]
        idx = self.dim - 1
        while w > self.limitation:
            while frogs[self.sorted_density_idx[idx]] == 0 and idx >= 0:
                idx -= 1
            frogs[idx] = 0
            w -= self.weights[self.sorted_density_idx[idx]]
        return frogs

    def add(self, frogs):
        w = self.sum_weight(frogs)[0]
        idx = 0
        while idx < self.dim and w < self.limitation:
            if frogs[self.sorted_density_idx[idx]] == 0 and \
                    w + self.weights[self.sorted_density_idx[idx]] < self.limitation:
                w += self.weights[self.sorted_density_idx[idx]]
                frogs[self.sorted_density_idx[idx]] = 1
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
        max_t = max(1, int(iterator_times * 0.1))
        cur_t = 0
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
                dis = (np.random.rand() *
                       (self.frogs[local_best] - self.frogs[local_worst])) \
                    .round().astype(np.int)
                temp = self.frogs[local_worst] + dis
                temp[temp < 0] = 0
                temp_f, temp = self.get_fitness_with_limit(temp)
                if temp_f > self.fitness[local_worst]:
                    self.frogs[local_worst] = temp
                    self.fitness[local_worst] = temp_f
                else:
                    dis = (np.random.rand() *
                           (self.frogs[global_best] - self.frogs[local_worst])) \
                        .round().astype(np.int)
                    temp = self.frogs[local_worst] + dis
                    temp[temp < 0] = 0
                    temp_f, temp = self.get_fitness_with_limit(temp)
                    if temp_f > self.fitness[local_worst]:
                        self.frogs[local_worst] = temp
                        self.fitness[local_worst] = temp_f
                    else:
                        self.frogs[local_worst] = np.random.randint(0, 2, size=(1, self.dim))
            for j in range(self.meme_size):
                if self.fitness[global_best] < self.fitness[self.groups[i][j]]:
                    global_best = self.groups[i][j]
        return global_best

    def train(self, times):
        max_t = max(1, int(times * 0.2))
        cur_t = 0
        global_best = []
        for i in range(times):
            self.fitness, self.frogs = self.get_fitness_with_limit(self.frogs)
            self.sort()
            self.grouping()
            global_best = self.frogs[self.groups[0][0]]
            new_best = self.evolve(100)
            if self.get_fitness(global_best) >= self.fitness[new_best]:
                cur_t += 1
                if cur_t >= max_t:
                    return global_best, self.get_fitness(global_best)
            else:
                cur_t = 0
                global_best = self.frogs[new_best]
        return global_best, self.get_fitness(global_best)
