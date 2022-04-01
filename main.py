import numpy as np
from greedy_frog import GreedyFrog

if __name__ == '__main__':
    model = GreedyFrog(6, 2, np.array([1, 2, 1, 1, 3, 2]), np.array([3, 1, 4, 2, 4, 1]), 6)
    model.init()
    res = model.train(100)
    print(res)


