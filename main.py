import numpy as np
from greedy_frog import GreedyFrog
import time


def test(inputs, w, limits):
    return np.mat(inputs) * np.mat(w).T <= limits


def dp(w, p, c):
    f = [[0 for i in range(c + 1)] for i in range(n + 1)]  # 初始化
    for i in range(1, n + 1):
        for j in range(0, c + 1):
            f[i][j] = f[i - 1][j]
            if j >= w[0][i - 1]:
                f[i][j] = max(f[i][j], f[i - 1][j - w[0][i - 1]] + p[0][i - 1])
    return f[n][c]


if __name__ == '__main__':
    n = 2000
    ans = 0
    best = 0
    w = np.random.randint(1, 10, (1, n))
    p = np.random.randint(1, 15, (1, n))
    c = int(np.sum(w) / 2)
    time_start1 = time.time()
    print("ground truth: ", dp(w, p, c))
    time_end1 = time.time()
    print('time cost', time_end1 - time_start1, 's')

    time_start2 = time.time()
    for i in range(5):
        model = GreedyFrog(20, 5, w, p, c)
        model.init()
        res = model.train(5)
        print(res[1])
        ans = ans + res[1]
        if res[1] > best:
            best = res[1]
    print("best: ", best)
    print("ave: ", ans / 5.0)
    time_end2 = time.time()
    print('time cost', (time_end2 - time_start2) / 5.0, 's')




