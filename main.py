import numpy as np
from greedy_frog import GreedyFrog


if __name__ == '__main__':
    n = 500
    ans = 0
    best = 0
    w = np.random.randint(1, 10, (1, n))
    p = w + 5
    c = int(np.sum(w) / 2)
    for i in range(30):
        model = GreedyFrog(100, 10, w, p, c)
        model.init()
        res = model.train(40)
        print(res[1])
        ans = ans + res[1]
        if res[1] > best:
            best = res[1]
    print("best: ", best)
    print("ave: ", ans / 30.0)




