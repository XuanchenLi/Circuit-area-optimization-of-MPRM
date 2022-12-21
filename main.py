import numpy as np
from greedy_frog import GreedyFrog
from mprm_frog import JumpFrog
import time
import os
from circuit.pla_parser import Parser
from circuit.circuit import *


def test(inputs, w, limits):
    return np.mat(inputs) * np.mat(w).T <= limits


def dp(n, w, p, c):
    f = [[0 for i in range(c + 1)] for i in range(n + 1)]  # 初始化
    for i in range(1, n + 1):
        for j in range(0, c + 1):
            f[i][j] = f[i - 1][j]
            if j >= w[0][i - 1]:
                f[i][j] = max(f[i][j], f[i - 1][j - w[0][i - 1]] + p[0][i - 1])
    return f[n][c]


def enu(file):
    parser = Parser("dataset/mcnc2")
    bool_c = parser.parse(file)
    # print(bool_c.terms, bool_c.outs)
    mprm = MPRM()
    nill = np.zeros((1, bool_c.in_num))[0]
    mprm.fromBoolean2(bool_c, nill)
    mina = 2 ** mprm.in_num
    for pp in range(3 ** mprm.in_num):
        nill = num_to_polarity(pp, mprm.in_num)
        mprm.turnTo(nill)
        if mina > mprm.get_area():
            mina = mprm.get_area()
        print(nill, mprm.get_area())
        # print(mprm.terms)
    print(mina)


def tests():
    bti = np.array(
        [[1, 1, -1],
        [-1, 1, 1],
        [1, -1, 1],
        [1, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
        [0, 0, 1]]
    )
    bto = np.array(
        [[0, 1],
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0]]
    )
    testi = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    testo = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 1]
    ])
    B1 = BooleanCircuit(3, 2, 7, bti, bto)
    M1 = MPRM()
    M1.fromBoolean2(B1, np.array([2, 1, 0]))
    print(M1.terms, M1.outs)
    M1.turnTo(np.array([0, 0, 0]))
    print(M1.terms, M1.outs)


def pack():
    n = 5000
    ans = 0
    best = 0
    w = np.random.randint(1, 10, (1, n))
    p = np.random.randint(1, 15, (1, n))
    c = int(np.sum(w) / 2)
    time_start1 = time.time()
    print("ground truth: ", dp(n, w, p, c))
    time_end1 = time.time()
    print('time cost', time_end1 - time_start1, 's')

    time_start2 = time.time()
    for i in range(5):
        model = GreedyFrog(20, 5, w, p, c)
        model.init()
        res = model.train(20)
        print(res[1])
        ans = ans + res[1]
        if res[1] > best:
            best = res[1]
    print("best: ", best)
    print("ave: ", ans / 5.0)
    time_end2 = time.time()
    print('time cost', (time_end2 - time_start2) / 5.0, 's')


if __name__ == '__main__':

    parser = Parser("dataset/mcnc2")
    files = os.listdir("dataset/mcnc2")
    files = ['clpl.pla']
    for f in files:
        ans = 0
        bool_c = parser.parse(f)
        best = 2**bool_c.in_num
        mprm = MPRM()
        nill = np.zeros((1, bool_c.in_num))[0]
        mprm.fromBoolean2(bool_c, nill)
        """
        mina = 2**mprm.in_num
        for pp in range(3**mprm.in_num):
            nill = num_to_polarity(pp, mprm.in_num)
            print(nill)
            mprm.turnTo(nill)
                if mina > mprm.get_area():
                    mina = mprm.get_area()
            print(nill, mprm.get_area())
        print(mina)
        os.system("pause")
        """

        time_start2 = time.time()
        for i in range(4):
            # print("m", i)
            model = JumpFrog(20, 5, mprm)
            model.init()
            gb, res = model.train(10)
            print(-res, best)
            ans = ans + res
            if -res < best:
                best = -res
            mprm.turnTo(nill)
        print(f, "best: ", best)
        print(f, "ave: ", -ans / 4.0)
        time_end2 = time.time()
        print('time cost', (time_end2 - time_start2) / 4.0, 's')









