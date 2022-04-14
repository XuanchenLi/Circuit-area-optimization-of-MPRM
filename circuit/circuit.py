import numpy as np


def get_xor(p1, p2):
    q = np.zeros(p1.shape)
    for i in p1.shape[1]:
        q[i] = p1[i] ^ p2[i]
    return q


class BooleanCircuit:
    def __init__(self, in_num, out_num, term_num, terms=None, outs=None):
        self.in_num = in_num
        self.out_num = out_num
        self.term_num = term_num
        self.terms = terms
        self.outs = outs

    def toMinimum(self):
        while (True):
            fix_num = 0
            for i in range(self.term_num):
                minterm = self.terms[i, :]
                position = minterm.where()
                fix = minterm
                if position:
                    fix_num += 1
                    fix[position[0]] = 0
                    np.append(self.terms, fix, axis=0)
                    fix[position[0]] = 1
                    np.append(self.terms, fix, axis=0)
                else:
                    np.append(self.terms)
            if fix_num == 0:
                # 矩阵中不存在-1跳出
                break
            np.delete(self.terms, np.s_[:self.term_num])  # 删除
            self.term_num = np.shape(self.terms)[0]  # 获取新行数

        self.terms = np.unique(self.terms, axis=0)  # 删除重复行

  
class MPRM:
    def __init__(self, in_num, out_num, term_num, polarity, terms=None, outs=None):
        self.in_num = in_num
        self.out_num = out_num
        self.term_num = term_num
        self.terms = terms
        self.outs = outs
        self.polarity = polarity

    def fromBoolean(self, booleanCircuit, polarity):
        booleanCircuit.toMinimum()
        l = 1
        k = booleanCircuit.in_num - 1
        

    def turnTo(self, polarity):
        Q = get_xor(self.polarity, polarity)
        l = 1
        t = self.term_num
        k = self.in_num - 1
        new_terms = []
        new_outs = []
        while True:
            # S2
            if Q[k] != 0:
                if Q[k] == 1 and self.terms[l][k] == 1:
                    new_t = self.terms[l][k]
                    new_t[k] = 0
                    new_o = self.outs[l][k]
                    new_terms.append(new_t)
                    new_outs.append(new_o)
                elif Q[k] == 2 and self.terms[l][k] == 0:
                    new_t = self.terms[l][k]
                    new_t[k] = 1
                    new_o = self.outs[l][k]
                    new_terms.append(new_t)
                    new_outs.append(new_o)
                elif Q[k] == 3 and self.polarity[k] == 2 and self.terms[l][k] == 1:
                    new_t = self.terms[l][k]
                    new_t[k] = 0
                    new_o = self.outs[l][k]
                    new_terms.append(new_t)
                    new_outs.append(new_o)
                elif Q[k] == 3 and self.polarity[k] == 1 and self.terms[l][k] == 0:
                    new_t = self.terms[l][k]
                    new_t[k] = 1
                    new_o = self.outs[l][k]
                    new_terms.append(new_t)
                    new_outs.append(new_o)
                # S3
                l = l + 1
                if l <= t:
                    continue  # 转S2
                # S4 S5
                self.merge(np.vstack(new_terms), np.vstack(new_outs))
            # S6
            l = 1
            k = k - 1
            if k < 0:
                break

    def merge(self, new_terms, new_outs):
        de = np.zeros(new_terms.shape[0])
        for i in range(new_terms.shape[0]):
            for j in range(self.term_num):
                if np.array_equal(new_terms[i], self.terms[j]):
                    os = get_xor(self.outs[j], new_outs)
                    if np.array_equal(os, np.zeros(self.out_num)):
                        np.delete(self.terms, j)
                        np.delete(self.outs, j)
                        self.term_num -= 1
                        de[i] = 1
                        break
                    else:
                        self.outs[j] = os
                        de[i] = 1
                        break
        for i in range(new_terms.shape[0]):
            if de[i] == 0:
                self.terms = np.concatenate(self.terms, new_terms[i])
                self.outs = np.concatenate(self.outs, new_outs[i])
                self.term_num += 1



