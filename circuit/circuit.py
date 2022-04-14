import numpy as np


class BooleanCircuit:
    def __init__(self, in_num, out_num, term_num, terms=None):
        self.in_num = in_num
        self.out_num = out_num
        self.term_num = term_num
        self.terms = terms

    def nimterm_convert(self):
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

        self.teems = np.unique(self.terms, axis=0)  # 删除重复行

class MPRM:
    def __init__(self, in_num, out_num, term_num, polarity, terms=None):
        self.in_num = in_num
        self.out_num = out_num
        self.term_num = term_num
        self.terms = terms
        self.polarity = polarity

    def fromBoolean(self, booleanCircuit):
        pass

    def turnTo(self, polarity):
        pass