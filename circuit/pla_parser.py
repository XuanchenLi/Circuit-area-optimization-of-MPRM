import numpy as np
import os
from circuit.circuit import BooleanCircuit


class Parser:
    def __init__(self, root_dir=""):
        self.root_dir = root_dir

    def parse(self, file_name):
        file_path = os.path.join(self.root_dir, file_name)
        f = open(file_path)
        in_num = 0
        term_num = 0
        out_num = 0
        terms = None
        cnt = 0
        flag = False
        s_terms = []
        while True:
            line = f.readline()
            line = line[:-1]
            if line:
                if line[0] == '.':
                    if line[1] == 'i' and line[2] == ' ':
                        in_num = int(line[3:])
                    elif line[1] == 'o' and line[2] == ' ':
                        out_num = int(line[3:])
                    elif line[1] == 'p' and line[2] == ' ':
                        term_num = int(line[3:])
                        terms = np.zeros((term_num, in_num))
                elif line[0] == '1' or line[0] == '0' or line[0] == '-':
                    if term_num == 0:
                        flag = True
                        s_terms.append(line[:in_num])
                    else:
                        term = line[:in_num]
                        for i in range(in_num):
                            if term[i] == '0':
                                terms[cnt][i] = 0
                            elif term[i] == '1':
                                terms[cnt][i] = 1
                            else:
                                terms[cnt][i] = -1
                    cnt += 1
            else:
                break
        f.close()
        if flag:
            term_num = cnt
            terms = np.zeros((term_num, in_num))
            for s in range(term_num):
                for i in range(in_num):
                    if s_terms[s][i] == '0':
                        terms[s][i] = 0
                    elif s_terms[s][i] == '1':
                        terms[s][i] = 1
                    else:
                        terms[s][i] = -1
        return BooleanCircuit(in_num, out_num, term_num, terms)

