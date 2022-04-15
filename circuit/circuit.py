import numpy as np


def get_xor(p1, p2):
    q = np.zeros(p1.shape)
    for i in range(p1.shape[0]):
        q[i] = (p1[i] ^ p2[i])
    return q


class BooleanCircuit:
    def __init__(self, in_num, out_num, term_num, terms=None, outs=None):
        self.in_num = in_num
        self.out_num = out_num
        self.term_num = term_num
        self.terms = terms
        self.outs = outs

    def toMinimum(self):

        Minterm=np.hstack((self.terms,self.outs))
        while (True):
            fix_num = 0
            for i in range(self.term_num):
                #遍历terms每一项 进行最小项转换
                minterm = np.array(Minterm[i, :])
                position = minterm.where(minterm==-1)
                fix = np.array(minterm)
                if position:
                    fix_num += 1
                    fix[position[0]] = 0
                    Minterm=np.append(Minterm, fix, axis=0)
                    fix[position[0]] = 1
                    Minterm=np.append(Minterm, fix, axis=0)
                else:
                    Minterm=np.append(Minterm, fix, axis=0)
            if fix_num == 0:
                # 矩阵中不存在-1跳出
                break
            # 删除
            Minterm=np.delete(Minterm,np.s_[:self.term_num])
            self.term_num = np.shape(Minterm)[0]  # 获取新行数
        #去重
        unique=np.unique(np.array(Minterm[:,:self.in_num]))
        index=[]
        for i in range(self.term_num):
            if (Minterm[i,:self.in_num]==unique[0,:]).all():
                unique=np.delete(unique,0,index=0)
            else:
                index.append(i)
        Minterm=np.delete(Minterm,index,axis=0)
        self.terms=Minterm[:,:self.in_num]
        self.outs=Minterm[:,:self.in_num+1:self.in_num+self.outs]
  
class MPRM:
    def __init__(self, in_num, out_num, term_num, polarity, terms=None, outs=None):
        self.in_num = in_num
        self.out_num = out_num
        self.term_num = term_num
        self.terms = terms
        self.outs = outs
        self.polarity = polarity

    def fromBoolean(self,booleanCircuit, polarity):
        booleanCircuit.toMinimum()
        l = 1
        k = booleanCircuit.in_num - 1
        mitrix=np.hstack((booleanCircuit.terms,booleanCircuit.outs))
        for i in range(len(polarity)):
            #从高到低遍历极性的每一位

            for j in range(self.term_num):
                #对所有行进行操作
                if polarity[i,0] == 2:
                    if (self.in_num-i)-1<0:
                        break;
                elif polarity[i,0] == 0:
                    if mitrix[j,i]==0:
                        new = np.array(mitrix[j, :])
                        new[i] = 1
                        new = [new]

                        mitrix=np.append(mitrix,new,axis=0)

                elif polarity[i,0]==1:
                    if mitrix[j,i]==1:
                        new = np.array(mitrix[j, :])
                        new[i] = 0
                        new=[new]
                        mitrix=np.append(mitrix, new, axis=0)
            #找相同的行
            if np.shape(mitrix)[0]!=np.shape(np.unique(np.array(mitrix[:,:self.in_num]),axis=0))[0]:
              #新行旧行有相同
                index = []
                for j in range(self.term_num,np.shape(mitrix)[0]):
                    #枚举新行遍历旧行找到相同一对
                    for k in range(self.term_num):
                        if (mitrix[j,:self.in_num]==self.terms[k,:]).all():
                            #相同的新旧行o值异或
                            bits=0
                            for n in range(self.in_num,self.in_num+self.out_num):
                                #计算所有o值
                                bit = mitrix[k,n]^mitrix[j,n]
                                bits+=bit
                                #修改旧行的o值 新行为相同行后续删除
                                mitrix[k,n]=bit
                            if bits==0:
                                #o 值都为0删除
                                index.append(k)
                            #重复新行下标加入
                            index.append(j)
                #查找结束
                #删除重复新行
                mitrix=np.delete(mitrix,index,axis=0)
            if polarity[i,0]==1:
                #ik取反
                mitrix[:,i]=-(mitrix[:,i]-1)
            print(mitrix)
            self.term_num = np.shape(mitrix)[0]
            self.terms=mitrix[:self.term_num,:self.in_num]
            print("\n")


    def turnTo(self, polarity):
        Q = get_xor(self.polarity, polarity)
        l = 0
        k = 0
        new_terms = []
        new_outs = []
        while True:
            # S2
            if Q[k] != 0:
                if Q[k] == 1 and self.terms[l][k] == 1:
                    new_t = self.terms[l].copy()
                    new_t[k] = 0
                    new_o = self.outs[l].copy()
                    new_terms.append(new_t)
                    new_outs.append(new_o)
                elif Q[k] == 2 and self.terms[l][k] == 0:
                    new_t = self.terms[l].copy()
                    new_t[k] = 1
                    new_o = self.outs[l].copy()
                    new_terms.append(new_t)
                    new_outs.append(new_o)
                elif Q[k] == 3 and self.polarity[k] == 2 and self.terms[l][k] == 1:
                    new_t = self.terms[l].copy()
                    new_t[k] = 0
                    new_o = self.outs[l].copy()
                    new_terms.append(new_t)
                    new_outs.append(new_o)
                elif Q[k] == 3 and self.polarity[k] == 1 and self.terms[l][k] == 0:
                    new_t = self.terms[l].copy()
                    new_t[k] = 1
                    new_o = self.outs[l].copy()
                    new_terms.append(new_t)
                    new_outs.append(new_o)
                # S3
                l = l + 1
                if l < self.term_num:
                    continue  # 转S2
                # S4 S5
                self.merge(np.vstack(new_terms), np.vstack(new_outs))
                new_terms = []
                new_outs = []
            # S6
            l = 0
            k = k + 1
            if k >= self.in_num:
                break
        self.polarity = polarity

    def merge(self, new_terms, new_outs):
        de = np.zeros(new_terms.shape[0])
        for i in range(new_terms.shape[0]):
            for j in range(self.term_num):
                if np.array_equal(new_terms[i], self.terms[j]):
                    os = get_xor(self.outs[j], new_outs[i])
                    if np.array_equal(os, np.zeros(self.out_num)):
                        self.terms = np.delete(self.terms, j, 0)
                        self.outs = np.delete(self.outs, j, 0)
                        self.term_num -= 1
                        de[i] = 1
                        break
                    else:
                        self.outs[j] = os
                        de[i] = 1
                        break
        for i in range(new_terms.shape[0]):
            if de[i] == 0:
                self.terms = np.concatenate((self.terms, new_terms[i].reshape(1, -1)))
                self.outs = np.concatenate((self.outs, new_outs[i].reshape(1, -1)))
                self.term_num += 1



