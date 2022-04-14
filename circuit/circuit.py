import numpy as np


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
                position = minterm.where(minterm==-1)
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
        return self.terms

  
class MPRM:
    def __init__(self, in_num, out_num, term_num, polarity, terms=None, outs=None):
        self.in_num = in_num
        self.out_num = out_num
        self.term_num = term_num
        self.terms = terms
        self.outs = outs
        self.polarity = polarity

    def fromBoolean(self, booleanCircuit, polarity):
        mixtir=booleanCircuit.toMinimum()
        l = 1
        k = booleanCircuit.in_num - 1
        for i in reversed(range(len(polarity))):
            #从高到低遍历极性的每一位
            for j in range(len(self.term_num)):
                #对所有行进行操作
                if polarity[i] == 2:
                    if i-1<0:
                        break;
                elif polarity[i] == 0:
                    if self.terms[j,i]==0:
                        new=self.terms[j,:]
                        new[i]=1
                        np.append(self.terms,new,axis=0)

                elif polarity[i]==1:
                    if self.terms[j,i]==1:
                        new = self.terms[j, :]
                        new[i] = 0
                        np.append(self.terms, new, axis=0)
            #找相同的行
            unique=np.unique(self.terms)
            if np.shape(self.terms)[0]!=np.shape(unique)[0]:
              #新行旧行有相同
              for j in range(self.term_num,np.shape(self.terms)[0]):
                #枚举新行遍历旧行找到相同一对
                for k in range(self.term_num):
                    if self.terms[j,:]==self.terms[k,:]:
                        #相同的新旧行o值异或
                        bits=0
                        for n in range(self.in_num,self.in_num+self.out_num):
                            #计算所有o值
                            bit = self.terms[k,n]^self.terms[j,n]
                            bits+=bit
                            #修改uniq的o值 可以不用第二次去重
                            unique[k,n]=bit
                        if bits==0:
                            #o 值都为0删除
                            np.delete(unique,k,axis=0)
        self.terms=unique
        self.term_num=np.shape(unique)[0]


    def turnTo(self, polarity):
        pass
