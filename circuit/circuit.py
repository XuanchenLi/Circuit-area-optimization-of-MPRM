import numpy as np


class BooleanCircuit:
    def __init__(self, in_num, out_num, term_num, terms=None, outs=None):
        self.in_num = in_num
        self.out_num = out_num
        self.term_num = term_num
        self.terms = terms
        self.outs = outs


class MPRM:
    def __init__(self, in_num, out_num, term_num, polarity, terms=None, outs=None):
        self.in_num = in_num
        self.out_num = out_num
        self.term_num = term_num
        self.terms = terms
        self.outs = outs
        self.polarity = polarity

    def fromBoolean(self, booleanCircuit):
        pass

    def turnTo(self, polarity):
        pass