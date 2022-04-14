import numpy as np


class BooleanCircuit:
    def __init__(self, in_num, out_num, term_num, terms):
        self.in_num = in_num
        self.out_num = out_num
        self.term_num = term_num
        self.terms = terms
