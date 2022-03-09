import math as m
import numpy as np
import scipy.special as ss
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt


class MagStim:
    def __init__(self, seg_pos, time_array):
        self.t_arr = time_array
        self.pos = seg_pos
        self.u_vec = np.array_like(seg_pos[0])


    def calc_seg_vec(self):
        for i in range(len(self.pos[0])):
            pass

    def calc_E_temp(self):
        pass

    def calc_E_spat(self):
        pass

    def calc_i_am(self):
        pass
