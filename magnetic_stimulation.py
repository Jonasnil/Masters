import math as m
import numpy as np
import scipy.special as ss


class MagStim:
    def __init__(self, cell, time_array, start_pos=10100):
        self.t_arr = time_array                                             # Time of neuron simulation in ms.
        self.Ra = cell.get_axial_resistance() * 10**6                       # Axial res from NEURON in Ohm.
        self.d = cell.d * 10**-6                                            # Segment diameter in m.
        self.l = cell.length * 10**-6                                       # Segment length in m.
        self.pos = (np.array([cell.x, cell.y, cell.z]) + start_pos)*10**-6  # Segment position [[x], [y], [z]] in m.
        self.num_of_seg = len(self.d)                                       # Number of segments.
        self.seg_coord = self.u_vec = np.zeros((self.num_of_seg + 1, 3))    # Segment coordinates [x, y, z] in m.
        self.u_vec = np.zeros((self.num_of_seg, 3))                         # Unit vector parallel to segment axis.
        self.i_E_temp = np.zeros_like(self.t_arr)                           # Temporal part of E-field.
        self.Ex_spat = np.zeros(self.num_of_seg + 1)                        # Spatial part of E-field in x-dir.
        self.Ey_spat = np.zeros(self.num_of_seg + 1)                        # Spatial part of E-field in y-dir.

        self.coil_data = {'N': 30,                           # Number of loops in coil
                          'r': 0.02,                         # Coil radius in m
                          'u': 4 * m.pi * 10**-7,            # Permeability constant, N/A**2
                          'cpd': 0.01}                       # Distance from the coil to the plane of the E-field in m.

        self.rlc_circuit_data_under = {'type': 'under',      # Type of stimuli (under-dampened)
                                       'tau': 0.4,           # Time constant
                                       'v0': 900,            # Initial charge of capacitor in V
                                       'R': 0.09,            # Resistance in ohm
                                       'C': 200 * 10**-6,    # Conductance in F
                                       'L': 13 * 10**-6}     # Inductance in H

    def _calc_seg_u_vec(self):
        self.seg_coord[0] = np.array([self.pos[0, 0, 0], self.pos[1, 0, 0], self.pos[2, 0, 0]])
        for i in range(self.num_of_seg):
            self.seg_coord[i + 1] = np.array([self.pos[0, i, 1], self.pos[1, i, 1], self.pos[2, i, 1]])
            dir_vec = self.seg_coord[i + 1] - self.seg_coord[i]
            self.u_vec[i] = dir_vec / self.l[i]

    def _calc_omega_under(self):
        R_u = self.rlc_circuit_data_under['R']
        C_u = self.rlc_circuit_data_under['C']
        L_u = self.rlc_circuit_data_under['L']
        w1_under = R_u / (2. * L_u)
        w2_under = m.sqrt((1. / (L_u * C_u)) - w1_under**2)
        return w1_under, w2_under

    def _calc_dIdt_under(self, V0, C, w1, w2, t):
        return V0 * C * w2 * ((w1 / w2)**2 + 1) * m.exp(-w1 * t * 10**-3) * (w2 * m.cos(w2 * t * 10**-3) -
                                                                             w1 * m.sin(w2 * t * 10**-3))

    def _calc_E_temp(self):
        stimuli_type = self.rlc_circuit_data_under['type']

        if stimuli_type.lower() == 'under':
            V0 = self.rlc_circuit_data_under['v0']
            C = self.rlc_circuit_data_under['C']
            w1, w2 = self._calc_omega_under()
            for i in range(len(self.t_arr)):
                self.i_E_temp[i] = self._calc_dIdt_under(V0, C, w1, w2, self.t_arr[i])

    def _calc_m_kk(self, r, z, h):
        return (4*r*h)/((r + h)**2 + z**2)

    def _calc_point_in_E_spat(self, N, r, u, z, h, cos_theta):
        m_kk = self._calc_m_kk(r, z, h)
        return -1*(u*N)/(m.pi*m.sqrt(m_kk))*m.sqrt(r/h)*(ss.ellipk(m_kk)*(1 - (1/2)*m_kk) -
                                                         ss.ellipe(m_kk))*-1*cos_theta

    def _calc_E_spat(self):
        N = self.coil_data['N']
        r = self.coil_data['r']
        u = self.coil_data['u']
        cpd = self.coil_data['cpd']

        for coord, i in zip(self.seg_coord, range(self.num_of_seg + 1)):
            dist_xy = m.sqrt(coord[0]**2 + coord[1]**2)
            dist_tot = m.sqrt(dist_xy**2 + cpd**2)
            cos_theta = dist_xy / dist_tot
            E_spat = self._calc_point_in_E_spat(N, r, u, cpd, dist_tot, cos_theta)
            cos_alpha = coord[0] / dist_xy
            sin_alpha = coord[1] / dist_xy
            self.Ex_spat[i] = E_spat * cos_alpha
            self.Ey_spat[i] = E_spat * sin_alpha

    def _calc_delta_i_am(self, Ex, Ey, seg):
        return -1./(self.Ra[seg]) * ((Ex[seg + 1] - Ex[seg])*self.u_vec[seg, 0] +
                                     (Ey[seg + 1] - Ey[seg])*self.u_vec[seg, 1])
#        return (self.d[seg]/(4.*self.Ra*self.l[seg]**2))*((Ex[1] - Ex[0])*self.u_vec[seg, 0] +
#                                                          (Ey[1] - Ey[0])*self.u_vec[seg, 1])

    def calc_i_am(self):
        self._calc_seg_u_vec()
        self._calc_E_temp()
        self._calc_E_spat()
        delta_i_am = np.zeros((self.num_of_seg, len(self.t_arr)))
        for i in range(len(self.t_arr)-1):
            Ex = self.Ex_spat * self.i_E_temp[i]
            Ey = self.Ey_spat * self.i_E_temp[i]
            for seg_idx in range(self.num_of_seg):
                delta_i_am[seg_idx, i + 1] = self._calc_delta_i_am(Ex, Ey, seg_idx)
        return delta_i_am


if __name__ == "__main__":
    pass
