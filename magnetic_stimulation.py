import math as m
import numpy as np
import scipy.special as scis
import neuron


class MagStim:
    def __init__(self, LFPy_cell, time_array, start=[-40000, 10000, 0]):
        self.cell = LFPy_cell                                               # Cell object from LFPy.
        self.children_dict = {}                                             # Dict, children sections.
        self.parent_connection_dict = self.cell.get_dict_parent_connections()  # Dict, conn to parent in float [0, 1].
        self.parent_connection_idx = {}                                     # Dict, seg idx of parent.
        self.has_children = {}                                              # Dict, boolean.
        self.cell_idx_dict = {}                                             # Dict, all seg idx per sec.
        self.p_c_center_u_vec_dict = {}                                     # Dict, center unit vec for p-c connections.
        self.p_c_l_c_dict = {}                                              # Dict, l between center of p-c connections.
        self.p_c_Ra = {}                                                    # Dict, Ra for p-c connections.
        self.t_arr = time_array                                             # Time of neuron simulation in ms.
        self.Ra = self.cell.get_axial_resistance() * 10**6                  # Axial res from NEURON in Ohm (pre->curr).
        self.d = self.cell.d * 10**-6                                       # Segment diameter in m.
        self.l = self.cell.length * 10**-6                                  # Segment length in m.
        self.pos = (np.array([self.cell.x+start[0],
                              self.cell.y+start[1],
                              self.cell.z+start[2]]))*10**-6                # Segment position [[x], [y], [z]] in m.
        self.num_of_seg = len(self.d)                                       # Number of segments.
        self.l_c = np.zeros(self.num_of_seg - 1)                            # Distance between center of segments in m.
        self.seg_coord = np.zeros((self.num_of_seg + 1, 3))                 # Segment coordinates [x, y, z] in m.
        self.center_coord = np.zeros((self.num_of_seg, 3))                  # Center segment coordinates [x, y, z] in m.
        self.center_vec = np.zeros((self.num_of_seg - 1, 3))                # Displacement vector between segments in m.
        self.center_u_vec = np.zeros((self.num_of_seg - 1, 3))              # Displacement vector as a unit vector.
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
                                       'v0': 30,             # Initial charge of capacitor in V
                                       'R': 0.09,            # Resistance in ohm
                                       'C': 200 * 10**-6,    # Conductance in F
                                       'L': 13 * 10**-6}     # Inductance in H

    def _get_local_parent_connection_idx(self, parent_name, child_name):
        num_of_seg_in_sec = len(self.cell_idx_dict[parent_name])
        parent_connection = self.parent_connection_dict[child_name]
        if parent_connection == 0:
            return parent_connection
        else:
            return m.ceil(parent_connection * num_of_seg_in_sec) - 1

    def _create_cell_structure_dicts(self):
        seg_idx = 0
        for sec in self.cell.allseclist:
            sec_name = sec.name()
            list_of_seg_idx_in_sec = []
            for _ in sec:
                list_of_seg_idx_in_sec.append(seg_idx)
                seg_idx += 1
            self.cell_idx_dict[sec_name] = list_of_seg_idx_in_sec

            child_list = []
            for child in neuron.h.SectionRef(sec=sec).child:
                child_list.append(child)
            self.children_dict[sec_name] = child_list
            if len(child_list) > 0:
                self.has_children[sec_name] = True
            else:
                self.has_children[sec_name] = False

        for sec in self.cell.allseclist:
            sec_name = sec.name()
            if self.has_children[sec_name]:
                for child in self.children_dict[sec_name]:
                    child_name = child.name()
                    local_parent_idx = self._get_local_parent_connection_idx(sec_name, child_name)
                    self.parent_connection_idx[child_name] = self.cell_idx_dict[sec_name][local_parent_idx]

    def _calc_interseg_data(self):
        for i in range(self.num_of_seg):
            self.center_coord[i] = (self.seg_coord[i] + self.seg_coord[i + 1]) / 2.
        for sec in self.cell.allseclist:
            sec_name = sec.name()
            sec_idx = self.cell_idx_dict[sec_name]
            for i in sec_idx[:-1]:
                self.center_vec[i] = self.center_coord[i + 1] - self.center_coord[i]
                self.l_c[i] = m.sqrt(self.center_vec[i, 0] ** 2
                                     + self.center_vec[i, 1] ** 2
                                     + self.center_vec[i, 2] ** 2)
                self.center_u_vec[i] = self.center_vec[i] / self.l_c[i]

            if self.has_children[sec_name]:
                for child in self.children_dict[sec_name]:
                    child_name = child.name()
                    child_idx = self.cell_idx_dict[child_name][0]
                    parent_idx = self.parent_connection_idx[child_name]
#                    if sec_name == 'soma[0]':
#                        if self.parent_connection_dict[sec_name] == 0.5:
#                            pass

                    center_vec = self.center_coord[child_idx] - self.center_coord[parent_idx]
                    l_c = m.sqrt(center_vec[0]**2 + center_vec[1]**2 + center_vec[2]**2)
                    self.p_c_l_c_dict[child_name] = l_c
                    self.p_c_center_u_vec_dict[child_name] = center_vec / l_c
                    self.p_c_Ra[child_name] = self.Ra[parent_idx+1] + self.Ra[child_idx]

    def _calc_i_am_multisection(self, Ra, Ex, Ey, seg_pre, seg_post, center_u_vec):
        return -1. / Ra * ((Ex[seg_post] - Ex[seg_pre]) * center_u_vec[0]
                           + (Ey[seg_post] - Ey[seg_pre]) * center_u_vec[1])

    def _calc_I_a_multisection(self, Ra, Ex, Ey, seg_pre, seg_post, center_u_vec):
        return 1. / Ra * ((Ex[seg_post] + Ex[seg_pre]) * 0.5 * center_u_vec[0]
                          + (Ey[seg_post] + Ey[seg_pre]) * 0.5 * center_u_vec[1]) * self.l_c[0]

    def _calc_i_am_array_multisection(self):
        i_am_array = np.zeros((self.num_of_seg, len(self.t_arr)))
        for i in range(len(self.t_arr)):
            Ex = self.Ex_spat * self.i_E_temp[i]
            Ey = self.Ey_spat * self.i_E_temp[i]
            for sec in self.cell.allseclist:
                sec_name = sec.name()
                for seg_idx in self.cell_idx_dict[sec_name][:-1]:
                    i_am = self._calc_I_a_multisection(self.Ra[seg_idx + 1],
                                                        Ex,
                                                        Ey,
                                                        seg_idx,
                                                        seg_idx + 1,
                                                        self.center_u_vec[seg_idx])
                    i_am_array[seg_idx, i] += i_am
                    i_am_array[seg_idx + 1, i] += -i_am

                if self.has_children[sec_name]:
                    for child in self.children_dict[sec_name]:
                        child_name = child.name()
                        parent_idx = self.parent_connection_idx[child_name]
                        child_idx = self.cell_idx_dict[child_name][0]
                        center_u_vec = self.p_c_center_u_vec_dict[child_name]
                        i_am_p_c = self._calc_I_a_multisection(self.p_c_Ra[child_name],
                                                                Ex,
                                                                Ey,
                                                                parent_idx,
                                                                child_idx,
                                                                center_u_vec)
                        i_am_array[parent_idx, i] += i_am_p_c
                        i_am_array[child_idx, i] += -i_am_p_c
        return i_am_array

    def calc_input_current_multisection(self):
        self._create_cell_structure_dicts()
        self._calc_seg_u_vec()
        self._calc_interseg_data()
        self._calc_E_temp()
        self._calc_E_spat()

        input_current = self._calc_i_am_array_multisection()
        return input_current


    def _calc_seg_u_vec(self):
        self.seg_coord[0] = np.array([self.pos[0, 0, 0], self.pos[1, 0, 0], self.pos[2, 0, 0]])
        for i in range(self.num_of_seg):
            self.seg_coord[i + 1] = np.array([self.pos[0, i, 1], self.pos[1, i, 1], self.pos[2, i, 1]])
            dir_vec = self.seg_coord[i + 1] - self.seg_coord[i]
            self.u_vec[i] = dir_vec / self.l[i]

    def _calc_seg_u_vec_center(self):
        self.center_coord[0] = (self.seg_coord[0] + self.seg_coord[1]) / 2.
        for i in range(self.num_of_seg - 1):
            self.center_coord[i + 1] = (self.seg_coord[i + 1] + self.seg_coord[i + 2]) / 2.
            self.center_vec[i] = self.center_coord[i + 1] - self.center_coord[i]
            self.l_c[i] = m.sqrt(self.center_vec[i, 0]**2 + self.center_vec[i, 1]**2 + self.center_vec[i, 2]**2)
            self.center_u_vec[i] = self.center_vec[i] / self.l_c[i]

    def _calc_omega_under(self):
        R_u = self.rlc_circuit_data_under['R']
        C_u = self.rlc_circuit_data_under['C']
        L_u = self.rlc_circuit_data_under['L']
        w1_under = R_u / (2. * L_u)
        w2_under = m.sqrt((1. / (L_u * C_u)) - w1_under**2)
        return w1_under, w2_under

    def _calc_dIdt_under(self, V0, C, w1, w2, t):
        return V0 * C * w2 * ((w1 / w2)**2 + 1) * m.exp(-w1 * t * 10**-3) * (w2 * m.cos(w2 * t * 10**-3)
                                                                             - w1 * m.sin(w2 * t * 10**-3))

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
        return -1*(u*N)/(m.pi*m.sqrt(m_kk))*m.sqrt(r/h)*(scis.ellipk(m_kk)*(1 - 0.5*m_kk) -
                                                         scis.ellipe(m_kk))*-1*cos_theta

    def _calc_E_spat(self):
        N = self.coil_data['N']
        r = self.coil_data['r']
        u = self.coil_data['u']
        cpd = self.coil_data['cpd']

        for coord, i in zip(self.center_coord, range(len(self.center_coord))):
            dist_xy = m.sqrt(coord[0]**2 + coord[1]**2)
            dist_tot = m.sqrt(dist_xy**2 + cpd**2)
            cos_theta = dist_xy / dist_tot
            E_spat = self._calc_point_in_E_spat(N, r, u, cpd, dist_tot, cos_theta)
            cos_alpha = coord[0] / dist_xy
            sin_alpha = coord[1] / dist_xy
            self.Ex_spat[i] = E_spat * cos_alpha
            self.Ey_spat[i] = E_spat * sin_alpha

    def _calc_i_am(self, Ex, Ey, seg):
        return -1./(self.Ra[seg+1]) * ((Ex[seg + 1] - Ex[seg])*self.center_u_vec[seg, 0] +
                                       (Ey[seg + 1] - Ey[seg])*self.center_u_vec[seg, 1])
#        return (self.d[seg]/(4.*self.Ra*self.l[seg]**2))*((Ex[1] - Ex[0])*self.u_vec[seg, 0] +
#                                                          (Ey[1] - Ey[0])*self.u_vec[seg, 1])

    def _calc_I_a(self, Ex, Ey, seg):
        return self.l_c[seg]/self.Ra[seg+1] * ((Ex[seg + 1] + Ex[seg])*0.5*self.center_u_vec[seg, 0]
                                               + (Ey[seg + 1] + Ey[seg])*0.5*self.center_u_vec[seg, 1])
#        return self.d[seg]/(4.*self.Ra*self.l_c[seg])*((Ex[seg + 1] + Ex[seg])*0.5*self.center_u_vec[seg, 0]
#                                                        + (Ey[seg + 1] + Ey[seg])*0.5*self.center_u_vec[seg, 1])

    def _calc_i_am_array(self):
        i_am_array = np.zeros((self.num_of_seg, len(self.t_arr)))
        for i in range(len(self.t_arr)):
            Ex = self.Ex_spat * self.i_E_temp[i]
            Ey = self.Ey_spat * self.i_E_temp[i]
            for seg_idx in range(self.num_of_seg - 1):
                i_am = self._calc_i_am(Ex, Ey, seg_idx)
                i_am_array[seg_idx, i] += i_am
                i_am_array[seg_idx+1, i] += -i_am
        return i_am_array

    def _calc_I_a_array(self):
        I_a_arr = np.zeros((self.num_of_seg, len(self.t_arr)))
        for i in range(len(self.t_arr)):
            Ex = self.Ex_spat * self.i_E_temp[i]
            Ey = self.Ey_spat * self.i_E_temp[i]
            for seg_idx in range(self.num_of_seg - 1):
                I_a = self._calc_I_a(Ex, Ey, seg_idx)
                I_a_arr[seg_idx, i] += I_a
                I_a_arr[seg_idx+1, i] += -I_a
        return I_a_arr

    def calc_input_current(self, input_type='i_am'):
        self._create_cell_structure_dicts()
        self._calc_seg_u_vec()
        self._calc_seg_u_vec_center()
        self._calc_E_temp()
        self._calc_E_spat()

        if input_type =='i_am':
            input_current = self._calc_i_am_array()
        elif input_type =='I_a':
            input_current = self._calc_I_a_array()
        return input_current

    def _calc_seg_ext_quasipot(self, seg_idx, Ex, Ey):
        ext_Ex = (Ex[seg_idx + 1] + Ex[seg_idx]) / 2.
        ext_Ey = (Ey[seg_idx + 1] + Ey[seg_idx]) / 2.
        ext_Ez = 0
        return np.array([ext_Ex, ext_Ey, ext_Ez])

    def calc_ext_quasipot(self):
        self._calc_seg_u_vec()
        self._calc_seg_u_vec_center()
        self._calc_E_temp()
        self._calc_E_spat()
        ext_quasipot = np.zeros((self.num_of_seg, len(self.t_arr)))
        for i in range(len(self.t_arr)):
            Ex = self.Ex_spat * self.i_E_temp[i]
            Ey = self.Ey_spat * self.i_E_temp[i]
            for j in range(self.num_of_seg - 1):
                ext_quasipot[j + 1, i] = (ext_quasipot[j, i] -
                                          np.dot(0.5 * (self._calc_seg_ext_quasipot(j + 1, Ex, Ey) +
                                                        self._calc_seg_ext_quasipot(j, Ex, Ey)),
                                                 self.center_vec[j]))
        return ext_quasipot


if __name__ == "__main__":
    pass
