from os.path import join
import math as m
import numpy as np
import neuron
from scipy.special import ellipk, ellipe
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt


class MagneticField:
    def __init__(self, time_array, cell=None, rlc_type='under'):
        self.cell = cell                                    # Cell object
        self.children_dict = {}                             # Dict, children sections.
        self.parent_connection_dict = self.cell.get_dict_parent_connections()  # Dict, conn to parent in float [0, 1].
        self.parent_connection_idx = {}                     # Dict, seg idx of parent.
        self.has_children = {}                              # Dict, boolean.
        self.cell_idx_dict = {}                             # Dict, all seg idx per sec.
        self.par_child_vec_dict = {}                        # Dict, vec p-c connections.
        self.par_child_center_point = {}                    # Dict, center coord between parent and child
        self.par_child_l_c_dict = {}                        # Dict, l between center of p-c connections.
        self.par_child_r_a = {}                             # Dict, axial resistance for p-c connections.
        self.time_array = time_array                        # Time array [ms]
        self.timestep = self.cell.dt                        # Timestep [ms]
        self.rlc_type = rlc_type                            # Over- or under-dampened rlc circuit
        self.r_a = self.cell.get_axial_resistance()         # Axial res from NEURON [Mohm] (pre->curr).
        self.d = cell.d                                     # Comp width [um]
        self.l = cell.length                                # Comp length [um]
        self.l_c = np.zeros(self.cell.totnsegs - 1)         # Dist from center of parent comp to center of child comp
        self.comp_coords = np.zeros((self.cell.totnsegs + 1, 3))
        self.comp_vecs = np.zeros((self.cell.totnsegs, 3))
        self.c_comp_coords = np.zeros((self.cell.totnsegs, 3))
        self.c_comp_vecs = np.zeros((self.cell.totnsegs - 1, 3))
        self.input_current = np.zeros((self.cell.totnsegs, len(self.time_array)))

        self.manual_sim_data = {'Em': -70 * 10**-3,         # Resting membrane potential [V] (default -70)
                                'Rm': 30000 * 10**-4,       # Membrane resistivity [Ohm m**2]
                                'Ra': 150 * 10**-2,         # Axial resistivity [Ohm m] (default 150)
                                'Cm': 1 * 10**-2            # Specific membrane capacitance [F/m**2]
                                }
        self.coil_data = {'N': 30,                          # Number of loops in coil
                          'r_c': 0.02,                      # Coil radius [m]
                          'u': 4 * m.pi * 10**-7,           # Permeability constant, [N/A**2]
                          'cpd': 0.01                       # Distance from the coil to the plane of the E-field [m]
                          }
        self.rlc_circuit_data_over = {'type': 'over',       # Type of stimuli (over-dampened)
                                      'V0': 7500,           # Initial charge of capacitor [V]
                                      'R': 3,               # Resistance [ohm]
                                      'C': 200 * 10**-6,    # Conductance [F]
                                      'L': 165 * 10**-6     # Inductance in [H]
                                      }
        self.rlc_circuit_data_under = {'type': 'under',     # Type of stimuli (under-dampened)
                                       'V0': 700,           # Initial charge of capacitor [V]
                                       'R': 0.09,           # Resistance [ohm]
                                       'C': 200 * 10**-6,   # Conductance [F]
                                       'L': 13 * 10**-6     # Inductance [H]
                                       }

        self.E_t, self.I_coil = self._temporal_field()
        self.magnetic_flux_density = ((np.max(self.I_coil) * self.coil_data['u'] * self.coil_data['N'])
                                      / (2. * self.coil_data['r_c']))

    def _calc_omega_1(self, R, L):
        return R / (2. * L)

    def _calc_omega_2_over(self, w_1, L, C):
        return m.sqrt(w_1**2 - 1. / (L * C))

    def _calc_omega_2_under(self, w_1, L, C):
        return m.sqrt(1. / (L * C) - w_1**2)

    def _calc_E_t_over(self, V_0, C, t, w_1, w_2):
        return V_0 * C * w_2 * ((w_1 / w_2)**2 - 1) * m.exp(-w_1 * t) * (w_2 * m.cosh(w_2 * t) - w_1 * m.sinh(w_2 * t))

    def _calc_I_coil_over(self, V_0, C, t, w_1, w_2):
        return V_0 * C * w_2 * ((w_1 / w_2)**2 - 1) * m.exp(-w_1 * t) * m.sinh(w_2 * t)

    def _calc_E_t_under(self, V_0, C, t, w_1, w_2):
        return V_0 * C * w_2 * ((w_1 / w_2)**2 + 1) * m.exp(-w_1 * t) * (w_2 * m.cos(w_2 * t) - w_1 * m.sin(w_2 * t))

    def _calc_I_coil_under(self, V_0, C, t, w_1, w_2):
        return V_0 * C * w_2 * ((w_1 / w_2)**2 + 1) * m.exp(-w_1 * t) * m.sin(w_2 * t)

    def _temporal_field(self):
        E_t = np.zeros_like(self.time_array)
        I_coil = np.zeros_like(self.time_array)

        if self.rlc_type == 'over':
            V_0 = self.rlc_circuit_data_over['V0']
            R = self.rlc_circuit_data_over['R']
            C = self.rlc_circuit_data_over['C']
            L = self.rlc_circuit_data_over['L']
            w_1 = self._calc_omega_1(R, L)
            w_2 = self._calc_omega_2_over(w_1, L, C)
            for i in range(len(self.time_array)):
                E_t[i] = self._calc_E_t_over(V_0, C, self.time_array[i]*10**-3, w_1, w_2)
                I_coil[i] = self._calc_I_coil_over(V_0, C, self.time_array[i]*10**-3, w_1, w_2)

        elif self.rlc_type == 'under':
            V_0 = self.rlc_circuit_data_under['V0']
            R = self.rlc_circuit_data_under['R']
            C = self.rlc_circuit_data_under['C']
            L = self.rlc_circuit_data_under['L']
            w_1 = self._calc_omega_1(R, L)
            w_2 = self._calc_omega_2_under(w_1, L, C)
            for i in range(len(self.time_array)):
                E_t[i] = self._calc_E_t_under(V_0, C, self.time_array[i]*10**-3, w_1, w_2)
                I_coil[i] = self._calc_I_coil_under(V_0, C, self.time_array[i] * 10 ** -3, w_1, w_2)

        return E_t, I_coil

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

    def _calc_multisec_data(self):
        for sec in self.cell.allseclist:
            sec_name = sec.name()
            sec_idx = self.cell_idx_dict[sec_name]
            for i in sec_idx:
                self.c_comp_coords[i] = np.array([self.cell.x[i, 0] + self.cell.x[i, 1],
                                                  self.cell.y[i, 0] + self.cell.y[i, 0],
                                                  self.cell.z[i, 0] + self.cell.z[i, 0]]) * 0.5
        for sec in self.cell.allseclist:
            sec_name = sec.name()
            sec_idx = self.cell_idx_dict[sec_name]
            for i in sec_idx[:-1]:
                self.c_comp_vecs[i] = self.c_comp_coords[i + 1] - self.c_comp_coords[i]
            if self.has_children[sec_name]:
                for child in self.children_dict[sec_name]:
                    child_name = child.name()
                    child_idx = self.cell_idx_dict[child_name][0]
                    parent_idx = self.parent_connection_idx[child_name]
                    center_vec = self.c_comp_coords[child_idx] - self.c_comp_coords[parent_idx]

                    self.par_child_vec_dict[child_name] = center_vec
                    self.par_child_center_point[child_name] = self.c_comp_coords[parent_idx] + center_vec * 0.5
                    l_c = m.sqrt(center_vec[0] ** 2 + center_vec[1] ** 2 + center_vec[2] ** 2)
                    self.par_child_l_c_dict[child_name] = l_c
                    # r_a_parent in MOhm, l and d in micro m, sec=parent section, child=child section.
                    r_a_parent = (4 * sec.Ra * self.l[parent_idx] * 0.5) / (
                                m.pi * self.d[parent_idx] ** 2)
                    self.par_child_r_a[child_name] = r_a_parent + self.r_a[child_idx]

    def _calc_input_current_multisec(self):
        for sec in self.cell.allseclist:
            sec_name = sec.name()
            sec_idx = self.cell_idx_dict[sec_name]
            between_comps_coord = self.comp_coords[sec_idx[0]:sec_idx[-1]]
            if np.size(between_comps_coord) > 0:
                E_s_value, E_dir = self._spatial_field(between_comps_coord)
                for seg_idx in self.cell_idx_dict[sec_name][:-1]:
                    E_idx = seg_idx - self.cell_idx_dict[sec_name][0]
                    c_vec = self.c_comp_vecs[seg_idx]
                    for i in range(len(self.time_array)):
                        E_value = E_s_value[E_idx] * self.E_t[i]
                        I_a = self._calc_I_a_ctc(E_value, E_dir[E_idx], c_vec * 10**-6, self.r_a[seg_idx + 1])
                        self.input_current[seg_idx, i] += -I_a
                        self.input_current[seg_idx + 1, i] += I_a

            if self.has_children[sec_name]:
                for child in self.children_dict[sec_name]:
                    child_name = child.name()
                    parent_idx = self.parent_connection_idx[child_name]
                    child_idx = self.cell_idx_dict[child_name][0]
                    c_vec = self.par_child_vec_dict[child_name]
                    r_a_pc = self.par_child_r_a[child_name]
                    between_comps_coord = self.par_child_center_point[child_name]
                    E_s_value, E_dir = self._spatial_field(between_comps_coord)

                    for i in range(len(self.time_array)):
                        E_value = E_s_value * self.E_t[i]
                        I_a = self._calc_I_a_ctc(E_value, E_dir, c_vec * 10**-6, r_a_pc)
                        self.input_current[parent_idx, i] += -I_a
                        self.input_current[child_idx, i] += I_a

    def _calc_m_kk(self, r_c, rho, z):
        return (4. * r_c * rho) / ((r_c + rho)**2 + z**2)

    def _calc_E_s(self, N, r_c, u, rho, m_kk):
        return -((u * N)/(m.pi * m.sqrt(m_kk))) * m.sqrt(r_c / rho) * (ellipk(m_kk) * (1 - 0.5 * m_kk) - ellipe(m_kk))

    def _spatial_field(self, input_array):
        # input_array [um]
        N = self.coil_data['N']
        r_c = self.coil_data['r_c']
        u = self.coil_data['u']
        cpd = self.coil_data['cpd']
        if isinstance(input_array[0], np.ndarray):
            num_of_coords = len(input_array[0])
            E_s_value = np.zeros(len(input_array))
            E_s_direction = np.zeros_like(input_array)

            for i in range(len(input_array)):
                x = input_array[i, 0] * 10**-6
                y = input_array[i, 1] * 10**-6
                theta = m.atan2(y, x)
                if num_of_coords == 3:
                    z = abs(input_array[i, 2]) * 10**-6
                    E_s_direction[i] = np.array([-m.sin(theta), m.cos(theta), 0])
                else:
                    z = cpd
                    E_s_direction[i] = np.array([-m.sin(theta), m.cos(theta)])
                rho = m.sqrt(x**2 + y**2)
                if rho == 0:
                    rho += 1 * 10**-18

                m_kk = self._calc_m_kk(r_c, rho, z)
                E_s_value[i] = self._calc_E_s(N, r_c, u, rho, m_kk)

        elif isinstance(input_array[0], float):
            num_of_coords = len(input_array)
            x = input_array[0] * 10**-6
            y = input_array[1] * 10**-6
            theta = m.atan2(y, x)
            if num_of_coords == 3:
                z = abs(input_array[2]) * 10**-6
                E_s_direction = np.array([-m.sin(theta), m.cos(theta), 0])
            else:
                z = cpd
                E_s_direction = np.array([-m.sin(theta), m.cos(theta)])
            rho = m.sqrt(x**2 + y**2)
            if rho == 0:
                rho += 1 * 10**-18

            m_kk = self._calc_m_kk(r_c, rho, z)
            E_s_value = self._calc_E_s(N, r_c, u, rho, m_kk)

        return E_s_value, E_s_direction

    def _create_comp_data(self):
        self.comp_coords[0] = np.array([self.cell.x[0, 0], self.cell.y[0, 0], self.cell.z[0, 0]])
        for i in range(self.cell.totnsegs):
            self.comp_coords[i + 1] = np.array([self.cell.x[i, 1], self.cell.y[i, 1], self.cell.z[i, 1]])
            self.comp_vecs[i] = self.comp_coords[i + 1] - self.comp_coords[i]

    def _create_c_comp_data_single_sec(self):
        self.c_comp_coords[0] = (self.comp_coords[0] + self.comp_coords[1]) * 0.5
        for i in range(self.cell.totnsegs - 1):
            self.c_comp_coords[i + 1] = (self.comp_coords[i + 1] + self.comp_coords[i + 2]) * 0.5
            self.c_comp_vecs[i] = self.c_comp_coords[i + 1] - self.c_comp_coords[i]
            self.l_c[i] = m.sqrt(self.c_comp_vecs[i, 0]**2 + self.c_comp_vecs[i, 1]**2 + self.c_comp_coords[i, 2]**2)

    def _calc_I_a_ctc(self, E_value, E_dir, c_vec, r_a):
        # Return value is in [nA] from (mV / MOhm)
        return 1./r_a * E_value * np.dot(c_vec, E_dir) * 1000

    def _calc_input_current_ctc(self):
        between_comps_coord = self.comp_coords[1:-1]
        E_s_value, E_dir = self._spatial_field(between_comps_coord)
        for seg_idx in range(self.cell.totnsegs - 1):
            c_vec = self.c_comp_vecs[seg_idx]
            for i in range(len(self.time_array)):
                E_value = E_s_value[seg_idx] * self.E_t[i]
                I_a = self._calc_I_a_ctc(E_value, E_dir[seg_idx], c_vec * 10**-6, self.r_a[seg_idx + 1])
                self.input_current[seg_idx, i] += -I_a
                self.input_current[seg_idx + 1, i] += I_a

    def make_input_currents(self, method='ctc', multi_sec=False):
        self._create_comp_data()
        if not multi_sec:
            self._create_c_comp_data_single_sec()
            # Method: ctc = Center of parent To Center of child (parent = previous comp, child = current comp)
            if method == 'ctc':
                self._calc_input_current_ctc()
        else:
            self._create_cell_structure_dicts()
            self._calc_multisec_data()
            self._calc_input_current_multisec()

    def insert_im_neuron(self):
        input_vec = []
        synlist = []

        seg_idx = 0
        for sec in self.cell.allseclist:
            for seg in sec:
                noise_vec = neuron.h.Vector(-self.input_current[seg_idx])
                syn = neuron.h.ISyn(seg.x, sec=sec)
                syn.dur = 1e9
                syn.delay = 0
                noise_vec.play(syn._ref_amp, self.cell.dt)
                synlist.append(syn)
                input_vec.append(noise_vec)
                seg_idx += 1

        return input_vec, synlist

    def _calc_dv_mem(self, Em, Rm, Ra, Cm, v_prev_seg, v_curr_seg, v_next_seg, d, l, I_a):
        # Returns V from (V/(Ohm*m**2) * (s/(F/m**2))) and (A/m**2) * (s/(F/m**2))
        return ((Em - v_curr_seg) / Rm
                + (d / (4 * Ra)) * ((v_next_seg - v_curr_seg) / l**2 + (v_prev_seg - v_curr_seg) / l**2)
                + I_a / (m.pi * d * l)) * ((self.timestep * 10**-3) / Cm)

    def simulate_cell(self):
        v_mem = np.zeros((self.cell.totnsegs, len(self.time_array)))
        Em = self.manual_sim_data['Em']
        Rm = self.manual_sim_data['Rm']
        Ra = self.manual_sim_data['Ra']
        Cm = self.manual_sim_data['Cm']
        v_mem[:, 0] = Em
        for i in range(len(self.time_array) - 1):
            for seg_idx in range(self.cell.totnsegs):
                if seg_idx == 0:
                    v_mem_prev_seg = v_mem[seg_idx, i]
                    v_mem_next_seg = v_mem[seg_idx + 1, i]
                elif seg_idx == self.cell.totnsegs - 1:
                    v_mem_prev_seg = v_mem[seg_idx - 1, i]
                    v_mem_next_seg = v_mem[seg_idx, i]
                else:
                    v_mem_prev_seg = v_mem[seg_idx - 1, i]
                    v_mem_next_seg = v_mem[seg_idx + 1, i]

                v_mem_curr_seg = v_mem[seg_idx, i]
                v_mem[seg_idx, i + 1] = v_mem[seg_idx, i] + self._calc_dv_mem(Em, Rm, Ra, Cm,
                                                                              v_mem_prev_seg,
                                                                              v_mem_curr_seg,
                                                                              v_mem_next_seg,
                                                                              self.d[seg_idx] * 10**-6,
                                                                              self.l[seg_idx] * 10**-6,
                                                                              self.input_current[seg_idx, i] * 10**-9)
        return v_mem * 1000  # Return mV

    def plot_E_t(self):
        visual_prestart = (self.time_array[-1] - self.time_array[0])*0.05
        plt.plot(self.time_array, self.E_t)
        plt.xlim(self.time_array[0] - visual_prestart, self.time_array[-1])
        plt.title('$E_t$')
        plt.xlabel('Time [ms]')
        plt.ylabel('Change in current [A/s]')
        plt.savefig(join('E_t_fig.png'))
        plt.clf()

    def plot_I_coil(self):
        visual_prestart = (self.time_array[-1] - self.time_array[0])*0.05
        plt.plot(self.time_array, self.I_coil)
        plt.xlim(self.time_array[0] - visual_prestart, self.time_array[-1])
        plt.title('Coil current')
        plt.xlabel('Time [ms]')
        plt.ylabel('Current [A/s]')
        plt.savefig(join('I_coil_fig.png'))
        plt.clf()

    def plot_vmem(self, vmem, filename):
        visual_prestart = (self.time_array[-1] - self.time_array[0]) * 0.05
        for seg_idx in range(len(vmem[:, 0])):
            plt.plot(self.time_array, vmem[seg_idx], label=str(seg_idx+1))
        plt.xlim(self.time_array[0] - visual_prestart, self.time_array[-1])
        plt.title('Membrane Potential')
        plt.xlabel('Time [ms]')
        plt.ylabel('V [mV]')
        plt.legend(loc='upper right', title='Seg. #')
        plt.savefig(join(filename))
        plt.clf()

    def plot_vmem_hay(self, cell, filename):
        visual_prestart = (self.time_array[-1] - self.time_array[0]) * 0.05
        plot_idxs = np.array([0, cell.get_closest_idx(x=cell.x[0].mean(),
                                                      y=cell.y[0].mean() + 500,
                                                      z=cell.z[0].mean())])
        ax1 = plt.subplot(1, 2, 1, aspect=1, title='Cell Position', xlabel='x [cm]', ylabel='y [cm]')
        ax1.plot(cell.x.T * 10**-4, cell.y.T * 10**-4, c='k')
        ax2 = plt.subplot(1, 2, 2, xlabel='Time [ms]', ylabel='V [mV]', title='Membrane Potential')
        for seg_idx in plot_idxs:
            ax1.plot(cell.x[seg_idx].mean() * 10**-4, cell.y[seg_idx].mean() * 10**-4, 'o')
            ax2.plot(self.time_array, cell.vmem[seg_idx, :-1], label=str(seg_idx+1))
        plt.xlim(self.time_array[0] - visual_prestart, self.time_array[-1])
        plt.legend(loc='upper right', title='Seg. #')
        plt.savefig(join(filename))
        plt.clf()

    def plot_vmem_normalized(self, vmem, filename, rlc_type):
        amplitude = np.max(vmem)
        visual_prestart = (self.time_array[-1] - self.time_array[0]) * 0.05
        for seg_idx in range(self.cell.totnsegs):
            plt.plot(self.time_array, vmem[seg_idx] / amplitude, label=str(seg_idx+1))
        plt.xlim(self.time_array[0] - visual_prestart, self.time_array[-1])
        plt.title('Membrane Potential at $P_t$')
        plt.xlabel('Time [ms]')
        plt.ylabel('Normalized V')
        plt.legend(loc='upper right', title='Seg. #')
        plt.savefig(join(rlc_type + '_' + filename))
        plt.clf()

    def plot_hh_axon_heatmap(self, data):
        fig, ax = plt.subplots()
        v_max = 20
        v_min = -90
        pcm = plt.pcolormesh(data.T, cmap='inferno', vmin=v_min, vmax=v_max)
        plt.colorbar(pcm, label='mV')
        plt.title('Visualization of Membrane Potential in Axon')
        plt.xticks(np.linspace(0, self.cell.totnsegs, 17), np.arange(-8, 8 + 1, 1))
        plt.yticks(np.linspace(0, len(self.time_array), int(self.time_array[-1])), np.arange(self.time_array[0],
                                                                                        self.time_array[-1], 1))
        plt.xlabel('x [cm]')
        plt.ylabel('Time [ms]')
        plt.savefig(join('hh_axon_heatmap.png'))
        plt.clf()

    def _plot_heatmap(self, data, x_array, y_array, filename, title, v_min, v_max, spat_res, cbar_label='[$Vs/Am$]'):
        # Color-maps: x->columns, y->rows, transpose the data array to make it fit for cartesian coordinates (array.T)
        # Useful cmaps: PuOr, PRGn, BrBG
        # Coil is only correctly places in the plot for symmetric axes with 0 as center coordinate
        r_c = self.coil_data['r_c']
        dx = (x_array[-1] - x_array[0]) / spat_res
        r_c_adjusted = (r_c * 10**6)/dx
        coil = plt.Circle((int(len(x_array)/2.), int(len(y_array)/2.)), r_c_adjusted, color='gray', fill=False, lw=2)
        fig, ax = plt.subplots()
        pcm = plt.pcolormesh(data.T, cmap='BrBG', vmin=v_min, vmax=v_max)
        ax.add_patch(coil)
        coil.set_label('Coil')
        plt.colorbar(pcm, label=cbar_label)
        plt.title(title)
        plt.xticks(np.linspace(0, len(x_array), 9), np.arange(-4, 4 + 1, 1))
        plt.yticks(np.linspace(0, len(y_array), 9), np.arange(-4, 4 + 1, 1))
        plt.xlabel('x [cm]')
        plt.ylabel('y [cm]')
        plt.legend()
        plt.savefig(join(filename))
        plt.clf()

    def plot_E_s_heatmaps(self):
        spat_res = 100
        x_array = np.linspace(-40000, 40000, spat_res)
        y_array = np.copy(x_array)
        coordinates = np.zeros((len(x_array), len(y_array), 2))
        E_s_values = np.zeros((len(x_array), len(y_array)))
        E_s_directions = np.zeros((len(x_array), len(y_array), 2))

        for i in range(len(x_array)):
            for j in range(len(y_array)):
                coordinates[i, j] = np.array([x_array[i], y_array[j]])
            E_s_values[i, :], E_s_directions[i, :] = self._spatial_field(coordinates[i, :])

        E_s_x = E_s_values * E_s_directions[:, :, 0]
        E_s_y = E_s_values * E_s_directions[:, :, 1]
        E_s = np.sqrt(E_s_x**2 + E_s_y**2)
        E_amp_max = E_s * abs(np.max(self.E_t))
        v_max = np.max(E_s)
        v_min = -v_max
        self._plot_heatmap(E_s_x, x_array, y_array, 'E_s_x_heatmap', '$E_{s_x}$', v_min, v_max, spat_res)
        self._plot_heatmap(E_s_y, x_array, y_array, 'E_s_y_heatmap', '$E_{s_y}$', v_min, v_max, spat_res)
        self._plot_heatmap(E_s, x_array, y_array, 'E_s_heatmap', '$|E_s|$', v_min, v_max, spat_res)
        v_max_amp = np.max(E_amp_max)
        v_min_amp = -v_max_amp
        self._plot_heatmap(E_amp_max, x_array, y_array, 'E_amp_max_heatmap_' + self.rlc_type, 'Maximum $|E|$',
                           v_min_amp, v_max_amp, spat_res, cbar_label='[$V/m$]')

    def plot_neuron_placement(self):
        r_c = self.coil_data['r_c']
        coil = plt.Circle((0, 0), r_c*10**2, color='gray', fill=False, lw=2)
        fig, ax = plt.subplots()
        for i in range(self.cell.totnsegs):
            ax.plot(self.cell.x[i]*10**-4, self.cell.y[i]*10**-4, 'o', linestyle='-', color='b')
        ax.add_patch(coil)
        plt.title('Placement')
        plt.xlabel('x [cm]')
        plt.ylabel('y [cm]')
        xmax = np.max(self.cell.x)*10**-4
        xmin = np.min(self.cell.x)*10**-4
        ymax = np.max(self.cell.y)*10**-4
        ymin = np.min(self.cell.y)*10**-4
        if xmax == xmin:
            xmax += 0.5
            xmin -= 0.5
        else:
            xmax += (xmax-xmin)*0.30
            xmin -= (xmax-xmin)*0.30
        if ymax == ymin:
            ymax += 0.5
            ymin -= 0.5
        else:
            ymax += (ymax-ymin)*0.30
            ymin -= (ymax-ymin)*0.30
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(join('neuron_placement'))
        plt.clf()

    def plot_neuron_placement_multisec(self):
        r_c = self.coil_data['r_c']
        coil = plt.Circle((0, 0), r_c*10**2, color='gray', fill=False, lw=2)
        fig, ax = plt.subplots()
        for sec in self.cell.allseclist:
            sec_name = sec.name()
            sec_idx = self.cell_idx_dict[sec_name]
            for seg_idx in sec_idx:
                ax.plot(self.cell.x[seg_idx]*10**-4, self.cell.y[seg_idx]*10**-4, linestyle='-', color='b')
        ax.add_patch(coil)
        ax.set_aspect(1)
        plt.title('Placement')
        plt.xlabel('x [cm]')
        plt.ylabel('y [cm]')
        xmax = np.max(self.cell.x)*10**-4
        xmin = np.min(self.cell.x)*10**-4
        ymax = np.max(self.cell.y)*10**-4
        ymin = np.min(self.cell.y)*10**-4
        if xmax == xmin:
            xmax += 0.5
            xmin -= 0.5
        else:
            xmax += (xmax-xmin)*0.30
            xmin -= (xmax-xmin)*0.30
        if ymax == ymin:
            ymax += 0.5
            ymin -= 0.5
        else:
            ymax += (ymax-ymin)*0.30
            ymin -= (ymax-ymin)*0.30
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
#        plt.xlim(xmin, xmax)
#        plt.ylim(ymin, ymax)
        plt.savefig(join('neuron_placement'))
        plt.clf()

    def plot_vmem_vs_d(self, d_range, vmem):
        visual_prestart = (d_range[-1] - d_range[0]) * 0.05
        plt.plot(d_range, vmem)
        plt.xlim(d_range[0] - visual_prestart, d_range[-1] + visual_prestart)
        plt.title('Maximum Membrane Potential')
        plt.xlabel('Compartment Diameter [$\mu$m]')
        plt.ylabel('V [mV]')
        plt.savefig(join('vmem_vs_d'))
        plt.clf()

if __name__ == "__main__":
    pass