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

        self.manual_sim_data = {'Em': -70 * 10**-3,         # Resting membrane potential [V]
                                'Rm': 30000 * 10**-4,       # Membrane resistivity [Ohm m**2]
                                'Ra': 150 * 10**-2,         # Axial resistivity [Ohm m]
                                'Cm': 1 * 10**-2            # Specific membrane capacitance [F/m**2]
                                }
        self.coil_data = {'N': 30,                          # Number of loops in coil
                          'r_c': 0.02,                      # Coil radius [m]
                          'u': 4 * m.pi * 10**-7,           # Permeability constant, [N/A**2]
                          'cpd': 0.01                       # Distance from the coil to the plane of the E-field [m]
                          }
        self.rlc_circuit_data_over = {'type': 'over',       # Type of stimuli (over-dampened)
                                      'V0': 30,             # Initial charge of capacitor [V]
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
        self.magnetic_field_magnitude = ((np.max(self.I_coil) * self.coil_data['u'] * self.coil_data['N'])
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

    def _calc_m_kk(self, r_c, rho, z):
        return (4. * r_c * rho) / ((r_c + rho)**2 + z**2)

    def _calc_E_s(self, N, r_c, u, rho, m_kk):
        return -((u * N)/(m.pi * m.sqrt(m_kk))) * m.sqrt(r_c / rho) * (ellipk(m_kk) * (1 - 0.5 * m_kk) - ellipe(m_kk))

    def _spatial_field(self, input_array):
        # input_array [um]
        num_of_coords = len(input_array[0])
        N = self.coil_data['N']
        r_c = self.coil_data['r_c']
        u = self.coil_data['u']
        cpd = self.coil_data['cpd']
        E_s_value = np.zeros(len(input_array))
        E_s_direction = np.zeros_like(input_array)

        for i in range(len(input_array)):
            x = input_array[i, 0]*10**-6
            y = input_array[i, 1]*10**-6

            theta = m.atan2(y, x)
            if num_of_coords == 3:
                z = abs(input_array[i, 2])*10**-6
                E_s_direction[i] = np.array([-m.sin(theta), m.cos(theta), 0])
            else:
                z = cpd
                E_s_direction[i] = np.array([-m.sin(theta), m.cos(theta)])

            rho = m.sqrt(x**2 + y**2)
            if rho == 0:
                rho += 1 * 10**-18

            m_kk = self._calc_m_kk(r_c, rho, z)
            E_s_value[i] = self._calc_E_s(N, r_c, u, rho, m_kk)

        return E_s_value, E_s_direction

    def _create_positional_data_single_sec(self):
        self.comp_coords[0] = np.array([self.cell.x[0, 0], self.cell.y[0, 0], self.cell.z[0, 0]])
        for i in range(self.cell.totnsegs):
            self.comp_coords[i + 1] = np.array([self.cell.x[i, 1], self.cell.y[i, 1], self.cell.z[i, 1]])
            self.comp_vecs[i] = self.comp_coords[i + 1] - self.comp_coords[i]

        self.c_comp_coords[0] = (self.comp_coords[0] + self.comp_coords[1]) * 0.5
        for i in range(self.cell.totnsegs - 1):
            self.c_comp_coords[i + 1] = (self.comp_coords[i + 1] + self.comp_coords[i + 2]) * 0.5
            self.c_comp_vecs[i] = self.c_comp_coords[i + 1] - self.c_comp_coords[i]
            self.l_c[i] = m.sqrt(self.c_comp_vecs[i, 0]**2 + self.c_comp_vecs[i, 1]**2 + self.c_comp_coords[i, 2]**2)

    def _calc_I_a_ctc(self, E_value, E_dir, c_vec, r_a):
        # Return value is in [nA] from (mV / MOhm)
        return 1./r_a * E_value * np.dot(c_vec, E_dir) * 1000

    def _calc_input_current_ctc(self):
        r_a = self.r_a
        E_s_value, E_dir = self._spatial_field(self.c_comp_coords)
        for seg_idx in range(self.cell.totnsegs - 1):
            c_vec = self.c_comp_vecs[seg_idx]
            E_s_c = (E_s_value[seg_idx] + E_s_value[seg_idx + 1]) * 0.5
            for i in range(len(self.time_array)):
                E_value = E_s_c * self.E_t[i]
                I_a = self._calc_I_a_ctc(E_value, E_dir[seg_idx], c_vec * 10**-6, r_a[seg_idx + 1])
                self.input_current[seg_idx, i] += -I_a
                self.input_current[seg_idx + 1, i] += I_a

    def make_input_currents(self, method='ctc', multi_sec=False):
        if not multi_sec:
            self._create_positional_data_single_sec()
            # Method: ctc = Center of parent To Center of child (parent = previous comp, child = current comp)
            if method == 'ctc':
                self._calc_input_current_ctc()
            # Method: ste = Start of comp To End of comp
            elif method == 'ste':
                print('Not yet implemented')
        else:
            print('Not yet implemented')

    def insert_im_neuron(self):
        input_vec = []
        synlist = []

        seg_idx = 0
        for sec in self.cell.allseclist:
            for seg in sec:
                noise_vec = neuron.h.Vector(self.input_current[seg_idx])
                syn = neuron.h.ISyn(seg.x, sec=sec)
                syn.dur = 1e9
                syn.delay = 0
                noise_vec.play(syn._ref_amp, self.cell.dt)
                synlist.append(syn)
                input_vec.append(noise_vec)
                seg_idx += 1

        return input_vec, synlist

    def _calc_dv_mem(self, Em, Rm, Ra, Cm, v_prev_seg, v_curr_seg, v_next_seg, d, l, I_a):
        return ((Em - v_curr_seg) / Rm
                + (d / (4 * Ra)) * ((v_next_seg - v_curr_seg) / l**2 + (v_prev_seg - v_curr_seg) / l**2)
                - I_a / (m.pi * d * l)) * ((self.timestep * 10**-3) / Cm)

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
        for seg_idx in range(self.cell.totnsegs):
            plt.plot(self.time_array, vmem[seg_idx], label='seg '+str(seg_idx+1))
        plt.xlim(self.time_array[0] - visual_prestart, self.time_array[-1])
        plt.title('$Membrane Potential$')
        plt.xlabel('Time [ms]')
        plt.ylabel('V [mV]')
        plt.legend()
        plt.savefig(join(filename))
        plt.clf()

    def _plot_heatmap(self, data, x_array, y_array, filename, title, v_min, v_max, spat_res):
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
        plt.colorbar(pcm, label='[$Vs/Am$]')
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
        v_max = np.max(E_s)
        v_min = -v_max
        self._plot_heatmap(E_s_x, x_array, y_array, 'E_s_x_heatmap', '$E_{s_x}$', v_min, v_max, spat_res)
        self._plot_heatmap(E_s_y, x_array, y_array, 'E_s_y_heatmap', '$E_{s_y}$', v_min, v_max, spat_res)
        self._plot_heatmap(E_s, x_array, y_array, 'E_s_heatmap', '$|E_s|$', v_min, v_max, spat_res)

if __name__ == "__main__":
    pass