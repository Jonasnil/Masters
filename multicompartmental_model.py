from os.path import join
import math as m
import numpy as np
import scipy.special as scis
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt


# Scale
radius = 40000*10**-6                       # Scale from the center of the coil in m
spatial_resolution = 100*10**-6             # Spatial resolution in m
time = 10                                   # milli sec
timestep = 0.001                            # milli sec
time_array = np.arange(0, time + timestep, timestep)


# Compartmental neuron model
class MultiCompartmentNeuron:
    def __init__(self, time_array, start_pos=[10100, 10100],
                 d=1*10**-6, Em=-70*10**-3, Cm=1*10**-6, Rm=30000*10-4, Ra=100*10-2):
        self.Em = Em
        self.Cm = Cm
        self.Rm = Rm
        self.Ra = Ra
        self.t_arr = time_array
        # seg_coords is start/divide/end pos of total comp-structure in [x, y, z] coords (micro m converted to m)
        self.seg_coord = np.array([[x + start_pos[0], 0 + start_pos[1]] for x in np.arange(0, 1001, 100)])*10**-6
        self.num_of_seg = len(self.seg_coord[:, 0]) - 1
        self.l = np.zeros(self.num_of_seg)
        self.dir_vec = np.zeros((self.num_of_seg, 2))
        self.center_coord = np.zeros((self.num_of_seg, 2))
        self.center_vec = np.zeros((self.num_of_seg - 1, 2))
        self.center_u_vec = np.zeros((self.num_of_seg - 1, 2))
        self.l_c = np.zeros(self.num_of_seg - 1)
        self.d = np.ones(self.num_of_seg)*d
        self.i_E_temp = np.zeros_like(self.t_arr)           # Temporal part of E-field.
        self.Ex_spat = np.zeros(self.num_of_seg + 1)        # Spatial part of E-field in x-dir.
        self.Ey_spat = np.zeros(self.num_of_seg + 1)        # Spatial part of E-field in y-dir.
        self.V = np.zeros((self.num_of_seg, self.t_arr.shape[0]))
        self.I_a = np.zeros((self.num_of_seg, len(self.t_arr)))

        self.coil_data = {'N': 30,                          # Number of loops in coil
                          'r': 0.02,                        # Coil radius in m
                          'u': 4 * m.pi * 10 ** -7,         # Permeability constant, N/A**2
                          'cpd': 0.01}                      # Distance from the coil to the plane of the E-field in m.

        self.rlc_circuit_data_under = {'type': 'under',         # Type of stimuli (under-dampened)
                                       'tau': 0.4,              # Time constant
                                       'v0': 30,                # Initial charge of capacitor in V
                                       'R': 0.09,               # Resistance in ohm
                                       'C': 200 * 10 ** -6,     # Conductance in F
                                       'L': 13 * 10 ** -6}      # Inductance in H

        self.rlc_circuit_data_over = {'type': 'over',           # Type of stimuli (over-dampened)
                                      'tau': 0.4,               # Time constant
                                      'v0': 30,                 # Initial charge of capacitor in V
                                      'R': 0.09,                # ohm
                                      'C': 200 * 10 ** -6,      # F
                                      'L': 13 * 10 ** -6        # H
                                      }

    def _calc_vec(self):
        for i in range(self.num_of_seg):
            self.dir_vec[i] = self.seg_coord[i + 1] - self.seg_coord[i]
            self.l[i] = m.sqrt(self.dir_vec[i, 0]**2 + self.dir_vec[i, 1]**2)
            self.center_coord[i] = self.seg_coord[i] + self.dir_vec[i]*0.5
            if i > 0:
                self.center_vec[i - 1] = self.center_coord[i] - self.center_coord[i - 1]
                self.l_c[i - 1] = m.sqrt(self.center_vec[i - 1, 0] ** 2 + self.center_vec[i - 1, 1] ** 2)
                self.center_u_vec[i - 1] = self.center_vec[i - 1] / self.l_c[i - 1]

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
        return -1*(u*N)/(m.pi*m.sqrt(m_kk))*m.sqrt(r/h)*(scis.ellipk(m_kk)*(1 - 0.5*m_kk)
                                                         - scis.ellipe(m_kk))*-1*cos_theta

    def _calc_E_spat(self, coordinates):
        N = self.coil_data['N']
        r = self.coil_data['r']
        u = self.coil_data['u']
        cpd = self.coil_data['cpd']

        for coord, i in zip(coordinates, range(len(coordinates))):
            dist_xy = m.sqrt(coord[0]**2 + coord[1]**2)
            dist_tot = m.sqrt(dist_xy**2 + cpd**2)
            cos_theta = dist_xy / dist_tot
            E_spat = self._calc_point_in_E_spat(N, r, u, cpd, dist_tot, cos_theta)
            cos_alpha = coord[0] / dist_xy
            sin_alpha = coord[1] / dist_xy
            self.Ex_spat[i] = E_spat * cos_alpha
            self.Ey_spat[i] = E_spat * sin_alpha

    def _calc_delta_i_am(self, Ex, Ey, seg):
        return -1*(self.d[seg]/(4.*self.Ra*self.l_c[seg]**2))*((Ex[seg + 1] - Ex[seg])*self.center_u_vec[seg, 0]
                                                               + (Ey[seg + 1] - Ey[seg])*self.center_u_vec[seg, 1])

    def _calc_I_a(self, Ex, Ey, seg):
        return self.d[seg]/(4.*self.Ra*self.l_c[seg])*((Ex[seg + 1] + Ex[seg])*0.5*self.center_u_vec[seg, 0]
                                                        + (Ey[seg + 1] + Ey[seg])*0.5*self.center_u_vec[seg, 1])

    def _calc_delta_V(self, V_pre, V_curr, V_post, delta_t, I_a, seg):
        return ((self.Em - V_curr)/self.Rm
                + (self.d[seg]/(4*self.Ra))*((V_pre - V_curr)/self.l[seg]**2 + (V_post - V_curr)/self.l[seg]**2)
                - I_a)*(delta_t/self.Cm)

    def _calc_delta_V_start(self, V_curr, V_post, delta_t, I_a, seg):
        return ((self.Em - V_curr)/self.Rm
                + (self.d[seg]/(4*self.Ra))*(V_post - V_curr)/self.l[seg]**2
                - I_a)*(delta_t/self.Cm)

    def _calc_delta_V_end(self, V_curr, V_pre, delta_t, I_a, seg):
        return ((self.Em - V_curr)/self.Rm
                + (self.d[seg]/(4*self.Ra))*(V_pre - V_curr)/self.l[seg]**2
                - I_a)*(delta_t/self.Cm)

    def _calc_input_current(self):
        for i in range(len(self.t_arr)):
            Ex = self.Ex_spat * self.i_E_temp[i]
            Ey = self.Ey_spat * self.i_E_temp[i]
            for seg_idx in range(self.num_of_seg-1):
                I_a = self._calc_I_a(Ex, Ey, seg_idx)
                self.I_a[seg_idx, i] += I_a
                self.I_a[seg_idx+1, i] += -I_a

    def simulate(self):
        self._calc_vec()
        self._calc_E_temp()
        coordinates = self.center_coord
        self._calc_E_spat(coordinates)
        self._calc_input_current()
        delta_t = (self.t_arr[1] - self.t_arr[0])*10**-3
        self.V[:, 0] = np.ones(self.num_of_seg)*self.Em

        for i in range(len(self.t_arr)-1):
            for seg_idx in range(self.num_of_seg):
                if seg_idx == 0:
                    self.V[seg_idx, i + 1] = self.V[seg_idx, i] + self._calc_delta_V_start(self.V[seg_idx, i],
                                                                                           self.V[[seg_idx+1], i],
                                                                                           delta_t,
                                                                                           self.I_a[seg_idx, i],
                                                                                           seg_idx)
                elif seg_idx == self.num_of_seg - 1:
                    self.V[seg_idx, i + 1] = self.V[seg_idx, i] + self._calc_delta_V_end(self.V[seg_idx, i],
                                                                                         self.V[[seg_idx-1], i],
                                                                                         delta_t,
                                                                                         self.I_a[seg_idx, i],
                                                                                         seg_idx)
                else:
                    self.V[seg_idx, i + 1] = self.V[seg_idx, i] + self._calc_delta_V(self.V[seg_idx-1, i],
                                                                                     self.V[seg_idx, i],
                                                                                     self.V[seg_idx+1, i],
                                                                                     delta_t,
                                                                                     self.I_a[seg_idx, i],
                                                                                     seg_idx)


def plot_voltage(v_array, t_array):
    for i in range(len(v_array[:, 0])):
        plt.plot(t_array, v_array[i], label=str(i+1))
    plt.xlabel('Time in ms')
    plt.ylabel('Potential in mV')
    plt.xlim(0, 2)
    plt.legend(loc='upper right', title='Comp')
    plt.savefig(join("multicompartmental_model.png"))


if __name__ == "__main__":
    model = MultiCompartmentNeuron(time_array)
    model.simulate()
    plot_voltage(model.V*1000, time_array)