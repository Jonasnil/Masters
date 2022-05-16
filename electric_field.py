from os.path import join
import math as m
import numpy as np
import scipy.special as ss
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt


# Scale
radius = 40000*10**-6                       # Scale from the center of the coil in m
spatial_resolution = 100*10**-6             # Spatial resolution in m
time = 10                                   # milli sec
timestep = 0.001                            # milli sec
time_array = np.arange(0, time + timestep, timestep)

# Two-compartment-neuron position in relation to center of coil in xy coordinates (in m) [[start], [mid], [stop]]
compartment_coords = np.array([[10000, 10000],
                               [10100, 10000],
                               [10200, 10000]])*10**-6

# Coil data
coil_data = {'N': 30,                       # Number of loops in coil
             'r': 0.02,                     # Coil radius, m
             'u': 4*m.pi*10**-7,            # Permeability constant, H/m (N/A**2)
             'cpd': 0.01                    # Distance from the plane of the coil to the plane of the ele field in m
             }

# RLC circuit data
rlc_circuit_data_under = {'type': 'under',  # Type of stimuli (over/under-dampened)
                          'tau': 0.4,       # Time constant
                          'v0': 30,         # Initial charge of capacitor in V
                          'R': 0.09,        # Resistance in ohm
                          'C': 200*10**-6,  # Conductance in F
                          'L': 13*10**-6    # Inductance in H
                          }

rlc_circuit_data_over = {'type': 'over',    # Type of stimuli (over/under-dampened)
                         'tau': 0.4,        # Time constant
                         'v0': 30,          # Initial charge of capacitor in V
                         'R': 0.09,         # ohm
                         'C': 200*10**-6,   # F
                         'L': 13*10**-6     # H
                         }


# Compartmental neuron model
class TwoCompartmentNeuron:
    def __init__(self, coord_start_mid_stop, coil_dict, d=1*10**-6, Em=-70*10**-3, Cm=1*10**-6,
                 Rm=30000*10-4, Ra=100*10-2, I_calc_method='i_am'):
        self.compartments = 2
        self.V = []
        self.Em = Em
        self.Cm = Cm
        self.Rm = Rm
        self.Ra = Ra
        self.d = d
        self.I_calc_method = I_calc_method
        self.coil_dict = coil_dict
        self.coord = coord_start_mid_stop
        self.dir_vec = np.zeros((self.compartments, 2))
        self.dir_unit_vec = np.zeros((self.compartments, 2))
        self.l = np.zeros((self.compartments, 1))
        self.center_coord = np.zeros((self.compartments, 2))
        self.center_dir_vec = np.zeros((self.compartments-1, 2))
        self.center_dir_unit_vec = np.zeros((self.compartments-1, 2))
        self.l_c = np.zeros((self.compartments-1, 1))

    def calc_vectors(self):
        for i in range(self.compartments):
            self.dir_vec[i] = self.coord[i+1] - self.coord[i]
            self.l[i] = m.sqrt(self.dir_vec[i, 0]**2 + self.dir_vec[i, 1]**2)
            self.dir_unit_vec[i, :] = self.dir_vec[i]/self.l[i]
            self.center_coord[i, :] = self.coord[i] + self.dir_vec[i]/2.
            if i > 0:
                self.center_dir_vec[i-1] = self.center_coord[i] - self.center_coord[i-1]
                self.l_c[i-1] = m.sqrt(self.center_dir_vec[i-1, 0] ** 2 + self.center_dir_vec[i-1, 1] ** 2)
                self.center_dir_unit_vec[i-1, :] = self.center_dir_vec[i-1]/self.l_c[i-1]

    def calc_m_kk(self, r, z, h):
        return (4*r*h)/((r + h)**2 + z**2)

    def calc_point_in_E_spat(self, N, r, u, z, h, cos_theta):
        m_kk = self.calc_m_kk(r, z, h)
        return -1*(u*N)/(m.pi*m.sqrt(m_kk))*m.sqrt(r/h)*(ss.ellipk(m_kk)*(1 - (1/2)*m_kk)
                                                         - ss.ellipe(m_kk))*-1*cos_theta

    def calc_E_spat(self):
        N = self.coil_dict['N']
        r = self.coil_dict['r']
        u = self.coil_dict['u']
        cpd = self.coil_dict['cpd']

        if self.I_calc_method == 'i_am' or self.I_calc_method == 'I_a':
            coord_to_use = self.center_coord
            i_range = range(self.compartments)
            Ex_spat = np.zeros(self.compartments)
            Ey_spat = np.zeros(self.compartments)
        elif self.I_calc_method == 'i_am_int':
            coord_to_use = self.coord
            i_range = range(self.compartments+1)
            Ex_spat = np.zeros(self.compartments+1)
            Ey_spat = np.zeros(self.compartments+1)
        else:
            print('Wrong I_calc_method, must be "I_a", "i_am" or "i_am_int".')

        for coord, i in zip(coord_to_use, i_range):
            dist_xy = m.sqrt(coord[0]**2 + coord[1]**2)
            dist_tot = m.sqrt(dist_xy**2 + cpd**2)
            cos_theta = dist_xy/dist_tot
            E_spat = self.calc_point_in_E_spat(N, r, u, cpd, dist_tot, cos_theta)
            cos_alpha = coord[0]/dist_xy
            sin_alpha = coord[1]/dist_xy
            Ex_spat[i] = E_spat*cos_alpha
            Ey_spat[i] = E_spat*sin_alpha
        return Ex_spat, Ey_spat

    def calc_delta_i_am(self, Ex, Ey, pos):
        return (self.d/(4.*self.Ra*self.l_c[pos]**2))*((Ex[1] - Ex[0])*self.center_dir_unit_vec[pos, 0]
                                                       + (Ey[1] - Ey[0])*self.center_dir_unit_vec[pos, 1])

    def calc_I_a(self, Ex, Ey, pos):
        return (self.d/(4.*self.Ra*self.l_c[pos]))*((Ex[1] + Ex[0])/2.*self.center_dir_unit_vec[pos, 0]
                                                    + (Ey[1] + Ey[0])/2.*self.center_dir_unit_vec[pos, 1])

    def calc_delta_i_am_int(self, Ex, Ey, pos):
        return (self.d/(4.*self.Ra*self.l[pos]**2))*((Ex[1] - Ex[0])*self.dir_unit_vec[pos, 0]
                                                     + (Ey[1] - Ey[0])*self.dir_unit_vec[pos, 1])

    def calc_delta_V(self, V0, V_o, delta_t, I_a, pos):
        return ((self.Em - V0)/self.Rm - (self.d/(4*self.Ra*self.l[pos]**2))*(V0 - V_o) - I_a)*(delta_t/self.Cm)

    def simulate(self, time_arr, E_I_temp):
        delta_t = (time_arr[1] - time_arr[0])*10**-3
        self.V = np.zeros((self.compartments, time_arr.shape[0]))
        self.calc_vectors()
        self.V[0, 0] = self.Em
        self.V[1, 0] = self.Em
        Ex_spat, Ey_spat = self.calc_E_spat()

        if self.I_calc_method == 'i_am':
            delta_I_am = np.zeros_like(time_arr)
            for i in range(len(time_arr)-1):
                Ex_full = Ex_spat*E_I_temp[i]
                Ey_full = Ey_spat*E_I_temp[i]
                delta_I_am[i+1] = self.calc_delta_i_am(Ex_full, Ey_full, 0)
                self.V[0, i+1] = self.V[0, i] + self.calc_delta_V(self.V[0, i], self.V[1, i],
                                                                  delta_t, delta_I_am[i+1], 0)
                self.V[1, i+1] = self.V[1, i] + self.calc_delta_V(self.V[1, i], self.V[0, i],
                                                                  delta_t, -1*delta_I_am[i+1], 0)

        elif self.I_calc_method == 'I_a':
            I_a = np.zeros_like(time_arr)
            for i in range(len(time_arr)-1):
                Ex_full = Ex_spat*E_I_temp[i]
                Ey_full = Ey_spat*E_I_temp[i]
                I_a[i+1] = self.calc_I_a(Ex_full, Ey_full, 0)
                self.V[0, i+1] = self.V[0, i] + self.calc_delta_V(self.V[0, i], self.V[1, i],
                                                                  delta_t, I_a[i+1], 0)
                self.V[1, i+1] = self.V[1, i] + self.calc_delta_V(self.V[1, i], self.V[0, i],
                                                                  delta_t, -1*I_a[i+1], 0)

        elif self.I_calc_method == 'i_am_int':
            delta_I_am_1 = np.zeros_like(time_arr)
            delta_I_am_2 = np.zeros_like(time_arr)
            for i in range(len(time_arr)-1):
                Ex_full = Ex_spat*E_I_temp[i]
                Ey_full = Ey_spat*E_I_temp[i]
                delta_I_am_1[i + 1] = self.calc_delta_i_am_int(Ex_full[:-1], Ey_full[:-1], 0)
                delta_I_am_2[i + 1] = self.calc_delta_i_am_int(Ex_full[1:], Ey_full[1:], 1)
                self.V[0, i + 1] = self.V[0, i] + self.calc_delta_V(self.V[0, i], self.V[1, i],
                                                                    delta_t, delta_I_am_1[i + 1], 0)
                self.V[1, i + 1] = self.V[1, i] + self.calc_delta_V(self.V[1, i], self.V[0, i],
                                                                    delta_t, -1*delta_I_am_2[i + 1], 0)
        else:
            print('Wrong I_calc_method, must be "I_a", "i_am" or "i_am_int".')


# Calculate m = k**2
def calc_m_kk_at_xy(r, z, h):
    return (4*r*h)/((r + h)**2 + z**2)


# Calculate the spatial electrical field
def calc_point_in_spat_ele_field(N, r, u, z, h, cos_theta):
    m_kk = calc_m_kk_at_xy(r, z, h)
    return -1*(u*N)/(m.pi*m.sqrt(m_kk))*m.sqrt(r/h)*(ss.ellipk(m_kk)*(1-(1/2)*m_kk)
                                                     - ss.ellipe(m_kk))*-1*cos_theta


# Create the spatial electrical field
def spatial_ele_field(coil_dict, rad_scale, spat_res):
    N = coil_dict['N']
    r = coil_dict['r']
    u = coil_dict['u']
    cpd = coil_dict['cpd']
    one_dim_xy_pos = np.arange(-rad_scale, rad_scale, spat_res)
    rad_points = int(rad_scale/spat_res)
    xy_points = rad_points*2
    spat_ele_field = np.zeros((xy_points, xy_points))
    spat_ele_field_x = np.zeros((xy_points, xy_points))
    spat_ele_field_y = np.zeros((xy_points, xy_points))
    x_coord = np.zeros((xy_points, xy_points))
    y_coord = np.zeros((xy_points, xy_points))

    for i in range(-rad_points, rad_points):
        for j in range(-rad_points, rad_points):
            x_coord[i, j], y_coord[i, j] = one_dim_xy_pos[i], one_dim_xy_pos[j]
            dist_xy = m.sqrt(x_coord[i, j]**2 + y_coord[i, j]**2)
            dist_tot = m.sqrt(dist_xy**2 + cpd**2)

            cos_theta = dist_xy/dist_tot
            spat_ele_field[i, j] = calc_point_in_spat_ele_field(N, r, u, cpd, dist_tot, cos_theta)

            cos_alpha = x_coord[i, j]/dist_xy
            sin_alpha = y_coord[i, j]/dist_xy
            spat_ele_field_x[i, j] = spat_ele_field[i, j]*cos_alpha
            spat_ele_field_y[i, j] = spat_ele_field[i, j]*sin_alpha
    return spat_ele_field, spat_ele_field_x, spat_ele_field_y, x_coord, y_coord


# Calculate the spatial electrical field
def calc_point_in_spat_ele_field_alt(N, r, u, z, h):
    m_kk = calc_m_kk_at_xy(r, z, h)
    return -1*(u*N)/(m.pi*m.sqrt(m_kk))*m.sqrt(r/h)*(ss.ellipk(m_kk)*(1-0.5*m_kk)
                                                     - ss.ellipe(m_kk))


# Create the spatial electrical field
def spatial_ele_field_alt(coil_dict, rad_scale, spat_res):
    N = coil_dict['N']
    r = coil_dict['r']
    u = coil_dict['u']
    cpd = coil_dict['cpd']
    one_dim_xy_pos = np.arange(-rad_scale, rad_scale, spat_res)
    rad_points = int(rad_scale/spat_res)
    xy_points = rad_points*2
    spat_ele_field = np.zeros((xy_points, xy_points))
    spat_ele_field_x = np.zeros((xy_points, xy_points))
    spat_ele_field_y = np.zeros((xy_points, xy_points))
    x_coord = np.zeros((xy_points, xy_points))
    y_coord = np.zeros((xy_points, xy_points))

    for i in range(-rad_points, rad_points):
        for j in range(-rad_points, rad_points):
            x_coord[i, j], y_coord[i, j] = one_dim_xy_pos[i], one_dim_xy_pos[j]
            dist_xy = m.sqrt(x_coord[i, j]**2 + y_coord[i, j]**2)
            dist_tot = m.sqrt(dist_xy**2 + cpd**2)
            theta = m.atan2(y_coord[i, j], x_coord[i, j])

            E_value = calc_point_in_spat_ele_field_alt(N, r, u, cpd, dist_tot)

            spat_ele_field_x[i, j] = E_value*m.cos(theta)
            spat_ele_field_y[i, j] = E_value*m.sin(theta)
            spat_ele_field[i, j] = E_value
    return spat_ele_field, spat_ele_field_x, spat_ele_field_y, x_coord, y_coord


# Create 2D heatmap
def plot_heatmap(f, filename):
    plt.imshow(f, cmap='jet', extent=[-4, 4, -4, 4])
    plt.colorbar()
    plt.title('E')
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')
    plt.savefig(join(filename))
    plt.clf()


# Calculate omega 1 and 2 for UNDERdamped case
def calc_omega_under(rlc_dict_under):
    R_u = rlc_dict_under['R']
    C_u = rlc_dict_under['C']
    L_u = rlc_dict_under['L']
    w1_under = R_u/(2.*L_u)
    w2_under = m.sqrt((1./(L_u*C_u)) - w1_under**2)
    return w1_under, w2_under


# Calculate omega 1 and 2 for OVERdamped case
def calc_omega_over(rlc_dict_over):
    R_o = rlc_dict_over['R']
    C_o = rlc_dict_over['C']
    L_o = rlc_dict_over['L']
    w1_over = R_o/(2.*L_o)
    w2_over = m.sqrt(w1_over**2 - 1./(L_o*C_o))
    return w1_over, w2_over


# Calculate I for UNDERdamped RLC circuit
def calc_I_under(V0, C, w1, w2, t):
    return V0*C*w2*m.exp(-w1*t*10**-3)*((w1/w2)**2 + 1)*m.sin(w2*t*10**-3)


# Calculate I for OVERdamped RLC circuit
def calc_I_over(V0, C, w1, w2, t):
    return V0*C*w2*m.exp(-w1*t*10**-3)*((w1/w2)**2 - 1)*m.sinh(w2*t*10**-3)


# Calculate the temporal electrical field
def temporal_ele_field_rlc(rlc_dict, t_array_in):
    t_array = np.append(t_array_in, [t_array_in[-1] + t_array_in[-1] - t_array_in[-2]])
    I_array = np.zeros_like(t_array)
    I_deriv_array = np.zeros_like(t_array)
    stimuli_type = rlc_dict['type']
    V0 = rlc_dict['v0']
    C = rlc_dict['C']

    if stimuli_type == 'under':
        w1, w2 = calc_omega_under(rlc_dict)
        I_array[0] = calc_I_under(V0, C, w1, w2, t_array[0])
        for i in range(1, len(t_array)-1):
            I_array[i] = calc_I_under(V0, C, w1, w2, t_array[i])
            I_deriv_array[i-1] = (I_array[i] - I_array[i-1]) / ((t_array[i] - t_array[i-1])*10**-3)

    elif stimuli_type == 'over':
        w1, w2 = calc_omega_over(rlc_dict)
        I_array[0] = calc_I_over(V0, C, w1, w2, t_array[0])
        for i in range(1, len(t_array) - 1):
            I_array[i] = calc_I_over(V0, C, w1, w2, t_array[i])
            I_deriv_array[i-1] = (I_array[i] - I_array[i-1]) / ((t_array[i] - t_array[i-1])*10**-3)
    else:
        print('Wrong RLC-circuit type, must be "over" or "under".')
    return I_array[:-2], I_deriv_array[:-2]


def plot_simple(data, x_axis):
    plt.plot(x_axis, data)



def plot_voltage(v_array, t_array):
    for i in range(len(v_array[:, 0])):
        plt.plot(t_array, v_array[i], label=str(i+1))
    plt.xlabel('Time in ms')
    plt.ylabel('Potential in mV')
    plt.legend(title='Comp')
    plt.savefig(join("ele_field_testing.png"))


if __name__ == "__main__":
    ele_field, ele_field_x, ele_field_y, x, y = spatial_ele_field(coil_data, radius, spatial_resolution)
    plot_heatmap(ele_field, "heatmap_E_tot.png")
    plot_heatmap(ele_field_x, "heatmap_x.png")
    plot_heatmap(ele_field_y, "heatmap_y.png")

#    I, I_deriv = temporal_ele_field_rlc(rlc_circuit_data_under, time_array)
#    I, I_deriv = temporal_ele_field_rlc(rlc_circuit_data_over, time_array)
#    plot_simple(I, time_array[:-1])
#    plot_simple(I_deriv, time_array[:-1])

#    model = TwoCompartmentNeuron(compartment_coords, coil_data)
#    model.simulate(time_array, I_deriv)
#    plot_voltage(model.V*1000, time_array)
