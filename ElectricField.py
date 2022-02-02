import math as m
import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt


# Scale
radius = 40000*10**-6                       # Scale from the center of the coil in m
spatial_resolution = 100*10**-6             # Spatial resolution in m
time = 10                                   # milli sec
timestep = 0.01                             # milli sec
timescale = 10**-3                          # Time is in milli s scale
time_array = np.arange(0, time + timestep, timestep)*timescale

# Two-compartment-neuron position in relation to center of coil in xy coordinates (in m) [[start], [mid], [stop]]
compartment_coords = np.array([[10000, 10000],
                               [10100, 10000],
                               [10200, 10000]])*10**-6

# Coil data
coil_data = {'N': 30,                       # Number of loops in coil
             'r': 0.02,                     # Coil radius, m
             'u': 4*m.pi*10**-7,            # Permeability constant, H/m (A/m**2)
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


# Compartmental neuron model, anchor = top left position of the center of the neuron model in the E-field
class TwoCompartmentNeuron:
    def __init__(self, coord_start_mid_stop, coil_dict, d=1*10**-6,
                 Em=-70*10**-3, Cm=1*10**-6, Rm=10000*10-2, Ra=100*10-2):
        self.compartments = 2
        self.V = []
        self.Em = Em
        self.Cm = Cm
        self.Rm = Rm
        self.Ra = Ra
        self.d = d
        self.coil_dict = coil_dict
        self.coord = coord_start_mid_stop
        self.dir_unit_vec = np.zeros((2, 2))

        for i in range(self.compartments):
            vec = self.coord[i+1] - self.coord[i]
            vec_length = m.sqrt(vec[0]**2 + vec[1]**2)
            self.dir_unit_vec[i, :] = vec/vec_length
        self.l = vec_length

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
        Ex_spat = np.zeros(3)
        Ey_spat = np.zeros(3)

        for coord, i in zip(self.coord, range(3)):
            dist_xy = m.sqrt(coord[0]**2 + coord[1]**2)
            dist_tot = m.sqrt(dist_xy**2 + cpd**2)
            cos_theta = dist_xy/dist_tot
            E_spat = self.calc_point_in_E_spat(N, r, u, cpd, dist_tot, cos_theta)
            cos_alpha = coord[0]/dist_xy
            sin_alpha = coord[1]/dist_xy
            Ex_spat[i] = E_spat*cos_alpha
            Ey_spat[i] = E_spat*sin_alpha
        return Ex_spat, Ey_spat

    def calc_I_am(self, Ex, Ey, comp):
        return (self.d/(4*self.Ra*self.l**2))*((Ex[1] - Ex[0])*self.dir_unit_vec[comp, 0]
                                                + (Ey[1] - Ey[0])*self.dir_unit_vec[comp, 1])

    def calc_delta_V1(self, V1, V2, delta_t, I_a):
        return ((self.Em - V1)/self.Rm - (self.d/(4*self.Ra))*(V1 - V2)/self.l**2 + I_a)*(delta_t/self.Cm)

    def calc_delta_V2(self, V1, V2, delta_t, I_a):
        return ((self.Em - V2)/self.Rm + (self.d/(4*self.Ra))*(V1 - V2)/self.l**2 + I_a)*(delta_t/self.Cm)

    def calc_I_a(self, Ex, Ey, comp):
        pass

    def simulate(self, time_arr, E_I_temp):
        delta_t = time_arr[1] - time_arr[0]
        self.V = np.zeros((self.compartments, time_arr.shape[0]))
        self.V[0, 0] = self.Em
        self.V[1, 0] = self.Em
        I_am_1 = np.zeros_like(time_arr)
        I_am_2 = np.zeros_like(time_arr)
        Ex_spat, Ey_spat = self.calc_E_spat()

        for i in range(len(time_arr)-1):
            Ex_full = Ex_spat*E_I_temp[i]
            Ey_full = Ey_spat*E_I_temp[i]

            I_am_1[i+1] = self.calc_I_am(Ex_full[:2], Ey_full[:2], 0)
            delta_I_am_1 = I_am_1[i+1] - I_am_1[i]
            self.V[0, i+1] = self.V[0, i] + self.calc_delta_V1(self.V[0, i], self.V[1, i], delta_t, delta_I_am_1)

            I_am_2[i+1] = self.calc_I_am(Ex_full[1:], Ey_full[1:], 1)
            delta_I_am_2 = I_am_2[i+1] - I_am_2[i]
            self.V[1, i+1] = self.V[1, i] + self.calc_delta_V2(self.V[0, i], self.V[1, i], delta_t, delta_I_am_2)


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


# Create 2D heatmap
def plot_heatmap(f):
    plt.imshow(f, cmap='jet', extent=[-4, 4, -4, 4])
    plt.colorbar()
    plt.title('E')
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')
    plt.show()


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
    return V0*C*w2*m.exp(-w1*t)*((w1/w2)**2 + 1)*m.sin(w2*t)


# Calculate I for OVERdamped RLC circuit
def calc_I_over(V0, C, w1, w2, t):
    return V0*C*w2*m.exp(-w1*t)*((w1/w2)**2 - 1)*m.sinh(w2*t)


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
            I_deriv_array[i-1] = (I_array[i] - I_array[i-1]) / (t_array[i] - t_array[i-1])

    elif stimuli_type == 'over':
        w1, w2 = calc_omega_over(rlc_dict)
        I_array[0] = calc_I_over(V0, C, w1, w2, t_array[0])
        for i in range(1, len(t_array) - 1):
            I_array[i] = calc_I_over(V0, C, w1, w2, t_array[i])
            I_deriv_array[i - 1] = (I_array[i] - I_array[i - 1]) / (t_array[i] - t_array[i - 1])
    else:
        print('Wrong RLC-circuit type, should be "over" or "under".')
    return I_array[:-2], I_deriv_array[:-2]


def plot_simple(data):
    plt.plot(data)
    plt.show()


def plot_simple_2(data1, data2):
    plt.plot(data1)
    plt.plot(data2)
    plt.show()


if __name__ == "__main__":
#    ele_field, ele_field_x, ele_field_y, x, y = spatial_ele_field(coil_data, radius, spatial_resolution)
#    plot_heatmap(ele_field)
#    plot_heatmap(ele_field_x)
#    plot_heatmap(ele_field_y)

    I, I_deriv = temporal_ele_field_rlc(rlc_circuit_data_under, time_array)
#    I, I_deriv = temporal_ele_field_rlc(rlc_circuit_data_over, time_array)
#    plot_simple(I)
#    plot_simple(I_deriv)

    model = TwoCompartmentNeuron(compartment_coords, coil_data)
    model.simulate(time_array, I_deriv)
    plot_simple_2(model.V[0, :], model.V[1, :])
