from os.path import join
import math as m
import numpy as np
import neuron
import LFPy
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from magnetic_field_new import MagneticField


time_start = 0
ts = 0.01
time_stop = 5
number_of_ts = int((time_stop - time_start) / ts)
time_array = np.linspace(time_start, time_stop, number_of_ts)

cell_parameters = {
        'morphology': '3comp_axon.hoc',
        'v_init': -70,
        'nsegs_method': None,
        "lambda_f": 100.,
        "tstart": time_start,
        "tstop": time_stop,
        "dt": ts,
        "Ra": 150,
        "cm": 1,
        "passive": True,
        "passive_parameters": {"g_pas": 1. / 30000,
                               "e_pas": -70},
}
cell = LFPy.Cell(**cell_parameters)
cell.set_pos(x=-2 * 1e4, y=1 * 1e4, z=-1 * 1e4)

MagField = MagneticField(time_array, cell=cell)
MagField.make_input_currents()

v_mem = MagField.simulate_cell()
MagField.plot_vmem(v_mem, "manual_v_mem.png")

input_vec, syn = MagField.insert_im_neuron()
cell.simulate(rec_vmem=True, rec_imem=True)
MagField.plot_vmem(cell.vmem[:, :-1], "neuron_v_mem.png")

# MagField.plot_I_coil()
# print('Maximum magnetic field magnitude =', MagField.magnetic_field_magnitude, 'Tesla in the center of the coil.')
# MagField.plot_E_t()
# MagField.plot_E_s_heatmaps()
