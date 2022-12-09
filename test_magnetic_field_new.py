from os.path import join
import math as m
import numpy as np
import neuron
import LFPy
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from magnetic_field_new import MagneticField

def axon_with_varying_diam(diam_range):
    v_mem = []
    hoc_file = 'shape_to_file.hoc'
    for d in diam_range:
        with open(hoc_file, "w") as hoc_edit:
            hoc_edit.write('create axon[1]\n')
            hoc_edit.write('axon[0] {\n')
            hoc_edit.write('    nseg = 100\n')
            hoc_edit.write('    pt3dadd(0, 0, 0, ' + str(d) + ')\n')
            hoc_edit.write('    pt3dadd(10000, 0, 0, ' + str(d) + ')\n')
            hoc_edit.write('}')

        cell = create_LFPy_cell(hoc_file)
        cell.set_pos(x=-4950, y=21000, z=-10000)

        rlc_type = 'over'
        MagField = MagneticField(time_array, cell=cell, rlc_type=rlc_type)
        MagField.make_input_currents(multi_sec=True)

        input_vec, syn = MagField.insert_im_neuron()
        cell.simulate(rec_vmem=True, rec_imem=True)

        v_mem.append(np.max(cell.vmem))

    MagField.plot_neuron_placement()
    MagField.plot_vmem_vs_d(diam_range, v_mem)

def create_LFPy_cell(hoc_file):
    neuron.h("forall delete_section()")
    cell_parameters = {
            'morphology': hoc_file,
            'v_init': -70,
            'nsegs_method': None,
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
    return cell

time_start = 0
ts = 0.01
time_stop = 3
number_of_ts = int((time_stop - time_start) / ts)
time_array = np.linspace(time_start, time_stop, number_of_ts)

hoc_file = 'simple_axon.hoc'
varying_diam = True

if not varying_diam:
    cell = create_LFPy_cell(hoc_file)
    cell.set_pos(x=-450, y=20000, z=-10000)

    rlc_type = 'over'
    MagField = MagneticField(time_array, cell=cell, rlc_type=rlc_type)
    MagField.make_input_currents(multi_sec=True)

    # v_mem = MagField.simulate_cell()
    # MagField.plot_vmem(v_mem, "manual_v_mem.png")
    # MagField.plot_vmem_normalized(v_mem, "manual_v_mem_norm.png", rlc_type)

    input_vec, syn = MagField.insert_im_neuron()
    cell.simulate(rec_vmem=True, rec_imem=True)
    v_mem_neur = cell.vmem[:, :-1]
    MagField.plot_vmem(v_mem_neur, "neuron_v_mem.png")

    MagField.plot_neuron_placement_multisec()

    # MagField.plot_I_coil()
    print('Maximum magnetic field magnitude =', MagField.magnetic_flux_density, 'Tesla in the center of the coil.')
    # MagField.plot_E_t()
    # MagField.plot_E_s_heatmaps()

else:
    diam_range = np.arange(1, 102, 5)
    axon_with_varying_diam(diam_range)
