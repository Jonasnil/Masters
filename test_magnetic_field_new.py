import os
import sys
from os.path import join
import math as m
import numpy as np
import neuron
from neuron import h
import LFPy
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from magnetic_field_new import MagneticField


cell_models_folder = os.path.abspath('.')
hay_folder = join(cell_models_folder, "L5bPCmodelsEH")


def download_hay_model():

    print("Downloading Hay model")
    if sys.version < '3':
        from urllib2 import urlopen
    else:
        from urllib.request import urlopen
    import ssl
    from warnings import warn
    import zipfile
    #get the model files:
    u = urlopen('http://senselab.med.yale.edu/ModelDB/eavBinDown.asp?o=139653&a=23&mime=application/zip',
                context=ssl._create_unverified_context())
    localFile = open(join(cell_models_folder, 'L5bPCmodelsEH.zip'), 'wb')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile(join(cell_models_folder, 'L5bPCmodelsEH.zip'), 'r')
    myzip.extractall(cell_models_folder)
    myzip.close()

    #compile mod files every time, because of incompatibility with Mainen96 files:
    mod_pth = join(hay_folder, "mod/")

    if "win32" in sys.platform:
        warn("no autompile of NMODL (.mod) files on Windows.\n"
             + "Run mknrndll from NEURON bash in the folder "
               "L5bPCmodelsEH/mod and rerun example script")
        if not mod_pth in neuron.nrn_dll_loaded:
            neuron.h.nrn_load_dll(join(mod_pth, "nrnmech.dll"))
        neuron.nrn_dll_loaded.append(mod_pth)
    else:
        os.system('''
                  cd {}
                  nrnivmodl
                  '''.format(mod_pth))
        neuron.load_mechanisms(mod_pth)


def return_hay_cell(tstop, dt, make_passive=False):
    if not os.path.isfile(join(hay_folder, 'morphologies', 'cell1.asc')):
        download_hay_model()

    if make_passive:
        cell_params = {
            'morphology': join(hay_folder, 'morphologies', 'cell1.asc'),
            'passive': True,
            'passive_parameters': {"g_pas": 1 / 30000,
                                   "e_pas": -70.},
            'nsegs_method': "lambda_f",
            "Ra": 150,
            "cm": 1.0,
            "lambda_f": 100,
            'dt': dt,
            'tstart': -1,
            'tstop': tstop,
            'v_init': -70,
            'pt3d': True,
        }

        cell = LFPy.Cell(**cell_params)
        cell.set_rotation(x=4.729, y=-3.166)

        return cell
    else:
        if not hasattr(neuron.h, "CaDynamics_E2"):
            neuron.load_mechanisms(join(hay_folder, 'mod'))
        cell_params = {
            'morphology': join(hay_folder, "morphologies", "cell1.asc"),
            'templatefile': [join(hay_folder, 'models', 'L5PCbiophys3.hoc'),
                             join(hay_folder, 'models', 'L5PCtemplate.hoc')],
            'templatename': 'L5PCtemplate',
            'templateargs': join(hay_folder, 'morphologies', 'cell1.asc'),
            'passive': False,
            'nsegs_method': None,
            'dt': dt,
            'tstart': -200,
            'tstop': tstop,
            'v_init': -75,
            'celsius': 34,
            'pt3d': True,
        }

        cell = LFPy.TemplateCell(**cell_params)

        cell.set_rotation(x=4.729, y=-3.166)
        cell.set_rotation(x=-np.pi/2)
        cell.set_pos(x=-17000, y=17000, z=0.5 * 1e4)
        return cell

def axon_with_varying_diam(diam_range):
    v_mem = []
    hoc_file = 'shape_to_file.hoc'
    for d in diam_range:
        with open(hoc_file, "w") as hoc_edit:
            hoc_edit.write('create axon[1]\n')
            hoc_edit.write('axon[0] {\n')
            hoc_edit.write('    nseg = 500\n')
            hoc_edit.write('    pt3dadd(0, 0, 0, ' + str(d) + ')\n')
            hoc_edit.write('    pt3dadd(160000, 0, 0, ' + str(d) + ')\n')
            hoc_edit.write('}')

        cell = create_LFPy_cell(hoc_file)
        cell.set_pos(x=-79840, y=20000, z=-10000)

        rlc_type = 'over'
        MagField = MagneticField(time_array, cell=cell, rlc_type=rlc_type)
        MagField.make_input_currents(multi_sec=True)

        input_vec, syn = MagField.insert_im_neuron()
        cell.simulate(rec_vmem=True, rec_imem=True)

        v_mem.append(np.max(cell.vmem))

    MagField.plot_neuron_placement_multisec()
    MagField.plot_vmem_vs_d(diam_range, v_mem)


def return_stick_cell(dt, tstop):
    axon_length = 16e4
    nsegs = 500
    h("forall delete_section()")
    h("""
    celsius = 18.5
    proc celldef() {
      topol()
      subsets()
      geom()
      biophys()
      geom_nseg()
    }

    create axon[1]

    proc topol() { local i
      basic_shape()
    }
    proc basic_shape() {
      axon[0] {pt3dclear()
      pt3dadd(0, 0, 0, 100)
      pt3dadd(%d, 0, 0, 100)}
    }

    objref all
    proc subsets() { local i
      objref all
      all = new SectionList()
        axon[0] all.append()

    }
    proc geom() {
    }
    proc geom_nseg() {
    forall {nseg = %d}
    }
    proc biophys() {
    }
    celldef()

    //Ra = 35.4
    //cm = 1.
    //Rm = 30000.

  forall {
    Ra = 35.4
    cm = 1
    insert hh
    gnabar_hh = 0.12
    gkbar_hh = 0.036
    gl_hh = 0.0003
    el_hh = -54.387
    ena=50
    ek=-77
    //insert xtra
  }
//    forall {
//        //insert pas // 'pas' for passive, 'hh' for Hodgkin-Huxley
//        insert hh
//        //g_pas = 1 / Rm
//        }
    """ % (axon_length, nsegs))

    cell_params = {
        'morphology': h.all,
        'delete_sections': False,
        'v_init': -65.,
        'passive': False,
        'nsegs_method': None,
        'dt': dt,
        'tstart': 0.,
        'tstop': tstop,
    }
    cell = LFPy.Cell(**cell_params)
    cell.set_pos(x=-cell.x[int(cell.totnsegs / 2)].mean(), y=20000, z=-10000)
    return cell

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
ts = 2**-5
time_stop = 5
number_of_ts = int((time_stop - time_start) / ts)
time_array = np.linspace(time_start, time_stop, number_of_ts)

hoc_file = 'simple_axon.hoc'
varying_diam = False
HH = False
hay_cell = True

if not varying_diam and not HH and not hay_cell:
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
    v_mem_neuron = cell.vmem[:, :-1]
    MagField.plot_vmem(v_mem_neuron, "neuron_v_mem.png")

    MagField.plot_neuron_placement_multisec()

    # MagField.plot_I_coil()
    print('Maximum magnetic field magnitude =', MagField.magnetic_flux_density, 'Tesla in the center of the coil.')
    # MagField.plot_E_t()
    # MagField.plot_E_s_heatmaps()

elif varying_diam:
    diam_range = np.arange(1, 152, 5)
    axon_with_varying_diam(diam_range)

elif HH:
    cell = return_stick_cell(ts, time_stop)

    rlc_type = 'over'
    MagField = MagneticField(time_array, cell=cell, rlc_type=rlc_type)
    MagField.make_input_currents(multi_sec=True)

    input_vec, syn = MagField.insert_im_neuron()
    cell.simulate(rec_vmem=True, rec_imem=True)
    v_mem_neuron = cell.vmem[:, :-1]

    MagField.plot_hh_axon_heatmap(v_mem_neuron)

    MagField.plot_vmem(v_mem_neuron, "neuron_v_mem.png")

    MagField.plot_neuron_placement_multisec()

elif hay_cell:
    cell = return_hay_cell(time_stop, ts)

    rlc_type = 'over'
    MagField = MagneticField(time_array, cell=cell, rlc_type=rlc_type)
    MagField.make_input_currents(multi_sec=True)

    input_vec, syn = MagField.insert_im_neuron()
    cell.simulate(rec_vmem=True, rec_imem=True)

    MagField.plot_vmem(cell.vmem[:, :-1], "neuron_v_mem_hay.png")

    MagField.plot_vmem_hay(cell, "neuron_v_mem_and_pos_hay.png")
