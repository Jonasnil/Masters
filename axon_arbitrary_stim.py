from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import neuron
import LFPy
from magnetic_stimulation import MagStim

'''

# Make a simple passive axon cell model:
h = neuron.h
h("""
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
  pt3dadd(0, 0, 0, 1)
  pt3dadd(1000, 0, 0, 1)}
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
forall {nseg = 500}
}
proc biophys() {
}
celldef()

Ra = 100.
cm = 1.
Rm = 30000

forall {
    insert pas // 'pas' for passive, 'hh' for Hodgkin-Huxley
    g_pas=1/Rm
    }
""")

'''

# Make cell model in LFPy:
end_t = 20
dt = 2**-9
cell_parameters = {
        'morphology': 'simple_axon.hoc',
        'v_init': -70,
        'nsegs_method': None,
        "lambda_f": 500,
        "tstart": -dt,
        "tstop": end_t,
        "dt": dt,
        "pt3d": True,
        "extracellular": True,
        "passive": True,
}
cell = LFPy.Cell(**cell_parameters)
num_tsteps = int(end_t / dt + 1)
tvec = np.arange(num_tsteps) * dt

mag_calc_method = 'int_i'  # Must be 'ext_e' or 'int_i'
i = 0
syn = None
synlist = []
input_vecs = []

# Create simulated magnetic stimulation.
stim = MagStim(cell, tvec)

# Define input locations (segments).
input_idxs = np.array(range(stim.num_of_seg))

if mag_calc_method == 'int_i':
    input_current = stim.calc_input_current_multisection()  # Current in A.
    for sec in cell.allseclist:
        for seg in sec:
            # print("Input inserted in ", sec.name(), seg.x)

            # Arbitrary input current can be inserted here
            input_array = input_current[i] * 10**9  # current with units nA

            # Insert the current into the compartment
            noise_vec = neuron.h.Vector(input_array)
            syn = neuron.h.ISyn(sec(seg.x))
            syn.dur = 1e9
            syn.delay = 0
            noise_vec.play(syn._ref_amp, cell.dt)
            synlist.append(syn)
            input_vecs.append(noise_vec)
            i += 1

elif mag_calc_method == 'ext_e':
    ext_quasipot = stim.calc_ext_quasipot() * 10**-3  # Potential V/m -> mV/um
    cell.insert_v_ext(ext_quasipot, tvec)


cell.simulate(rec_vmem=True, rec_imem=True)


# Create 2D heatmap
# plt.imshow(cell.vmem, cmap='jet')
# plt.axis('auto')
# plt.colorbar()
# plt.title('Axon Membrane Potential')
# plt.xlabel('Compartment number')
# plt.ylabel('time [ms]')
# plt.savefig(join("axon_heatmap.png"))
# plt.clf()

# The rest is plotting
fig = plt.figure(figsize=[9, 6])
fig.subplots_adjust(wspace=0.5, top=0.9, bottom=0.1, hspace=0.6)
ax1 = fig.add_subplot(311, aspect=1, frameon=False, title="cell",
                      xlabel="x (µm)", ylabel="z (µm)",
                      ylim=[-200, 200], xlim=[-100, 1000])
ax2 = fig.add_subplot(312, xlabel="time (ms)",
                      ylabel="membrane\npotential (mV)",
                      xlim=[0, 2.5])
ax3 = fig.add_subplot(313, xlabel="time (ms)",
                      ylabel="membrane\ncurrent (nA)",
                      xlim=[0, 2.5])

[ax1.plot(cell.x[idx],  cell.z[idx], c='gray', lw=1)
 for idx in range(cell.totnsegs)]

plot_idxs = input_idxs[np.array([0, int(len(input_idxs)/2.), -1])]
plot_idx_clrs = {idx: plt.cm.viridis(num / (len(plot_idxs) - 1))
                 for num, idx in enumerate(plot_idxs)}

for idx in plot_idxs:
    ax1.plot(cell.x[idx].mean(), cell.z[idx].mean(), 'o', c=plot_idx_clrs[idx])
    ax2.plot(cell.tvec, cell.vmem[idx, :] - cell.vmem[idx, 0, None], c=plot_idx_clrs[idx], lw=2)
    ax3.plot(cell.tvec, cell.imem[idx, :], c=plot_idx_clrs[idx], lw=2)

if mag_calc_method == 'int_i':
    plt.savefig(join("axon_testing.png"))
elif mag_calc_method == 'ext_e':
    plt.savefig(join("axon_testing_ext.png"))
