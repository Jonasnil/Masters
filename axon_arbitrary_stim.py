from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import neuron
import LFPy
from magnetic_stimulation import MagStim

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
forall {nseg = 20}
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

# Make cell model in LFPy:
end_t = 20
dt = 2**-5
cell_parameters = {
        'morphology': h.all,
        'v_init': -70,
        'nsegs_method': "lambda_f",
        "lambda_f": 500,
        "tstart": 0,
        "tstop": end_t,
        "dt": dt,
        "pt3d": True,
}
cell = LFPy.Cell(**cell_parameters)

num_tsteps = int(end_t / dt + 1)
tvec = np.arange(num_tsteps) * dt

i = 0
syn = None
synlist = []
input_vecs = []

# Create simulated magnetic stimulation.
stim = MagStim(cell, tvec)
i_am_current = stim.calc_i_am()

# Define input locations for current.
input_idxs = np.array(range(stim.num_of_seg))

for sec in cell.allseclist:
    for seg in sec:
        # print("Input inserted in ", sec.name(), seg.x)

        # Arbitrary input current can be inserted here
        input_array = i_am_current[i] * 10**9  # current with units nA

        # Insert the current into the compartment
        noise_vec = neuron.h.Vector(input_array)
        syn = neuron.h.ISyn(sec(seg.x))
        syn.dur = 1e9
        syn.delay = 0
        noise_vec.play(syn._ref_amp, cell.dt)
        synlist.append(syn)
        input_vecs.append(noise_vec)
        i += 1

cell.simulate(rec_vmem=True, rec_imem=True)


# The rest is plotting
fig = plt.figure(figsize=[9, 6])
fig.subplots_adjust(wspace=0.5, top=0.9, bottom=0.1, hspace=0.6)
ax1 = fig.add_subplot(211, aspect=1, frameon=False, title="cell",
                      xlabel="x (µm)", ylabel="z (µm)",
                      ylim=[-200, 200], xlim=[-100, 1000])
ax2 = fig.add_subplot(212, xlabel="time (ms)",
                      ylabel="membrane\npotential (mV)")

[ax1.plot(cell.x[idx],  cell.z[idx], c='gray', lw=1)
 for idx in range(cell.totnsegs)]

plot_idxs = input_idxs
plot_idx_clrs = {idx: plt.cm.viridis(num / (len(plot_idxs) - 1))
                 for num, idx in enumerate(plot_idxs)}

for idx in plot_idxs:
    ax1.plot(cell.x[idx].mean(), cell.z[idx].mean(), 'o', c=plot_idx_clrs[idx])
    ax2.plot(cell.tvec, cell.vmem[idx, :], c=plot_idx_clrs[idx], lw=2)

plt.savefig(join("axon_testing.png"))
