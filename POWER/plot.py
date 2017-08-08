#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

run_name = sys.argv[1]

python_strain = np.loadtxt("./Extrapolated_Strain/"+run_name+"/"+run_name+"_radially_extrapolated_strain.dat")
py_t = python_strain[:, 0]
py_hplus = python_strain[:, 1]

run_name_without_resolution = run_name.split("_")[0]

matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'font.family': 'serif'})
plt.title(run_name_without_resolution)
plt.plot(py_t, py_hplus)
plt.xlim((0,None))
plt.ylabel('${\mathfrak{R}} [h(t)]$')
plt.xlabel('Time [M]')
plt.savefig("./Extrapolated_Strain/"+run_name+"/strain_plot.png")
