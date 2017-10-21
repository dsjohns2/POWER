#!/usr/bin/env python

# Copyright (c) 2017 The Board of Trustees of the University of Illinois
# All rights reserved.
#
# Developed by: Daniel Johnson, E. A. Huerta, Roland Haas
#               NCSA Gravity Group
#               National Center for Supercomputing Applications
#               University of Illinois at Urbana-Champaign
#               http://gravity.ncsa.illinois.edu/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal with the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimers.
#
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimers in the documentation
# and/or other materials provided with the distribution.
#
# Neither the names of the National Center for Supercomputing Applications,
# University of Illinois at Urbana-Champaign, nor the names of its
# contributors may be used to endorse or promote products derived from this
# Software without specific prior written permission.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# WITH THE SOFTWARE.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

run_name = sys.argv[1]

python_strain = np.loadtxt("./Extrapolated_Strain/"+run_name+"/"+run_name+"_radially_extrapolated_strain_l2_m2.dat")
python_phase = np.loadtxt("./Extrapolated_Strain/"+run_name+"/"+run_name+"_radially_extrapolated_phase_l2_m2.dat")
python_amp = np.loadtxt("./Extrapolated_Strain/"+run_name+"/"+run_name+"_radially_extrapolated_amplitude_l2_m2.dat")
py_t = python_strain[:, 0]
py_hplus = python_strain[:, 1]
py_phase = python_phase[:, 1]
py_amp = python_amp[:, 1]

cur_max_time = python_strain[0][0]
cur_max_amp = abs(pow(python_strain[0][1], 2)) + abs(pow(python_strain[0][2], 2))
for i in python_strain[:]:
	cur_time = i[0]
	cur_amp = abs(pow(i[1], 2)) + abs(pow(i[2], 2))
	if(cur_amp>cur_max_amp):
		cur_max_amp = cur_amp
		cur_max_time = cur_time

run_name_without_resolution = run_name.split("_")[0]

matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'font.family': 'serif'})
plt.title(run_name_without_resolution)
plt.plot(py_t, py_hplus)
plt.xlim((0,cur_max_time+200))
plt.ylabel('${\mathfrak{R}} [h(t)]$')
plt.xlabel('Time [M]')
plt.savefig("./Extrapolated_Strain/"+run_name+"/"+run_name+"_strain_plot_l2_m2.png")
plt.close()

matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'font.family': 'serif'})
plt.title(run_name_without_resolution)
plt.plot(py_t, py_phase)
plt.xlim((0,cur_max_time+200))
plt.ylabel('Phase')
plt.xlabel('Time [M]')
plt.savefig("./Extrapolated_Strain/"+run_name+"/"+run_name+"_phase_plot_l2_m2.png")
plt.close()

matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'font.family': 'serif'})
plt.title(run_name_without_resolution)
plt.plot(py_t, py_amp)
plt.xlim((0,cur_max_time+200))
plt.ylabel('Amplitude')
plt.xlabel('Time [M]')
plt.savefig("./Extrapolated_Strain/"+run_name+"/"+run_name+"_amplitude_plot_l2_m2.png")
plt.close()
