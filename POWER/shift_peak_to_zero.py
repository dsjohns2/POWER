#!/usr/bin/env python

import numpy
import sys
import os

for path in sys.argv[1:]:
	main_dir = path
	sim = path.split("/")[-2]

	strain_data = numpy.loadtxt(path)
	cur_max_time = strain_data[0][0]
	cur_max_amp = abs(pow(strain_data[0][1], 2)) + abs(pow(strain_data[0][2], 2))
	for i in strain_data[:]:
		cur_time = i[0]
		cur_amp = abs(pow(i[1], 2)) + abs(pow(i[2], 2))
		if(cur_amp>cur_max_amp):
			cur_max_amp = cur_amp
			cur_max_time = cur_time
	for i in strain_data[:]:
		i[0] -= cur_max_time
	numpy.savetxt("./Extrapolated_Strain/"+sim+"/"+sim+"_shifted_radially_extrapolated_strain.dat", strain_data)
	print(cur_max_time)
