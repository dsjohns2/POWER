#!/bin/bash

for folder in "$@"
do
	run=${folder##*/}
	./power.py $folder
	./plot_monitor.py $run
done

