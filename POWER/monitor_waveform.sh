#!/bin/bash

folder=$1

run=${folder##*/}

./power.py $folder
./plot_monitor.py $run
