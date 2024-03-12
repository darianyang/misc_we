#!/bin/bas
python free_energy_plotter_whole.py --pdist-input pdist.h5 --first-iter 1 --xlabel "central C-N distance ($\AA$)" --ylabel 'alternate C-N distance ($\AA$)' --xrange "(1.55,6)" --yrange "(1.575,6)" --plot-mode contourf_l --output trace_whole.pdf --pdist-axes "(0,1)" --cmap "Blues_r" --zmax 16 --zmin 0 --zbins 8 --smooth-curves 0.4 --smooth-data 0.4
#--postprocess plotting.func
