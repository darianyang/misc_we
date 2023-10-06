#!/bin/bash
#
# --------------------------------
# Amber Trajectory Tool for WESTPA
# --------------------------------
# 
# Written by Anthony Bogetti on 28.08.18, updated by Darian Yang on 06Oct2023
# 
# This script will stitch together a trajectory file from your Amber-WESTPA
# simulation that can be viewed in VMD or another molecular dynmaics 
# visualization software.  Run this script with the command ./amberTraj.sh
# from the same directory where the west.h5 file from your WESTPA simulation
# is located.  The results of this analysis will be stored in a new folder
# called trajAnalysis as the file trace.nc.  Load trace.nc into VMD to 
# visualize the trajectory.  As a note, you will need to have your computer
# configured to run w_succ from the WESTPA software package and cpptraj from 
# the Amber software package. Though, if the simulation has completed successfully,
# these commands will most likely be ready to run.

siter=$1
sseg=$2
PARM=$3

w_trace $siter:$sseg -W west.h5

cat $(echo 'traj_'$siter'_'$sseg'_trace.txt') | tail -n +9 > path.txt

echo Tracing trajectory

while read file; do
    iter=$(echo $file | awk '{print $1}')
    echo $iter
    seg=$(echo $file | awk '{print $2}')
    filestring='traj_segs/'$(printf "%06d" $iter)'/'$(printf "%06d" $seg)'/''seg.nc' 
    echo "trajin $filestring" >> cpptraj.in
done < "path.txt"

printf "autoimage\n" >> cpptraj.in 
printf "trajout TRAJ_it${siter}_seg${sseg}_trace.nc\nrun" >> cpptraj.in 


cpptraj -p $PARM -i cpptraj.in > traj.log

echo Trajectory file creation is complete.

rm cpptraj.in path.txt trajs.h5

