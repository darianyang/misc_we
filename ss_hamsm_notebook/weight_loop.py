# weight_loop.py
#
# Module made to reweight a west.h5 file with weights generated from a msm_we object.
#
# Contains code written by JD Russo's haMSM restarting plugin for WESTPA.
#
# Contains some design choices which might be upsetting to some:
# - msm_bin_we_weights (denominator of Alg 5.3) are normalized all runs/per iteration.
# - some iterations (mostly early ones) might not occupy all msm_bins. Those weights
#   are redistributed by setting that corresponding pSS to 0 then renormalizing, essentially
#   redistributing those weights to occupied bins based on their pSS weights.
#
# Written by Jeremy Leung
#
# Last Edited: Oct 18th, 2022
#

import logging

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("msm_we")
log.setLevel(logging.DEBUG)

import numpy
import mdtraj
import io
import h5py
import os
import shutil
from pathlib import Path
from tqdm.auto import trange, tqdm
from contextlib import ExitStack


def create_reweighted_h5_global(msmwe_obj, west_name=None, new_name="west_reweight.h5", copy=True, gen_sstates=False, struct_filetype="rst7", pdb_out=True, link_out=False, link_path="/ocean/projects/mcb180038p/jml230/bdpa_wsh2045_p3",):
    """
    This function duplicates h5 files and replace those new files with new weights generated by the msm_we module.
    Normalizes based on the GLOBAL distribution.

    Output file name can be changed using the new_name option.

    It uses code written by JD Russo in the haMSM restarting plugin and accepts a msm_we object as an input.

    It works with jdrusso/msm_we as of September 28th, 2021 (49ab465). It's always being updated, so you never know...

    Essentially runs as follows:
        1) Duplicate all west.h5 files (based on msmwe_obj.fileList) and open them within a context manager.
        2) Extract the pSS into bin_prob and set source/sink to 0 for eq simualations. (Not necessary in newer versions of msm_we)
        3) Sums up msm_bin_we_weights for all runs all iterations.
        4) Goes through a loop per iteration that:
            a. Run through each segment and apply Alg 5.3
            b. If gen_sstates is True, proceed to generate startingstates.
        5) Check if all iterations sum up to 1.

    Parameters
    ==========

    msmwe_obj : class
        The msmwe_obj model object from msm_we.

    west_name : str or None, default: None
        Alternate name for the west.h5 files. This is useful for when your west.h5 is really bulky with the coordinates.
        None implies it will use whatever's available in the msmwe_obj.fileList.

    copy : bool, default: True
        Copy the h5 files. Or else, it would just fail if those files don't exist already.

    new_name : str, default: "west_reweight.h5"
        Name of the new west.h5 with the reweighted weights. 

    gen_sstates : bool, default: False
        Generate new start states file for restarting.

    struct_filetype : str, default: "rst7"
        Filetype to be outputed for start states. 

    pdb_out : bool, default: True
        Sets gen_sstates output to pdb files from the west.h5 coordinates array. Does not work unless gen_sstates=True.

    link_out : bool, default: False
        Sets gen_sstates output to include the bash script to link your existing restart files.
        Does not work unless gen_sstates=True. 
    """

    with ExitStack() as stack:
        file_list = []
        new_file_list = []
        new_name = new_name.rsplit(".h5", maxsplit=1)[0]

        # Copies each file.
        for i in trange(0, len(msmwe_obj.fileList), desc='Managing H5 Files I/O'):
            new_file = (msmwe_obj.fileList[i].rsplit("/", maxsplit=1)[0] + "/" + str(new_name) + ".h5")
            if type(west_name) is str:
                old_file = (msmwe_obj.fileList[i].rsplit("/", maxsplit=1)[0] + "/" + str(west_name))
            else:
                old_file = msmwe_obj.fileList[i]
            if copy is True:
                shutil.copyfile(old_file, new_file)
            else:
                if not exists(new_file):
                    assert FileNotFoundError, f"{new_file} not found. Use copy=True to copy the files."
            file_list.append(new_file)

        # For outputting starting states
        if gen_sstates:
            from westpa.core.extloader import get_object

            STRUCT_FILETYPES = {
                "pdb": "mdtraj.formats.PDBtrajectoryFile",
                "rst7": "mdtraj.formats.AmberRestartFile",
                "ncrst": "mdtraj.formats.AmberNetCDFRestartFile",
            }

            home_directory = msmwe_obj.fileList[0].rsplit("/", maxsplit=2)[0]
            Path(f"{home_directory}/sstates").mkdir(exist_ok=True)
            sstates_filename = f"{home_directory}/startingstates.txt"

            # Just a dumb way to remake the sstates file....
            with open(sstates_filename, 'w'):
                pass

            if struct_filetype in STRUCT_FILETYPES:
                output_filetype = get_object(STRUCT_FILETYPES[struct_filetype])
            else:
                try:
                    output_filetype = get_object(struct_filetype)
                except ImportError:
                    output_filetype = STRUCT_FILETYPES["pdb"]
            
            if link_out:
                with open(f"{home_directory}/link_ss.sh", "w") as fb:
                    fb.write("#!/bin/bash \n\n")
                    fb.write('mkdir -p sstates\n')

                with open(f"{home_directory}/pcoord.dat", 'w'):
                    pass


        # Open each file within the conext_manager
        # TODO: this requires opening each file at the same time. Might want to change that to streaming or sth as it could be RAM intensive.
        for f_path in file_list:
            f = stack.enter_context(h5py.File(f_path, "r+"))
            new_file_list.append(f)

        # Creating an index of bin prob and msm_bin_we_weight.
        bin_prob = numpy.zeros(msmwe_obj.nBins)
        msm_bin_we_weight = numpy.zeros(msmwe_obj.nBins)

        # Make sure pSS is actually the right type/shape
        if type(msmwe_obj.pSS) is numpy.matrix:
            ss_alg = numpy.squeeze(msmwe_obj.pSS.A)
        else:
            ss_alg = numpy.squeeze(msmwe_obj.pSS)

        # Get pSS and total amount of WE weight for each MSM microbin
        for msm_bin_idx in range(0, len(bin_prob)):

            try:
                bin_prob[msm_bin_idx] = ss_alg[msm_bin_idx]
            except IndexError:
                bin_prob[msm_bin_idx] = 0
                log.info(
                    f"MSM-Bin {msm_bin_idx} does not exist. Either it's not cleaned up properly or you're running an equilibrium simulation. Assuming the latter so bin probability's been set to 0."
                )

            # try:
            #    msm_bin_we_weight[msm_bin_idx] = sum(msmwe_obj.cluster_structure_weights[msm_bin_idx])
            # except KeyError:
            #    msm_bin_we_weight[msm_bin_idx] = 0
            #    log.info(f"MSM-Bin {msm_bin_idx} does not exist. Either it's not cleaned up properly or you're running an equilibrium simulation. Assuming the latter so msm_bin_we_weight's been set to 0.")

        assert numpy.isclose(sum(bin_prob), 1), "Your pSS doesn't add up to 1."

        # Calculate for second to last iter (msmwe_obj.maxIter-1)
        # TODO: Change so it reads from cluster_structure_weights + loop through maxIter-1
        for iter_count in trange(0, msmwe_obj.maxIter - 1, desc="Summing Weights"):  # Goes through each iter

            iters = "iter_" + str(iter_count + 1).rjust(8, "0")
            file_idx = 0  # File number
            seg_idx = 0  # Seg index within each file

            # Calculate the total amount of WE weight in each MSM microbin this iteration
            for seg_count in range(0, int(msmwe_obj.numSegments[iter_count])):

                # Label this segment with the cluster number
                seg_cluster_index = msmwe_obj.dtrajs[iter_count][seg_count]

                # Add the weight to that microbin
                # TODO Change so it's not necessary to read from the h5 directly every time?
                try:
                    msm_bin_we_weight[seg_cluster_index] += new_file_list[file_idx][f"iterations/{iters}/seg_index"]["weight", seg_idx]

                except IndexError:
                    file_idx += 1
                    seg_idx = 0

                    msm_bin_we_weight[seg_cluster_index] += new_file_list[file_idx][f"iterations/{iters}/seg_index"]["weight", seg_idx]

                seg_idx += 1

        assert numpy.isclose(
            sum(msm_bin_we_weight), ((msmwe_obj.maxIter - 1) * (msmwe_obj.n_data_files))
        ), f"Your msm_bin_we_weights doesn't add up to expected value of {((msmwe_obj.maxIter-1)*(msmwe_obj.n_data_files))}. They add up to {sum(msm_bin_we_weight)}"

        total_weight = 0.0  # Counter to track total weight
        struct_idx = 0  # For naming purposes

        # For each iter and each seg, create bin_prob, then go through each seg to save new weights into h5
        for iter_count in trange(
            0, msmwe_obj.maxIter - 1, desc="Outputting Weights"
        ):  # Goes through each iter

            iters = "iter_" + str(iter_count + 1).rjust(8, "0")
            file_idx = 0  # File number
            seg_idx = 0  # Seg index within each file

            # Doing the actual calcs for the weights

            # Resetting the counters
            file_idx = 0  # File number
            seg_idx = 0  # Seg index within each file

            for seg_count in range(0, int(msmwe_obj.numSegments[iter_count])):  # goes through each seg

                try:
                    # Structure weights are set according to Algorithm 5.3 in
                    # Aristoff, D. & Zuckerman, D. M. Optimizing Weighted Ensemble Sampling of Steady States.
                    # Multiscale Model Sim 18, 646–673 (2020).

                    seg_cluster_index = msmwe_obj.dtrajs[iter_count][seg_count]

                    seg_we_weight = new_file_list[file_idx][f"iterations/{iters}/seg_index"]["weight", seg_idx]
                    structure_weight = seg_we_weight * (bin_prob[seg_cluster_index] / msm_bin_we_weight[seg_cluster_index])

                    g = new_file_list[file_idx]["/iterations/" + iters + "/seg_index"]
                    g["weight", seg_idx] = structure_weight

                except IndexError:
                    file_idx += 1
                    seg_idx = 0

                    seg_cluster_index = msmwe_obj.dtrajs[iter_count][seg_count]

                    seg_we_weight = new_file_list[file_idx][f"iterations/{iters}/seg_index"]["weight", seg_idx]
                    structure_weight = seg_we_weight * (bin_prob[seg_cluster_index] / msm_bin_we_weight[seg_cluster_index])

                    g = new_file_list[file_idx]["/iterations/" + iters + "/seg_index"]
                    g["weight", seg_idx] = structure_weight

                # Now to output file and substitute into startstates.txt
                if gen_sstates:

                    structure_name = f"bin{seg_cluster_index}_run{file_idx+1}_it{iter_count+1}_seg{seg_idx}"
                    structure_filename = f"sstates/{structure_name}"
                    structure_fullpath = f"{home_directory}/{structure_filename}.{struct_filetype}"

                    if link_out:
                        with open(f"{home_directory}/link_ss.sh", "a") as fb:
                            fb.write(f"mkdir -p {structure_filename} &&\n")
                            fb.write(
                                f"ln -s {link_path}_{file_idx+1}/traj_segs/{iter_count+1:>06}/{seg_idx:>06}/seg.{struct_filetype} {structure_filename}/seg.{struct_filetype} \n")

                            pcoord = new_file_list[file_idx][f"/iterations/{iters}/pcoord"][seg_idx, -1, 0] # only the first dimension
                            fb.write(f"echo {pcoord} > {structure_filename}/pcoord.init\n")
                        
                        # Pcoord output
                        with open(f"{home_directory}/pcoord.dat",'a') as fp:
                            fp.write(f'{structure_name}\t{pcoord}\n')

                    if pdb_out:
                        with output_filetype(structure_fullpath, "w") as struct_file:

                            try:
                                angles = msmwe_obj.reference_structure.unitcell_angles[0]
                                lengths = (msmwe_obj.reference_structure.unitcell_lengths[0] * 10)
                            # This throws TypeError if reference_structure.unitcell_angles is None, or AttributeError
                            #   if reference_structure.unitcell_angles doesn't exist.
                            except (TypeError, AttributeError):
                                angles, lengths = None, None

                            coords = new_file_list[file_idx][f"/iterations/{iters}/auxdata/coord"][seg_idx, -1] * 10  # Correct units
  # Correct units

                            # Write the structure file
                            if output_filetype is mdtraj.formats.PDBTrajectoryFile:
                                struct_file.write(
                                    coords,
                                    topology,
                                    modelIndex=1,
                                    unitcell_angles=angles,
                                    unitcell_lengths=lengths,
                                )

                            elif output_filetype is mdtraj.formats.AmberRestartFile:
                                # AmberRestartFile takes slightly differently named keyword args
                                struct_file.write(
                                    coords,
                                    time=None,
                                    cell_angles=angles,
                                    cell_lengths=lengths,
                                )

                            else:
                                # Otherwise, YOLO just hope all the positional arguments are in the right place
                                log.warning(
                                    f"This output filetype ({struct_filetype}) is probably supported, "
                                    f"but not explicitly handled."
                                    " You should ensure that it takes argument as (coords, topology)"
                                )
                                struct_file.write(coords, topology)
                                raise Exception(
                                    "Don't know what extension to use for this filetype"
                                )

                    with open(sstates_filename, "a") as fp:
                        # Add this start-state to the start-states file
                        fp.write(
                            f"b{seg_cluster_index}_s{struct_idx}\t{structure_weight}\t{structure_filename}\n"
                        )

                total_weight += structure_weight
                seg_idx += 1
                struct_idx += 1

            # Final check for normalization.
            # print(str(iter_count+1) + ": " + str(total_weight) + " " + str(sum(msm_bin_we_weight))) # For debugging

        assert numpy.isclose(
            total_weight, (1)
        ), f"Total steady-state structure weights are not normalized in this simulation! Sum up to {total_weight}."


# The Following are all deprecated. It imposes artificial conditions which might violate Alg 5.3's assumptions.
def create_reweighted_h5(msmwe_obj, new_name="west_reweight.h5"):
    """
    This function duplicates h5 files and replace those new files with new weights generated by the msm_we module.

    Output file name can be changed using the new_name option.

    It uses code written by JD Russo in the haMSM restarting plugin and accepts a msm_we object as an input.

    It works with jdrusso/msm_we as of September 28th, 2021 (49ab465). It's always being updated, so you never know...

    Essentially runs as follows:
        1) Duplicate all west.h5 files (based on msmwe_obj.fileList) and open them within a context manager.
        2) Extract the pSS into bin_prob and set source/sink to 0 for eq simulations.
        3) Goes through a loop per iteration that:
            a. Sum up msm_bin_we_weights for all runs that iteration.
            b. Copy bin_prob into bin_prob_it. If any of those msm_bins are empty in msm_bin_we_weights, set = 0.
            c. Run through each segment and apply Alg 5.3
            d. Check if each iteration sum up to 1.
    """
    with ExitStack() as stack:
        file_list = []
        new_file_list = []
        new_name = new_name.rsplit(".h5", maxsplit=1)[0]

        # Copies each file.
        for i in range(0, len(msmwe_obj.fileList)):
            new_file = (
                msmwe_obj.fileList[i].rsplit("/", maxsplit=1)[0]
                + "/"
                + str(new_name)
                + ".h5"
            )
            shutil.copyfile(msmwe_obj.fileList[i], new_file)
            file_list.append(new_file)

        # Open each file within the conext_manager
        # TODO: this requires opening each file at the same time. Might want to change that to streaming or sth as it could be RAM intensive.
        for f_path in file_list:
            f = stack.enter_context(h5py.File(f_path, "r+"))
            new_file_list.append(f)

        # Creating an index of bin prob.
        bin_prob = numpy.zeros(msmwe_obj.nBins)

        # Make sure pSS is actually the right type/shape
        if type(msmwe_obj.pSS) is numpy.matrix:
            ss_alg = numpy.squeeze(msmwe_obj.pSS.A)
        else:
            ss_alg = numpy.squeeze(msmwe_obj.pSS)

        # Get pSS for each msm_bin
        for msm_bin_idx in range(0, len(bin_prob)):

            try:
                bin_prob[msm_bin_idx] = ss_alg[msm_bin_idx]
            except IndexError:
                bin_prob[msm_bin_idx] = 0
                log.info(
                    f"MSM-Bin {msm_bin_idx} does not exist. Either it's not cleaned up properly or you're running an equilibrium simulation. Assuming the latter so it's been set to 0."
                )

        assert numpy.isclose(sum(bin_prob), 1), "Your pSS doesn't add up to 1."

        # For each iter and each seg, create msm_bin_we_weight, then go through each seg to save new weights into h5
        for iter_count in trange(0, msmwe_obj.maxIter - 1):  # Goes through each iter

            iters = "iter_" + str(iter_count + 1).rjust(8, "0")
            file_idx = 0  # File number
            seg_idx = 0  # Seg index within each file

            msm_bin_we_weight = numpy.zeros(msmwe_obj.nBins)



            # Calculate the total amount of WE weight in each MSM microbin this iteration
            for seg_count in range(0, int(msmwe_obj.numSegments[iter_count])):

                # Label this segment with the cluster number
                seg_cluster_index = msmwe_obj.dtrajs[iter_count][seg_count]

                # Add the weight to that microbin
                # TODO Change so it's not necessary to read from the h5 directly every time?
                try:
                    msm_bin_we_weight[seg_cluster_index] += new_file_list[file_idx][
                        "iterations"
                    ][iters]["seg_index"]["weight"][seg_idx]
                    seg_idx += 1

                except IndexError:
                    file_idx += 1
                    seg_idx = 0

                    msm_bin_we_weight[seg_cluster_index] += new_file_list[file_idx][
                        "iterations"
                    ][iters]["seg_index"]["weight"][seg_idx]
                    seg_idx += 1

            assert numpy.isclose(
                sum(msm_bin_we_weight), msmwe_obj.n_data_files
            ), "msm_bin_we_weight is not adding up to the number of files you have. Were the weights in those files previously normalized or did something went wrong?"

            # Adjusting the pSS based on msm_bin occupancy this iteration, then normalize
            bin_prob_it = numpy.copy(bin_prob)
            adj_pSS = numpy.where(msm_bin_we_weight == 0)[0]

            for zero_idx in adj_pSS:
                bin_prob_it[zero_idx] = 0

            bin_prob_it = numpy.divide(bin_prob_it, sum(bin_prob_it))

            # print("The pSS of iteration " + str(iter_count+1) + " is: " + str(bin_prob_it)) # For debugging

            # Doing the actual calcs for the weights

            # Resetting the counters
            file_idx = 0  # File number
            seg_idx = 0  # Seg index within each file
            total_weight = 0.0  # Counter to track total weight

            for seg_count in range(
                0, int(msmwe_obj.numSegments[iter_count])
            ):  # goes through each seg

                try:
                    # Structure weights are set according to Algorithm 5.3 in
                    # Aristoff, D. & Zuckerman, D. M. Optimizing Weighted Ensemble Sampling of Steady States.
                    # Multiscale Model Sim 18, 646–673 (2020).

                    seg_cluster_index = msmwe_obj.dtrajs[iter_count][seg_count]

                    seg_we_weight = new_file_list[file_idx]["iterations"][iters][
                        "seg_index"
                    ]["weight"][seg_idx]
                    structure_weight = seg_we_weight * (
                        bin_prob_it[seg_cluster_index]
                        / msm_bin_we_weight[seg_cluster_index]
                    )

                    g = new_file_list[file_idx]["/iterations/" + iters + "/seg_index"]
                    g["weight", seg_idx] = structure_weight

                    total_weight += structure_weight

                    seg_idx += 1

                except IndexError:
                    file_idx += 1
                    seg_idx = 0

                    seg_cluster_index = msmwe_obj.dtrajs[iter_count][seg_count]

                    seg_we_weight = new_file_list[file_idx]["iterations"][iters][
                        "seg_index"
                    ]["weight"][seg_idx]
                    structure_weight = seg_we_weight * (
                        bin_prob_it[seg_cluster_index]
                        / msm_bin_we_weight[seg_cluster_index]
                    )

                    g = new_file_list[file_idx]["/iterations/" + iters + "/seg_index"]
                    g["weight", seg_idx] = structure_weight

                    total_weight += structure_weight

                    seg_idx += 1

            # Final check for normalization.
            # print(str(iter_count+1) + ": " + str(total_weight) + " " + str(sum(msm_bin_we_weight))) # For debugging
            assert numpy.isclose(
                total_weight, (1 - bin_prob_it[-1] - bin_prob_it[-2])
            ), f"Total steady-state structure weights are not normalized in iteration {iter_count+1}! Sum up to {total_weight}."


def create_reweighted_h5_block(msmwe_obj, new_name="west_reweight.h5", use_block=0.25):
    """
    This function duplicates h5 files and replace those new files with new weights generated by the msm_we module.

    Normalizes the weights based on a blocks of iteration specified by the `use_block` variable, which is bounded from 0 to 1. If you choose 1, use the create_reweighted_h5_global() function instead.

    Output file name can be changed using the new_name option.

    It uses code written by JD Russo in the haMSM restarting plugin and accepts a msm_we object as an input.

    It works with jdrusso/msm_we as of September 28th, 2021 (49ab465). It's always being updated, so you never know...

    Essentially runs as follows:
        1) Duplicate all west.h5 files (based on msmwe_obj.fileList) and open them within a context manager.
        2) Extract the pSS into bin_prob and set source/sink to 0 for eq simulations.
        3) Goes through a loop per iteration "block":
            a. Sum up msm_bin_we_weights for that block.
            b. Runs through each segment in block and apply Alg 5.3
            c. Check if each block sums up to 1.
    """
    with ExitStack() as stack:

        # Check if input varibles are valid
        assert (
            use_block != 1
        ), "Use `create_reweighted_h5_global()` if you want to reweight for all data"
        assert (
            use_block > 0 and use_block < 1
        ), "Invalid use_block value. Must be between 0 (exclusive) and 1 (inclusive)."

        # Setup variables
        file_list = []
        new_file_list = []
        new_name = new_name.rsplit(".h5", maxsplit=1)[0]

        # Copies each file.
        for i in range(0, len(msmwe_obj.fileList)):
            new_file = (
                msmwe_obj.fileList[i].rsplit("/", maxsplit=1)[0]
                + "/"
                + str(new_name)
                + ".h5"
            )
            shutil.copyfile(msmwe_obj.fileList[i], new_file)
            file_list.append(new_file)

        # Open each file within the conext_manager
        # TODO: this requires opening each file at the same time. Might want to change that to streaming or sth as it could be RAM intensive.
        for f_path in file_list:
            f = stack.enter_context(h5py.File(f_path, "r+"))
            new_file_list.append(f)

        # Creating an index of bin prob.
        bin_prob = numpy.zeros(msmwe_obj.nBins)

        # Make sure pSS is actually the right type/shape
        if type(msmwe_obj.pSS) is numpy.matrix:
            ss_alg = numpy.squeeze(msmwe_obj.pSS.A)
        else:
            ss_alg = numpy.squeeze(msmwe_obj.pSS)

        # Get pSS for each msm_bin
        for msm_bin_idx in range(0, len(bin_prob)):

            try:
                bin_prob[msm_bin_idx] = ss_alg[msm_bin_idx]
            except IndexError:
                bin_prob[msm_bin_idx] = 0
                log.info(
                    f"MSM-Bin {msm_bin_idx} does not exist. Either it's not cleaned up properly or you're running an equilibrium simulation. Assuming the latter so it's been set to 0."
                )

        assert numpy.isclose(sum(bin_prob), 1), "Your pSS doesn't add up to 1."

        # Calculate the block boundaries, rounding up since in python, stop is exclusive. There are a lot of intricasies regarding floating point errors, so here it is.
        block_array = numpy.array([0])
        test_bound = msmwe_obj.maxIter - 1

        delta = use_block * float(msmwe_obj.maxIter - 1)
        if numpy.isclose(delta, int(delta)) or numpy.isclose(delta, int(delta) + 1):
            delta = round(delta)
        else:
            delta = int(numpy.ceil(delta))

        log.info(
            f"New weights will be normalized using blocks {delta} iterations long, potentially except the last one if {use_block} doesn't divide properly."
        )

        test_bound = 0

        for iblock in range(0, msmwe_obj.maxIter - 1):
            if test_bound + delta <= msmwe_obj.maxIter - 1:
                test_bound += delta
                block_array = numpy.append(block_array, test_bound)
            else:
                if block_array[-1] == msmwe_obj.maxIter - 1:
                    break
                else:
                    block_array = numpy.append(block_array, msmwe_obj.maxIter - 1)
                    log.info(
                        f"Number of iterations could not be divided evenly by {use_block}. Last block only covers {block_array[-1]-block_array[-2]} iterations while the rest are {delta} long."
                    )
                    break
        log.debug(f"Block boundaries are: {block_array}")

        # For each block, create msm_bin_we_weight, then go through each iter and seg to save new weights into h5

        for iblock in trange(0, len(block_array) - 1, desc="Processing Iteration"):

            delta = block_array[iblock + 1] - block_array[iblock]
            msm_bin_we_weight = numpy.zeros(msmwe_obj.nBins)
            total_weight = 0  # For control purposes

            for iter_count in trange(
                block_array[iblock],
                block_array[iblock + 1],
                desc="Calculating Weights",
                leave=False,
            ):  # Goes through each iter

                iters = "iter_" + str(iter_count + 1).rjust(8, "0")
                file_idx = 0  # File number
                seg_idx = 0  # Seg index within each file

                # Calculate the total amount of WE weight in each MSM microbin this iteration
                for seg_count in range(0, int(msmwe_obj.numSegments[iter_count])):

                    # Label this segment with the cluster number
                    seg_cluster_index = msmwe_obj.dtrajs[iter_count][seg_count]

                    # Add the weight to that microbin
                    # TODO Change so it's not necessary to read from the h5 directly every time?
                    try:
                        msm_bin_we_weight[seg_cluster_index] += new_file_list[file_idx][
                            "iterations"
                        ][iters]["seg_index"]["weight"][seg_idx]
                        seg_idx += 1

                    except IndexError:
                        file_idx += 1
                        seg_idx = 0

                        msm_bin_we_weight[seg_cluster_index] += new_file_list[file_idx][
                            "iterations"
                        ][iters]["seg_index"]["weight"][seg_idx]
                        seg_idx += 1

            assert numpy.isclose(
                sum(msm_bin_we_weight), (msmwe_obj.n_data_files * delta)
            ), f"msm_bin_we_weight is not adding up to the number of files you have. Were the weights in those files previously normalized or did something went wrong? They add up to {sum(msm_bin_we_weight)}."

            # Adjusting the pSS based on msm_bin occupancy this iteration, then normalize
            bin_prob_block = numpy.copy(bin_prob)
            adj_pSS = numpy.where(msm_bin_we_weight == 0)[0]

            for zero_idx in adj_pSS:
                bin_prob_block[zero_idx] = 0

            bin_prob_block = numpy.divide(bin_prob_block, sum(bin_prob_block))

            assert numpy.isclose(
                sum(bin_prob_block), 1
            ), "bin_prob_block is not normalized."
            # print(f"The pSS of this block {i+1} is: " + str(bin_prob_it)) # For debugging

            # Doing the actual calcs for the weights this block
            for iter_count in trange(
                block_array[iblock],
                block_array[iblock + 1],
                desc="Outputting Weights",
                leave=False,
            ):  # Goes through each iter
                iters = "iter_" + str(iter_count + 1).rjust(8, "0")
                file_idx = 0  # File number
                seg_idx = 0  # Seg index within each file

                for seg_count in range(
                    0, int(msmwe_obj.numSegments[iter_count])
                ):  # goes through each seg
                    try:
                        # Structure weights are set according to Algorithm 5.3 in
                        # Aristoff, D. & Zuckerman, D. M. Optimizing Weighted Ensemble Sampling of Steady States.
                        # Multiscale Model Sim 18, 646–673 (2020).

                        seg_cluster_index = msmwe_obj.dtrajs[iter_count][seg_count]

                        seg_we_weight = new_file_list[file_idx]["iterations"][iters][
                            "seg_index"
                        ]["weight"][seg_idx]
                        structure_weight = seg_we_weight * (
                            bin_prob_block[seg_cluster_index]
                            / msm_bin_we_weight[seg_cluster_index]
                        )

                        g = new_file_list[file_idx][
                            "/iterations/" + iters + "/seg_index"
                        ]
                        g["weight", seg_idx] = structure_weight

                        total_weight += structure_weight

                        seg_idx += 1

                    except IndexError:
                        file_idx += 1
                        seg_idx = 0

                        seg_cluster_index = msmwe_obj.dtrajs[iter_count][seg_count]

                        seg_we_weight = new_file_list[file_idx]["iterations"][iters][
                            "seg_index"
                        ]["weight"][seg_idx]
                        structure_weight = seg_we_weight * (
                            bin_prob_block[seg_cluster_index]
                            / msm_bin_we_weight[seg_cluster_index]
                        )

                        g = new_file_list[file_idx][
                            "/iterations/" + iters + "/seg_index"
                        ]
                        g["weight", seg_idx] = structure_weight

                        total_weight += structure_weight

                        seg_idx += 1

                # Final check for normalization.
                # print(str(iter_count+1) + ": " + str(total_weight) + " " + str(sum(msm_bin_we_weight))) # For debugging
            assert numpy.isclose(
                total_weight, (1 - bin_prob_block[-1] - bin_prob_block[-2])
            ), f"Total steady-state structure weights are not normalized in block bounded by ({block_array[i]},{block_array[i+1]})! Sum up to {total_weight}."
