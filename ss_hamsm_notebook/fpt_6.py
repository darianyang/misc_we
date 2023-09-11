import numpy as np
import sys
import h5py
import pickle
import os
import subprocess
import mdtraj as md
from msm_we.fpt import MatrixFPT
from westpa.core.binning import RectilinearBinMapper
import matplotlib.pyplot as plt
import westpa.analysis as analysis

sys.path.append("/home/STORAGE/qmmm/qmmm_2022/cat10_rc")
from msm_we import msm_we

import ray
ray.init()

#import logging
#from rich.logging import RichHandler
#log = logging.getLogger()
#log.addHandler(RichHandler())
#
#msm_log = logging.getLogger("msm_we.msm_we")
#
## Debug mode
#logging.getLogger('msm_we.msm_we').setLevel('DEBUG')

"""
This code
1. Initializes a model with the given PDBs and file globs
2. Opens the associated H5 file, in append mode
3. Reads through each of the trajectory files, putting the coordinates into the H5 file's auxdata/coord (if they don't already exist)
"""

fileSpecifier = ['/home/STORAGE/qmmm/qmmm_2022/cat10_rc/multi.h5']
refPDBfile    = '/home/STORAGE/qmmm/qmmm_2022/cat10_rc/cat10.pdb'
modelName     = 'cat10_all'
WEfolder      = '/home/STORAGE/qmmm/qmmm_2022/cat10_rc'

def processCoordinates(self, coords):

   if self.dimReduceMethod == "none":
       nC = np.shape(coords)
       nC = nC[0]
       new_coords = coords[:, :37, :]
       data = new_coords.reshape(nC, 3 * 37)
       model.nAtoms = 37
       return data

   if self.dimReduceMethod == "pca" or self.dimReduceMethod == "vamp":

       xt = md.Trajectory(xyz=coords, topology=None)
       indCAT = self.reference_structure.topology.select("resid 0 to 1")
       pair1, pair2 = np.meshgrid(indCAT, indCAT, indexing="xy")
       indUT = np.where(np.triu(pair1, k=1) > 0)
       pairs = np.transpose(np.array([pair1[indUT], pair2[indUT]])).astype(int)
       dist = md.compute_distances(xt, pairs, periodic=True, opt=True)

       return dist

def reduceCoordinates(self, coords):

    log.debug("Reducing coordinates")

    if self.dimReduceMethod == "none":
        nC = np.shape(coords)
        nC = nC[0]
        data = coords.reshape(nC, 3 * self.nAtoms)
        return data

    if self.dimReduceMethod == "pca" or self.dimReduceMethod == "vamp":
        coords = self.processCoordinates(coords)
        coords = self.coordinates.transform(coords)
        return coords

    raise Exception

msm_we.modelWE.processCoordinates = processCoordinates

new_bins = RectilinearBinMapper([[-np.inf, np.inf], [0, 1.6, 2, 5, 10, 20, 50]])
#new_bins = RectilinearBinMapper([[-np.inf, np.inf], [0, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5, 10, 50]])
#new_bins = RectilinearBinMapper([[-np.inf, np.inf], [0, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5, 10, 50]])
#new_bins = RectilinearBinMapper([[-np.inf, np.inf], [0, 1.6, 2, 2.25, 3, 4, 5, np.inf]])

# Model is loaded to get the number of iterations and segments
model = msm_we.modelWE()
print("initializing...")
model.initialize(fileSpecifier, refPDBfile, modelName, [[-np.inf, np.inf], [10, 50]], [[-np.inf, np.inf],[0,1.6]], "pca", 0.5E-12, 2)
#model.initialize(fileSpecifier, refPDBfile, modelName, [[-np.inf, np.inf], [10, 50]], [[-np.inf, np.inf],[0,2]], "pca", 0.5E-12, 2)
model.get_iterations()

#f = h5py.File(model.fileList[0], "a")

## Get coordinates
#getCoordinates(model=model, h5file=f)

#f.close()

model.get_iterations()
model.get_coordSet(last_iter=500)
model.dimReduce()
#
# Save model
from pickle import load, dump
with open("checkpoint1.pickle", "wb") as fo:
    dump(model, fo)

#### Load model
###from pickle import load, dump
###with open("checkpoint1.pickle", "rb") as fi:
###    model = load(fi)
#
model.cluster_coordinates(n_clusters=100, first_cluster_iter=100,  user_bin_mapper=new_bins)
#
# Save model
from pickle import load, dump
with open("checkpoint2.pickle", "wb") as fo:
    dump(model, fo)
#
#### Load model
###from pickle import load, dump
###with open("checkpoint2.pickle", "rb") as fi:
###    model = load(fi)
#
model.get_fluxMatrix(0, first_iter=100, use_ray=True)
#
# Save model
from pickle import load, dump
with open("checkpoint3.pickle", "wb") as fo:
    dump(model, fo)

## Load model
#from pickle import load, dump
#with open("checkpoint3.pickle", "rb") as fi:
#    model = load(fi)

model.organize_stratified(use_ray=True)
model.get_Tmatrix()

# Save model
from pickle import load, dump
with open("checkpoint4.pickle", "wb") as fo:
    dump(model, fo)

## Load model 4
#from pickle import load, dump
#with open("checkpoint4.pickle", "rb") as fi:
#    model = load(fi)

# Not sure if the following is correct yet
Tm = model.Tmatrix
print(Tm.shape)
np.save("Tmatrix.npy", Tm)
fptd = MatrixFPT.fpt_distribution(Tm, [model.indBasis], [model.indTargets], initial_distrib=[0.02], max_n_lags=500, clean_recycling=True)
print(fptd)
np.save("fpt_distribution.npy", fptd)

def generate_plots(model, restart_directory):

    model = model

#    log.info("Producing flux-profile, pseudocommittor, and target flux comparison plots.")
    flux_pcoord_fig, flux_pcoord_ax = plt.subplots()
    model.plot_flux(ax=flux_pcoord_ax, suppress_validation=True)
    flux_pcoord_fig.text(x=0.1, y=-0.05, s='This flux profile should become flatter after restarting', fontsize=12)
    flux_pcoord_ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
    flux_pcoord_fig.savefig(f'{restart_directory}/flux_plot.pdf', bbox_inches="tight")

    flux_pseudocomm_fig, flux_pseudocomm_ax = plt.subplots()
    model.plot_flux_committor(ax=flux_pseudocomm_ax, suppress_validation=True)
    flux_pseudocomm_fig.text(
        x=0.1,
        y=-0.05,
        s='This flux profile should become flatter after restarting.'
        '\nThe x-axis is a "pseudo"committor, since it may be '
        'calculated from WE trajectories in the one-way ensemble.',
        fontsize=12,
    )
    flux_pseudocomm_ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
    flux_pseudocomm_fig.savefig(f'{restart_directory}/pseudocomm-flux_plot.pdf', bbox_inches="tight")

    flux_comparison_fig, flux_comparison_ax = plt.subplots(figsize=(7, 3))
    # Get haMSM flux estimates
    models = [model]
    models.extend(model.validation_models)
    n_validation_models = len(model.validation_models)

    flux_estimates = []
    for _model in models:
        flux_estimates.append(_model.JtargetSS)

    hamsm_flux_colors = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    direct_flux_colors = iter(plt.cm.cool(np.linspace(0.2, 0.8, len(model.fileList))))

    # Get WE direct flux estimate
    for _file in model.fileList:

        run = analysis.Run(_file)
        last_iter = run.num_iterations
        recycled = list(run.iteration(last_iter - 1).recycled_walkers)
        target_flux = sum(walker.weight for walker in recycled) / model.tau

        # TODO: Correct for time!
        if len(_file) >= 15:
            short_filename = f"....{_file[-12:]}"
        else:
            short_filename = _file

        if target_flux == 0:
            continue

        flux_comparison_ax.axhline(
            target_flux,
            color=next(direct_flux_colors),
            label=f"Last iter WE direct {target_flux:.2e}",# f"\n  ({short_filename})",
            linestyle='--',
        )

    flux_comparison_ax.axhline(
        flux_estimates[0], label=f"Main model estimate\n  {flux_estimates[0]:.2e}", color=next(hamsm_flux_colors)
    )
    for i in range(1, n_validation_models + 1):
        flux_comparison_ax.axhline(
            flux_estimates[i],
            label=f"Validation model {i - 1} estimate\n  {flux_estimates[i]:.2e}",
            color=next(hamsm_flux_colors),
        )

    flux_comparison_ax.legend(bbox_to_anchor=(1.01, 0.9), loc='upper left')
    flux_comparison_ax.set_yscale('log')
    flux_comparison_ax.set_ylabel('Flux')
    flux_comparison_ax.set_xticks([])
    flux_comparison_fig.tight_layout()
    flux_comparison_fig.savefig(f'{restart_directory}/hamsm_vs_direct_flux_comparison_plot.pdf', bbox_inches="tight")

generate_plots(model, "./")
model.get_committor()
model.plot_committor()
