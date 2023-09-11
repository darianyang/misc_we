# %% [markdown]
# # Loading data

# %%
import numpy as np
from matplotlib import pyplot as plt
import h5py
import tqdm.auto

# %%
pwd

# %%
from  msm_we import msm_we
import mdtraj as md

# %%
import ray
ray.init()

# %%
#ray.shutdown()

# %%
import logging
from rich.logging import RichHandler
log = logging.getLogger()
log.addHandler(RichHandler())

msm_log = logging.getLogger("msm_we.msm_we")

# %%
def processCoordinates(self, coords):
#     log.debug("Processing coordinates")

    if self.dimReduceMethod == "none":
        xt = md.Trajectory(xyz=coords, topology=None)
        indCA = self.reference_structure.topology.select("not type H and (resid 5 to 17 or resid 23 to 35 or resid 40 to 53)")
        indAlign = self.reference_structure.topology.select("not type H and resid 40 to 53")
        x2 = xt.superpose(self.reference_structure,atom_indices=indAlign)
        coords = x2._xyz.astype('float64')
        nA = np.shape(indCA)[0]
        nC = np.shape(coords)[0]
        new_coords = coords[:,tuple(indCA),:]
        #new_coords = numpy.double(new_coords)
        #new_coords = coords[:, :923, :]
        data = new_coords.reshape(nC, 3 * nA)
        model.nAtoms = nA
        return data

    if self.dimReduceMethod == "pca" or self.dimReduceMethod == "vamp":

        # Dimensionality reduction

        xt = md.Trajectory(xyz=coords, topology=None)
        indCA = self.reference_structure.topology.select("name CA and (resid 5 to 17 or resid 23 to 35 or resid 40 to 53)")

        #indCA = self.reference_structure.topology.select("not type H and (resid 5 to 17 or resid 23 to 35 or resid 40 to 53)")
        #indCA = self.reference_structure.topology.select("name CA")
        pair1, pair2 = np.meshgrid(indCA, indCA, indexing="xy")
        indUT = np.where(np.triu(pair1, k=1) > 0)
        pairs = np.transpose(np.array([pair1[indUT], pair2[indUT]])).astype(int)
        dist = md.compute_distances(xt, pairs, periodic=True, opt=True)

        return dist
    
msm_we.modelWE.processCoordinates = processCoordinates

# %% [markdown]
# ## Build block-wise haMSMs

# %% [markdown]
# Build the model

# %%
# 'uneven_bin' for zero2 (and 3)
import westpa
from westpa.core.binning import RectilinearBinMapper
import numpy as np

new_mapper = RectilinearBinMapper([[ 0.  ,  2.  ,  2.4 , 2.45, 2.5725, 2.75,2.825, 2.9,  3.05  ,
                                    3.1, 3.15,3.2, 3.25,3.3,3.35,3.4,
                                    3.5,  3.6, 3.7 , 3.75, 3.8 ,  3.85,  3.9 ,  3.95,  4.  , 4.05,
                                    4.1 , 4.15, 4.2 , 4.25, 4.3 ,  4.4 ,  4.5 ,  4.6 ,  4.7 ,  4.8 ,  4.9 ,
                                    5.  ,  5.25,  5.5 ,  5.75,  6.  ,  6.25,  6.5 ,  6.75,  7.  ,
                                    7.5 , 8,9,10,11, 12.5,15, 21.  ,   np.inf]])

# %%
h5_glob = ['restart3_scikit_nowat/run1/west.h5','restart3_scikit_nowat/run2/west.h5','restart3_scikit_nowat/run3/west.h5',
           'restart4_scikit_nowat/run1/west.h5','restart4_scikit_nowat/run2/west.h5','restart4_scikit_nowat/run3/west.h5',
           'restart5_scikit_nowat/run1/west.h5','restart5_scikit_nowat/run2/west.h5','restart5_scikit_nowat/run3/west.h5',]


model =  msm_we.modelWE()
model.initialize(
    fileSpecifier=h5_glob, 
    refPDBfile = 'wsh2029_eq3_noions.pdb',
    modelName = 'BdpA_wsh2029_p3',
    target_pcoord_bounds = [[0,2.5]],
    #basis_pcoord_bounds  = [[12.5,np.inf]],
    basis_pcoord_bounds  = [[15,np.inf]],
    dim_reduce_method = 'vamp',
    tau = 1e-10,
)

model.get_iterations()
model.get_coordSet(model.maxIter)
#model.get_traj_coordinates(from_iter=1, traj_le)

# %%
model.maxIter

# %%
model.target_pcoord_bounds

# %% [markdown]
# PCA on all the data

# %%
logging.getLogger('msm_we.msm_we').setLevel('DEBUG')

# %%
model.dimReduce()

# %%
# For loading the clustered model data
from pickle import load
with open('NEW-MSMWE_clustering_scikit_c400_uneven_bin_heavy_step0-restart345-t3.pickle', 'rb') as fo:
    model = load(fo)

# %%
# For saving the clustered model data
from pickle import dump
with open('NEW-MSMWE_clustering_scikit_c400_uneven_bin_heavy_step0-restart345-t3.pickle', 'wb') as fo:
    dump(model, fo)

# %% [markdown]
# Cluster on all the data, and store the original clustering.

# %%
# Fresh clustering
model.cluster_coordinates(n_clusters=400, tol = 1e-5, streaming=True, use_ray=True, user_bin_mapper=new_mapper, store_validation_model=True)

# %%
# For loading the clustered model data
from pickle import load
with open('NEW-MSMWE_clustering_scikit_c400_uneven_bin_heavy_step1-restart345-t3.pickle', 'rb') as fo:
    model = load(fo)

# %%
# For saving the clustered model data
from pickle import dump
with open('NEW-MSMWE_clustering_scikit_c400_uneven_bin_heavy_step1-restart345-t3.pickle', 'wb') as fo:
    dump(model, fo)

# %%
model.n_clusters

# %%
model.get_fluxMatrix(0,1,300,use_ray=True)

# %%
# For loading the clustered model data
from pickle import load
with open('NEW-MSMWE_clustering_scikit_c200_uneven_bin_heavy_step2-restart345-t3.pickle', 'rb') as fo:
    model = load(fo)

# %%
# For saving the clustered model data
from pickle import dump
with open('NEW-MSMWE_clustering_scikit_c400_uneven_bin_heavy_step2-restart345-t3.pickle', 'wb') as fo:
    dump(model, fo)

# %%
model.organize_fluxMatrix(use_ray=True)

# %%
# For loading the clustered model data
from pickle import load
with open('NEW-MSMWE_clustering_scikit_c200_uneven_bin_heavy_step3-restart345-t2.pickle', 'rb') as fo:
    model = load(fo)
    
# NEW-MSMWE_clustering_scikit_c200_uneven_bin_heavy_step3-restart345-t2.pickle

# %%
# For saving the clustered model data
from pickle import dump
with open('NEW-MSMWE_clustering_scikit_c400_uneven_bin_heavy_step3-restart345-t3.pickle', 'wb') as fo:
    dump(model, fo)

# %%
model.fluxMatrix

# %%
model.get_Tmatrix()

# %%
model.Tmatrix

# %%
model.get_steady_state()

# %%
# For checking just in case your bins are not good.
array = np.squeeze(model.targetRMSD_minmax, axis=1)
print(array[23],array[142],array[132],array[142],array[148],array[151])

# %%
model.update_cluster_structures()

# %%
model.get_steady_state_target_flux()

print(f'Steady-state target rate is {model.JtargetSS:.2e}')

# %% [markdown]
# # Validation

# %%
array = np.squeeze(model.targetRMSD_minmax, axis=1)
array

# %%
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 10]
# len-2 because the last two are NaN (eq simulation; target/basis states)
for idx in range(0,len(array)):
    plt.errorbar(idx, array[idx,0],yerr=[[0],[array[idx,1]-array[idx,0]]])
    #plt.plot(idx, array[idx,1], 'ro')
plt.xlabel('cluster number')
plt.ylabel('RMSD to target state')
plt.title('MSM Bins Coverage Versus Progress Coordinates')

# %%
model.plot_flux(suppress_validation=True)

# %%
model.get_committor()

# %%
model.plot_committor()

# %%
model.targetRMSD_centers[:,0]

# %%
model.q

# %%
model.plot_flux_committor(suppress_validation=True)
plt.gca().set_xscale('linear')
#plt.savefig('flux_pseudo_uneven_bin-restart34-t2-c18.png')

# %%
# For loading the clustered model data
from pickle import load
with open('NEW-MSMWE_clustering_scikit_c200_uneven_bin_heavy_step4-restart345-t2.pickle', 'rb') as fo:
    model = load(fo)

# %%
## For saving the clustered model data
from pickle import dump
with open('NEW-MSMWE_clustering_scikit_c400_uneven_bin_heavy_step4-restart345-t3.pickle', 'wb') as fo:
    dump(model, fo)

# %% [markdown]
# # Post-Featurization Steps

# %%
model.do_block_validation(cross_validation_groups=2, cross_validation_blocks=8)#, skip=[1])

# %%
from copy import deepcopy
do_block_validation(self=model, cross_validation_groups=2, cross_validation_blocks=8, use_ray=True)

# %%
# For loading the clustered model data
from pickle import load
with open('clustering_scikit_c18_uneven_bin_heavy_step5-restart34-zero7.pickle', 'rb') as fo:
    model = load(fo)

# %%
# For saving the clustered model data
from pickle import dump
with open('NEW-MSMWE_clustering_scikit_c200_uneven_bin_heavy_step5-restart345-t3.pickle', 'wb') as fo:
    dump(model, fo)

# %%


# %% [markdown]
# # Block Validation Rates

# %%
for i in range(len(model.validation_models)):
    model.validation_models[i].get_steady_state_target_flux()

    print(f'Steady-state target rate is {model.validation_models[i].JtargetSS:.2e} for model {i}')

# %%
import numpy
list_of_rates = [model.JtargetSS, model.validation_models[0].JtargetSS, model.validation_models[1].JtargetSS]
print(numpy.std(list_of_rates))
print(numpy.average(list_of_rates))

# %%
for i in range(len(model.validation_models)):
    model.validation_models[i].get_steady_state_target_flux()

    print(f'Steady-state target rate is {model.validation_models[i].JtargetSS:.2e} for model {i}')

# %%
import numpy
list_of_rates = [model.JtargetSS, model.validation_models[0].JtargetSS, model.validation_models[1].JtargetSS]
print(numpy.std(list_of_rates))
print(numpy.average(list_of_rates))

# %%
ray.shutdown()

# %% [markdown]
# # Block Validation Output

# %%
import weight_loop
for idx, v_model in enumerate(model.validation_models):
    if idx == 0:
        continue
    weight_loop.create_reweighted_h5_global(v_model,new_name=f'west_reweight_c10_block_to20_v2-try2_v{idx}.h5')

# %% [markdown]
# # Main model Output

# %%
import weight_loop

weight_loop.create_reweighted_h5_global(model, west_name='west_nocoords.h5', copy=True, struct_filetype='ncrst', new_name=f'west_reweight.h5', gen_sstates=True, pdb_out=False, link_out=True, link_path="/ocean/projects/mcb180038p/jml230/bdpa_wsh2029_p3_r3",)

# %%
import weight_loop

weight_loop.create_reweighted_h5_global(model, west_name='west_nocoords.h5', copy=True, struct_filetype='ncrst', new_name=f'west_restart345_reweight-vamp.h5', gen_sstates=False, pdb_out=False, link_out=False, link_path="/ocean/projects/mcb180038p/jml230/bdpa_wsh2029_p3_r3",)

# %%


# %% [markdown]
# # Extract transition state (pseudo-committor of a certain range)

# %%
# Parsing through the list so we get the real seg number (and the file number as well)
import h5py
from copy import deepcopy
import numpy

def output_pseudocommittor(lower, upper, both):
    # Finding all the iter/seg_index
    output = []
    for j in range(0, len(model.dtrajs)):
        val = numpy.argwhere(numpy.isin(model.dtrajs[j], both))
        #print(j+1, numpy.squeeze(val,axis=1))
        output.append([j+1, numpy.squeeze(val, axis=1)])
    output2 = [[val[0], seg] for val in output for seg in val[1]]
    #output2

    file_list = []
    for file in model.fileList:
        with h5py.File(file,'r') as f:
            file_list.append(f['summary']['n_particles'])

    output3 = deepcopy(output2)

    for j in output3:
        seg_index = j[1]
        tol = 0
        for i in range(0,len(file_list)):
            tol += file_list[i][j[0]-1]
            #print(i, tol, seg_index)
            if seg_index < tol:
                #print('break')
                break
            #print(tol)
        if i != 0: 
            #print('here')
            tol -= file_list[i][j[0]-1]
            j[1] -= tol
        else:
            j[1] = seg_index
        j.insert(0,i+1)

    output3 = numpy.asarray(output3)
    with open(f'transition_state/validation_models/transition_state_a_{lower}to{upper}.npy','wb') as fo:
        numpy.save(fo,output3)
    output3

# %%
import h5py
from copy import deepcopy
import numpy as np
import numpy
from tqdm.auto import tqdm
ranges = [(0,0.2), (0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1), (0,0.1),(0.1,0.2),(0.8,0.9),(0.9,1), (0,0.3), (0.3,0.6),(0.6,1),(0.45,0.55)]
#ranges = [(0.4,0.6),(0.45,0.55)]
#ranges = [(0.35,0.45)]

# specify upper, lower range.
for val in tqdm(ranges):
    under = np.argwhere(model.q < val[1])
    over = np.argwhere(model.q > val[0])

    both = [i for [i] in under if i in over]
    both
    output_pseudocommittor(val[0],val[1],both)

# %%
# Parsing through the list so we get the real seg number (and the file number as well)
import h5py
from copy import deepcopy
import numpy

def output_validation_pseudocommittor(lower, upper, both, x):
    # Finding all the iter/seg_index
    output = []
    for j in range(0, len(model.validation_models[x].dtrajs)):
        val = numpy.argwhere(numpy.isin(model.validation_models[x].dtrajs[j], both))
        #print(j+1, numpy.squeeze(val,axis=1))
        output.append([j+1, numpy.squeeze(val, axis=1)])
    #print(output)
    output2 = [[val[0], seg] for val in output for seg in val[1] if val[0] in model.validation_iterations[x]]
    print(output2)

    file_list = []
    for file in model.fileList:
        with h5py.File(file,'r') as f:
            file_list.append(f['summary']['n_particles'])

    #print(file_list)
    output3 = deepcopy(output2)

    for j in output3:
        seg_index = j[1]
        tol = 0
        for i in range(0,len(file_list)):
            tol += file_list[i][j[0]-1]
            #print(i, tol, seg_index)
            if seg_index < tol:
                #print('break')
                break
            #print(tol)
        if i != 0: 
            #print('here')
            tol -= file_list[i][j[0]-1]
            j[1] -= tol
        else:
            j[1] = seg_index
        j.insert(0,i+1)

    output3 = numpy.asarray(output3)
    with open(f'transition_state/validation_models/transition_state_validation{x}_{lower}to{upper}.npy','wb') as fo:
        numpy.save(fo,output3)
    output3

# %%
import h5py
from copy import deepcopy
import numpy as np
import numpy
from tqdm.auto import tqdm,trange
ranges = [(0,0.2), (0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1), (0,0.1),(0.1,0.2),(0.8,0.9),(0.9,1), (0,0.3), (0.3,0.6),(0.6,1),(0.45,0.55)]
#ranges = [(0.4,0.6),(0.45,0.55),(0.35,0.45)]
# ranges = [(0.45,0.55)]

# specify upper, lower range.
for j in trange(len(model.validation_models)):
    for val in tqdm(ranges):
        under = np.argwhere(model.validation_models[j].q < val[1])
        over = np.argwhere(model.validation_models[j].q > val[0])

        both = [i for [i] in under if i in over]
        print(both)
        output_validation_pseudocommittor(val[0],val[1],both,j)

# %%


# %%


# %%


# %%
# OLDER VERSIONS

# %%
import h5py
from copy import deepcopy
import numpy as np
import numpy
from tqdm.auto import tqdm
ranges = [(0.45,0.55),(0,0.2), (0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1), (0,0.1),(0.1,0.2),(0.8,0.9),(0.9,1), (0,0.3), (0.3,0.6),(0.6,1)]

# specify upper, lower range.
for val in tqdm(ranges):
    under = np.argwhere(model.q < val[1])
    over = np.argwhere(model.q > val[0])

    both = [i for [i] in under if i in over]
    both
    output_pseudocommittor(val[0],val[1],both)

# %%
ray.shutdown()

# %%
# Parsing through the list so we get the real seg number (and the file number as well)
import h5py
from copy import deepcopy
import numpy

def output_pseudocommittor(lower, upper, both):
    # Finding all the iter/seg_index
    output = []
    for j in range(0, len(model.dtrajs)):
        val = numpy.argwhere(numpy.isin(model.dtrajs[j], both))
        #print(j+1, numpy.squeeze(val,axis=1))
        output.append([j+1, numpy.squeeze(val, axis=1)])
    output2 = [[val[0], seg] for val in output for seg in val[1]]
    #output2

    file_list = []
    for file in model.fileList:
        with h5py.File(file,'r') as f:
            file_list.append(f['summary']['n_particles'])

    output3 = deepcopy(output2)

    for j in output3:
        seg_index = j[1]
        tol = 0
        for i in range(0,len(file_list)):
            tol += file_list[i][j[0]-1]
            #print(i, tol, seg_index)
            if seg_index < tol:
                #print('break')
                break
            #print(tol)
        if i != 0: 
            #print('here')
            tol -= file_list[i][j[0]-1]
            j[1] -= tol
        else:
            j[1] = seg_index
        j.insert(0,i+1)

    output3 = numpy.asarray(output3)
    with open(f'transition_state/transition_state_{lower}to{upper}.npy','wb') as fo:
        numpy.save(fo,output3)
    output3

# %%
# Double check to see that I didn't do something odd
numpy.argwhere(output3<0)

# %%


# %% [markdown]
# # Plotting and Stuff

# %%
# Lets get the list of file all here
listf = []
for idx in range(1,6):
    #listf.append(f"restart0_scikit_nowat/run{idx}/west_reweight_c10_block_to20_v2-try2_v0.h5")
    listf.append(f"restart0_scikit_nowat/run{idx}/west_reweight_c10_block_to20_v2-try2_v1.h5")
    #listf.append(f"restart0_scikit_nowat/run{idx}/west_reweight_c10_block_to20_v2.h5")
    
file_path = listf[0].rsplit('.',maxsplit=1)[0]
print(file_path)

# %%
# Generating the cool text file
import h5py
import numpy
from tqdm.auto import tqdm

aux = 'RoG'

with open(f'{file_path}_RoG_iter.dat', 'w') as g:
    for idx, stuff in enumerate(listf):
        f = h5py.File(stuff, 'r')
        for i in tqdm(range(1,301), desc=f"File {idx+1}"): # change indices to number of iteration
            i = str(i)
            iteration = "iter_" + str(numpy.char.zfill(i,8))
            s = f['iterations'][iteration]['seg_index'].shape[0]
            r1 = f['iterations'][iteration]['auxdata'][aux][:,-1] # These are the auxillary coordinates you're looking for
            r2 = f['iterations'][iteration]['seg_index']['weight'][:] # These are the auxillary coordinates you're looking for
            for j in range(0,s):
                array1=[]
                array1 = r2 # Weights normalized across whole simulation
                g.write(str(array1[j]) + "\t" + str(r1[j]) + "\n")

# %%
# Loading Data
import numpy
block_new0_wt, block_new0_data = numpy.loadtxt('restart0_scikit_nowat/run1/west_reweight_c10_block_to20_v2_v0_RoG_iter.dat',usecols=(0,1), unpack=True)
block_new1_wt, block_new1_data = numpy.loadtxt('restart0_scikit_nowat/run1/west_reweight_c10_block_to20_v2-try2_v1_RoG_iter.dat',usecols=(0,1), unpack=True)
block_main_wt, block_main_data = numpy.loadtxt('restart0_scikit_nowat/run1/west_reweight_c10_block_to20_v2_RoG_iter.dat',usecols=(0,1), unpack=True)

global_wt, global_data = numpy.loadtxt('restart0_scikit_nowat/run1/wsh2045_reweight_global_rmsheavy_iter.dat',usecols=(0,1), unpack=True)
raw_wt, raw_data = numpy.loadtxt( 'restart0_scikit_nowat/run1/wsh2045_reweight_rmsheavy_iter.dat',usecols=(0,1), unpack=True)

# %%
block_new0b_wt, block_new0b_data = numpy.loadtxt('restart0_scikit_nowat/run1/west_reweight_c10_block_v0b_mod_rmsheavy_iter.dat',usecols=(0,1), unpack=True)
block_new1b_wt, block_new1b_data = numpy.loadtxt('restart0_scikit_nowat/run1/west_reweight_c10_block_v1b_mod_rmsheavy_iter.dat',usecols=(0,1), unpack=True)

# %%
# Colors and Labels and other parameters
import matplotlib
import matplotlib.pyplot as plt
wsh2045color = (0,0,0,1)
wsh2029color = (0.867,0.317,0.5098,1)
wsh2036color = (1,0.431,0.32994,1)
wsh2057color = (1,0.651,0,1)
wsh2044color = (0.267,0.306,0.525,1)
wsh2054color = (0.584,0.3176,0.588,1)
colors = [wsh2045color, wsh2054color, wsh2029color, wsh2057color, wsh2044color, wsh2054color]
#data = [raw_wt, global_wt, block_main_wt, block_new0_wt, block_new1_wt]
plt.rcParams.update({'figure.figsize': [4.5,4.5], 'font.size': 17.5, 'figure.dpi': 300, 'font.family': 'Arial',
                     'ytick.major.width': 2, 'xtick.major.width': 2, 'axes.linewidth': 2})
labels = [u'Raw', u'Old, Global Aggregated', u'New Stratified Model', u'New Stratified Validation Model 0', u'New Stratified Validation Model 1']
transparency=[1,1,0.25,0.5, 0.5]

# %%
# For the Legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=colors[1], alpha=transparency[1], lw=4),
                Line2D([0], [0], color=colors[2], alpha=transparency[0], lw=4),
                Line2D([0], [0], color=colors[3], alpha=transparency[3], lw=4),
                Line2D([0], [0], color=colors[4], alpha=transparency[4], lw=4),
                Line2D([0], [0], color=colors[0], alpha=transparency[0], lw=4),]

labels = [u'all data',
          'block 1',
          u'block 2',
          'run 4',
          u'run 5']

# %%
# Everything
for i,dataset in enumerate(data):
    plt.hist(dataset,bins=100, color=colors[i], range=(0,0.001), histtype='bar', log=True, alpha=transparency[i], label=labels[i])
plt.legend()
plt.title('Distribution of Weights');
plt.savefig('new_stratified_c10_plots/all_weight_dist.png',dpi=300)

# %%
# Before Reweighting
plt.hist(data[0], bins=300, color=colors[0], range=(0,0.001),log=True)
plt.title(labels[0]);
plt.xlabel('Weights')
plt.ylabel('Counts')
plt.savefig('new_stratified_c10_plots/raw_weight_dist.png', dpi=300)

# %%
# Global Reweighting (Way before Stratified)
plt.hist(data[1], bins=300, color=colors[1], range=(0,0.001), log=True)
plt.title(labels[1])
plt.xlabel('Weights')
plt.ylabel('Counts')
plt.savefig('new_stratified_c10_plots/old_global_dist.png',dpi=300)

# %%
# New Stratified Main Model Reweighting
plt.hist(data[2], bins=300, color=colors[2], range=(0,0.001), log=True)
plt.title(labels[2])
plt.xlabel('Weights')
plt.ylabel('Counts')
plt.savefig('new_stratified_c10_plots/new_main_model_dist.png',dpi=300)

# %%
# New Stratified Validation Model 0 Reweighting
plt.hist(data[3], bins=300, color=colors[3], range=(0,0.001), log=True)
plt.title(labels[3])
plt.xlabel('Weights')
plt.ylabel('Counts')
plt.savefig('new_stratified_c10_plots/new_model_v0_dist.png',dpi=300)

# %%
# New Stratified Validation Model 1 Reweighting
plt.hist(data[4], bins=300, color=colors[4], range=(0,0.001), log=True)
plt.title(labels[4])
plt.xlabel('Weights')
plt.ylabel('Counts')
plt.savefig('new_stratified_c10_plots/new_model_v1_dist.png',dpi=300)

# %%
# Block Reweighting (Main Model + Validation Sets 0 + 1)
plt.hist(block_main_wt, bins=300, color=colors[0], range=(0,0.001), log=True, label=labels[2], 
         alpha=transparency[3])
plt.hist(block_new0_wt, bins=300, color=colors[3], range=(0,0.001), log=True, label=labels[3], 
         alpha=transparency[3])
plt.hist(block_new1_wt, bins=300, color=colors[4], range=(0,0.001), log=True, label=labels[4], 
         alpha=transparency[4])
plt.legend()
plt.xlabel('probability weights')
plt.ylabel('counts')
plt.title(u'Block Reweighted (Main + Validation Sets 0 + 1)');
#plt.savefig('new_stratified_c10_plots/new_model_all_dist.png',dpi=300)

# %%
# Block Reweighting (Everything)

plt.hist(raw_data, bins=100, color=colors[0], histtype='step', weights=raw_wt, label=labels[0])
#plt.hist(global_data, bins=100, color=colors[1], histtype='step', weights=global_wt, label=labels[1])
plt.hist(block_main_data, bins=100, color=colors[2], weights=block_main_wt, label=labels[2])
plt.hist(block_new0_data, bins=100, color=colors[3], histtype = 'step', weights=block_new0_wt, label='Validation Model 0')
plt.hist(block_new1_data, bins=100, color=colors[4], histtype = 'step', weights=block_new1_wt, label='Validation Model 1')

plt.title(u'RMSD to Folded State');
plt.legend()
plt.xlabel(u'RMSD to folded state (\u212B)')
plt.ylabel('probability weights')
plt.savefig('new_stratified_c10_plots/new_model_rmsd_heavy_dist.png',dpi=300)


# %%
# Just the main + validation models
plt.hist(block_main_data, bins=100, range=(11,18), color=colors[2], histtype = 'step', weights=block_main_wt, label=labels[2])
plt.hist(block_new0_data, bins=100, range=(11,18), color=colors[3], histtype = 'step', weights=block_new0_wt, label='Validation Model 0')
plt.hist(block_new1_data, bins=100, range=(11,18), color=colors[4], histtype = 'step', weights=block_new1_wt, label='Validation Model 1')

#plt.title(u'RMSD to Folded State, Comparison with Same Cluster Models');
#plt.legend()
#plt.xlabel(u'RMSD to folded state (\u212B)')
#plt.ylabel('probability weights')
plt.ylabel('probability')
plt.xlabel(r'$\it{Rg}$ (Å)')
plt.legend(custom_lines[1:4], labels[:3], frameon=False)
plt.tight_layout()
plt.savefig('RoG_block_comp_haMSM_runs.pdf',dpi=300)
plt.savefig('RoG_block_comp_haMSM_runs.png', dpi=300)
#plt.savefig('new_stratified_c10_plots/new_model_rmsd_heavy_main_dist.png',dpi=300)

# %%
# Just the main + validation models
plt.hist(block_main_data, bins=100, range=(11,17), color=colors[2], histtype = 'step', weights=block_main_wt, label=labels[2])
plt.hist(block_new0_data, bins=100, range=(11,17), color=colors[3], histtype = 'step', weights=block_new0_wt, label='Validation Model 0')
plt.hist(block_new1_data, bins=100, range=(11,17), color=colors[4], histtype = 'step', weights=block_new1_wt, label='Validation Model 1')

plt.title(u'RMSD to Folded State, Comparison with Same Cluster Models');
plt.legend()
plt.xlabel(u'RMSD to folded state (\u212B)')
plt.ylabel('probability weights')
#plt.savefig('new_stratified_c10_plots/new_model_rmsd_heavy_main_dist.png',dpi=300)

# %%
plt.hist(block_main_data, bins=100, color=colors[2], weights=block_main_wt, label='Main Model')
plt.hist(block_new0b_data, bins=100, color=colors[3], histtype = 'step', weights=block_new0b_wt, label='Validation Model 0b')
plt.hist(block_new1b_data, bins=100, color=colors[4], histtype = 'step', weights=block_new1b_wt, label='Validation Model 1b')

plt.title(u'RMSD to Folded State, Comparison with Reclustered Models');
plt.legend()
plt.xlabel(u'RMSD to folded state (\u212B)')
plt.ylabel('probability weights')
plt.savefig('new_stratified_c10_plots/new_model_rmsd_heavy_main_dist2.png',dpi=300)

# %%
import numpy
import matplotlib.pyplot as plt
block_main_sasa_wt, block_main_sasa = numpy.loadtxt('restart0_scikit_nowat/run1/west_reweight_c10_block_mod_totalSASA_iter.dat',usecols=(0,1), unpack=True);
block_2029_sasa_wt, block_2029_sasa = numpy.loadtxt('../bdpa_wsh2029_p2_haMSM/restart0_scikit_nowat/run1/west_reweight_c10_block_totalSASA_iter.dat',usecols=(0,1), unpack=True);

plt.hist(block_main_sasa, bins=100, color=colors[0], histtype='step', weights=block_main_sasa_wt, label='WT')
plt.hist(block_2029_sasa, bins=100, color=colors[1], histtype='step', weights=block_2029_sasa_wt, label='\u03B2\u00B3 H2')
plt.title(u'Total SASA');
plt.legend()
plt.xlabel(u'Total SASA (\u212B\u00B2)')
plt.ylabel('probability weights')
plt.savefig('new_stratified_c10_plots/new_model_totalSASA_main_dist.png',dpi=300)

# %%
import numpy
import matplotlib.pyplot as plt
block_main_RoG_wt, block_main_RoG = numpy.loadtxt('restart0_scikit_nowat/run1/west_reweight_c10_block_mod_RoG_iter.dat',usecols=(0,1), unpack=True);
block_2029_RoG_wt, block_2029_RoG = numpy.loadtxt('../bdpa_wsh2029_p2_haMSM/restart0_scikit_nowat/run1/west_reweight_c10_block_RoG_iter.dat',usecols=(0,1), unpack=True);


plt.hist(block_main_RoG, bins=100, color=colors[0], histtype = 'step', weights=block_main_RoG_wt, label='WT')
plt.hist(block_2029_RoG, bins=100, color=colors[1], histtype = 'step', weights=block_2029_RoG_wt, label=u'\u03B2\u00B3 H2')
plt.title(u'Radius of Gyration');
plt.legend()
plt.xlabel(u'RoG (\u212B)')
plt.ylabel('probability weights')
plt.savefig('new_stratified_c10_plots/new_model_RoG_main_dist.png',dpi=300)

# %%
import numpy
import matplotlib.pyplot as plt
block_main_pCH_wt, block_main_pHC = numpy.loadtxt('restart0_scikit_nowat/run1/west_reweight_c10_block_mod_percentContacts_iter.dat',usecols=(0,1), unpack=True);
block_2029_pHC_wt, block_2029_pHC = numpy.loadtxt('../bdpa_wsh2029_p2_haMSM/restart0_scikit_nowat/run1/west_reweight_c10_block_percentContacts_iter.dat',usecols=(0,1), unpack=True);

plt.hist(block_main_pHC, bins=100, color=colors[0], histtype = 'step', weights=block_main_RoG_wt, label='WT')
plt.hist(block_2029_pHC, bins=100, color=colors[1], histtype = 'step', weights=block_2029_pHC_wt, label=u'\u03B2\u00B3 H2')
plt.title(u'Percent of Native Contacts');
plt.legend()
plt.xlabel(u'Percent of Native Contacts')
plt.ylabel('probability weights')
plt.savefig('new_stratified_c10_plots/new_model_percentContacts_main_dist.png',dpi=300)

# %%
import numpy
import matplotlib.pyplot as plt
block_main_pCH_wt, block_main_pHC = numpy.loadtxt('restart0_scikit_nowat/run1/west_reweight_c10_block_to20_rms_iter.dat',usecols=(0,1), unpack=True);
block_main_pCH_wt2, block_main_pHC2 = numpy.loadtxt('restart0_scikit_nowat/run1/west_reweight_c10_block_rms_iter.dat',usecols=(0,1), unpack=True);
raw_wt, raw_data = numpy.loadtxt( 'restart0_scikit_nowat/run1/wsh2045_reweight_rmsheavy_iter.dat',usecols=(0,1), unpack=True)

plt.hist(block_main_pHC, bins=100, color=colors[0], histtype = 'step', weights=block_main_pCH_wt, label='WT-extended bins')
plt.hist(block_main_pHC2, bins=100, color=colors[1], histtype = 'step', weights=block_main_pCH_wt2, label='old WT')
plt.hist(raw_data, bins=100, color=colors[3], histtype = 'step', weights=raw_wt, label='Raw')
plt.title(u'RMSD to folded state');
plt.legend()
plt.xlabel(u'RMSD to folded state')
plt.ylabel('probability weights')
#plt.savefig('new_stratified_c10_plots/new_model_percentContacts_main_dist.png',dpi=300)

# %%
import numpy
import matplotlib.pyplot as plt
block_main_pCH_wt, block_main_rms = numpy.loadtxt('restart0_scikit_nowat/run1/west_reweight_c10_block_to20_v2_rms_iter.dat',usecols=(0,1), unpack=True);
block_main_pCH_wt, block_main_rms_0b = numpy.loadtxt('west_reweight_c10_block_to20_v2_v0b_rms_iter.dat',usecols=(0,1), unpack=True);
block_main_pCH_wt2, block_main_pHC2 = numpy.loadtxt('west_reweight_c10_block_to20_v2_v1b_rms_iter.dat',usecols=(0,1), unpack=True);
raw_wt, raw_data = numpy.loadtxt( 'restart0_scikit_nowat/run1/wsh2045_reweight_rmsheavy_iter.dat',usecols=(0,1), unpack=True)

plt.hist(block_main_pHC, bins=100, color=colors[0], histtype = 'step', weights=block_main_pCH_wt, label='WT-extended bins')
plt.hist(block_main_pHC2, bins=100, color=colors[1], histtype = 'step', weights=block_main_pCH_wt2, label='old WT')
plt.hist(raw_data, bins=100, color=colors[3], histtype = 'step', weights=raw_wt, label='Raw')
plt.title(u'RMSD to folded state');
plt.legend()
plt.xlabel(u'RMSD to folded state')
plt.ylabel('probability weights')
#plt.savefig('new_stratified_c10_plots/new_model_percentContacts_main_dist.png',dpi=300)

# %%
model.fluxMatrix

# %%
model.Tmatrix[0,1]

# %%
model.Tmatrix[1,0]


# %%
import pyemma
import msmtools
import numpy

# %%
sym_matrix = (model.Tmatrix.T + model.Tmatrix) 
sym_matrix = sym_matrix/sym_matrix.sum(axis=1, keepdims=1)
msmtools.analysis.is_transition_matrix(sym_matrix)

# %%
import pyemma
import msmtools
pcca_model = pyemma.msm.markov_model(sym_matrix,'100 ps')
pcca_model.is_reversible = True

# %%
pcca_model

# %%
nstates=10
pcca_model.pcca(nstates)

# %%
pcca_model._metastable_memberships[0]

# %% [markdown]
# # Now to test new things

# %%
import weight_loop
weight_loop.create_reweighted_h5_global(model,new_name='west_reweight_block_new.h5')

# %% [markdown]
# # TROUBLE SHOOTING

# %%
model.get_eqTmatrix()
model.Tmatrix.shape

# %%
model.indBasis, model.indTargets = np.array([]),np.array([])

# %%
model.pSS

# %%
model.Tmatrix.shape


# %%
model.Tmatrix[-1]


# %%
model.Tmatrix[0]

# %%
import numpy
a = []
for w in range(0,len(model.dtrajs)):
    a = numpy.append(a, model.dtrajs[w])

for z in range(0,131):
    if z not in a:
        print(z)


