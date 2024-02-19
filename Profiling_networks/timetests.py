"""
Code used for obtaining inference times from DistillNet
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import pandas as pd
import os
import torch.utils.data as data
from tqdm import tqdm
import sys
import timeit

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from distillnet_setup import load_bestmodel
from data_helpers import fl_inputs, gettraindata, join_and_makedir
from distillnet_config import hparams, trainparams, dirs, bool_val


def time_forwardpass(model, dataloader, batch_size, num_eventparticles):
    """
    Time forwardpass for DistillNet for given variable batch of particles.
    """
    model.eval()
    event_time_mod_list = []
    with torch.no_grad():
        for i, features in tqdm(enumerate(dataloader)):
            t1 = timeit.default_timer()
            _ = model(features)
            t2 = timeit.default_timer()
            time_per_batch = t2 - t1
            time_per_particle = time_per_batch / batch_size
            event_time_mod = time_per_particle * num_eventparticles
            event_time_mod_list.append(event_time_mod)

    return event_time_mod_list


class FeatureDataset(Dataset):  # create Dataset object for Dataloader to iterate over
    def __init__(self, data, transform=None, target_transform=None):
        # define traindata and truth labels
        alldata, labelNN = data[0], data[1]
        # transform feature_inputvector to torch tensor
        self.ftensor = torch.tensor(alldata).float()
        # transform truth_weight to torch tensor
        self.NNweight = torch.tensor(labelNN).float()

    # define iterator for dataloader, returns the inputvector and truth_value
    def __getitem__(self, index):
        return self.ftensor[index]  # , self.NNweight[index]

    def __len__(self):
        return self.NNweight.size(0)

    def numfeatures(
        self,
    ):  # get length if input vector
        return len(self.ftensor[0])


flist_inputs = [member.value for member in fl_inputs]

device = "cpu"
minbatchsize = 0
maxbatchsize = 19
testsize_total = 1000000
savedir = dirs["savedir"]  # Directory where results like plots and small files are saved
saveinfo = f"trainpart_{hparams['maketrain_particles']:.2E}__Batchs_{hparams['batch_size']}__numep_{trainparams['n_epochs']}__wgt_{trainparams['weightedlossval']}"
if bool_val["is_min_max_scaler"]:
    saveinfo += "_minmaxscaler"
if bool_val["is_standard_scaler"]:
    saveinfo += "_stdscaler"
print(saveinfo)
filedir = dirs["filedir"]  # directory for input samples
w_sample = "distill_wjets_emd_prl.h5"  # specific input sample
modelsavedir = join_and_makedir(savedir, "Models/")

model = load_bestmodel(
    saveinfo,
    modelsavedir,
    "bestmodel_trainloss",
    device,
    16,
    128,
    64,
    1,
)


feat, abc, nevents = gettraindata(
    filedir, w_sample, flist_inputs, filedir, Is_makeplots=False, Is_standard=True, Is_dtrans=False
)


dataset_features = FeatureDataset((feat[0:testsize_total], abc[0:testsize_total]))


batchlist = 2 ** np.arange(
    minbatchsize, maxbatchsize, 1
)  
print(batchlist)
times_batches = []
times_batches_err = []
for batch_size_scan in batchlist:
    dataset_loader = data.DataLoader(
        dataset=dataset_features, shuffle=False, batch_size=int(batch_size_scan), drop_last=True
    )
    batch_timelist = time_forwardpass(model, dataset_loader, batch_size_scan, 9000)
    batch_timelist = np.multiply(batch_timelist, 1000)  # from s to ms
    print(batch_size_scan)
    times_mean = np.mean(batch_timelist)
    varcalc = np.std(batch_timelist, ddof=1) / np.sqrt(len(batch_timelist))  # get error
    times_batches.append(times_mean)
    times_batches_err.append(varcalc)
    print("ms time Event", times_mean)
    print("error", varcalc)


timedata = {
    "batchlist_pow2": batchlist,
    "times_pow2": times_batches,
    "times_pow2_err": times_batches_err,
}
df = pd.DataFrame.from_dict(timedata, orient="index")
h5_save_dir = savedir + "timetest_results/"
if not os.path.isdir(h5_save_dir):
    os.makedirs(h5_save_dir)
h5_str = "timetests_batchscan.h5"
df.to_hdf(h5_save_dir + h5_str, key="timeresults", mode="w")
