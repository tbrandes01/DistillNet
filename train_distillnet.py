import torch
from torch.utils.data import Dataset
import torch.utils.data as data
import torch.nn as nn
import h5py
import os
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import vector
from tqdm import tqdm
import time
from data_helpers import (fl_inputs, fl_vecs, convertvec_etaphipt, gettraindata, make_lossplot,
                        make_histoweight, make_histoweight_mod, make_metplots, join_and_makedir)
from distillnet_setup import makedataloaders, nn_setup, validation, do_training, load_bestmodel
from distillnet_config import hparams, trainparams, bool_val
from calc_met import get_mets, resolution, genfunc, make_resolutionplots
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
matplotlib.rc("font", size=22, family="serif")
matplotlib.rcParams["text.usetex"] = True

filedir = #specify where the downloaded sample is
savedir = './Results/'

modelsavedir = join_and_makedir(savedir, 'Models/')
plotdir = join_and_makedir(savedir, 'Plots/')
plotdir_pdf = join_and_makedir(plotdir, 'pdf/')
scalerdir = join_and_makedir(savedir, 'scalers/')

timestr = time.strftime("%Y%m%d-%H%M%S")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use gpu
print("Device:", device)

flist_inputs = [member.value for member in fl_inputs]
flist_names = [fl_inputs(i).name for i in flist_inputs]
print(flist_names)
print(len(flist_inputs))

saveinfo = f"_trainpart_{hparams['maketrain_particles']:.2E}__Batchs_{hparams['batch_size']}__numep_{trainparams['n_epochs']}_7_3_bnL4_werr3_nopup"

is_displayplots = bool_val['is_displayplots']
is_savefig = bool_val['is_savefig']
is_remove_padding = bool_val['is_remove_padding']
is_min_max_scaler = bool_val['is_mix_max_scaler']
is_standard_scaler = bool_val['is_standard_scaler']
is_dtrans = bool_val['is_dtrans']
is_do_taylor = bool_val['is_do_taylor']
is_weighted_error = bool_val['is_weighted_error']
if is_min_max_scaler:
    saveinfo += "_minmaxscaler"
if is_standard_scaler:
    saveinfo += "_stdscaler"
#MODIFIED LOADING PROCEDURE

def main():
    print(saveinfo)
    nn_inputdata = gettraindata(filedir, trainparams['train_sample'], trainparams['test_sample'], flist_inputs, scalerdir, is_dtrans=is_dtrans, is_standard=True, is_remove_padding=is_remove_padding,
                                is_min_max_scaler=is_min_max_scaler, is_standard_scaler=is_standard_scaler, is_makeplots=False)
    model, criterion, optimizer, train_loader, test_loader, test, input_size, weights_highval = nn_setup(nn_inputdata, device, hparams['batch_size'],
                                                                                                        hparams['maketrain_particles'], hparams['L1_hsize'],
                                                                                                        hparams['L2_hsize'], hparams['n_outputs'],
                                                                                                        )
    print('Model hyperparams ', hparams)
    print('Model trainparams ', trainparams)
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    weights_highval = 3
    model, losslist, validationloss = do_training(model, criterion, optimizer, device, train_loader, test_loader, test, savedir,
                                                modelsavedir, saveinfo, weights_highval, trainparams['n_epochs'], is_dotaylor=is_do_taylor,
                                                is_weighted_error=is_weighted_error)

    make_lossplot(losslist, validationloss, plotdir, plotdir_pdf, saveinfo, timestr, is_savefig=is_savefig, is_displayplots=is_displayplots)
    met_model = load_bestmodel(saveinfo, savedir, modelsavedir, 'bestmodel_trainloss', device, input_size, hparams['L1_hsize'], hparams['L2_hsize'],
                                hparams['n_outputs'])

    distill_wgts, abc_wgts, puppi_wgts, met_d, met_a, met_p, met_g = [], [], [], [], [], [], []
    maxevent = int(nn_inputdata[3])
    print("Maxevent", maxevent)
    minevent = int(hparams['maketrain_particles'] / 9000)
    print("Minevent", minevent)
    for i in tqdm(range(minevent, maxevent)):
        pred, abc, puppi, met_distill, met_abc, met_puppi, met_gen = get_mets(filedir, trainparams['test_sample'], flist_inputs, met_model, device, i, hparams['maketrain_particles'],
                                                                        is_min_max_scaler=is_min_max_scaler, is_standard_scaler=is_standard_scaler, is_dtrans=is_dtrans)
        distill_wgts.append(pred)
        abc_wgts.append(abc)
        puppi_wgts.append(puppi)
        met_d.append(met_distill)
        met_a.append(met_abc)
        met_p.append(met_puppi)
        met_g.append(met_gen)

    resolution_model, resolution_abc, resolution_puppi = resolution(met_d, met_g), resolution(met_a, met_g), resolution(met_p, met_g)

    print("Resolution DistillNet", resolution_model)
    print("Resolution AbcNet", resolution_abc)
    print("Resolution Puppi", resolution_puppi)
    make_resolutionplots(met_a, met_p, met_d, met_g, plotdir, saveinfo, timestr, is_displayplots=is_displayplots)
    last_bin_ratio = make_histoweight_mod(distill_wgts, abc_wgts, puppi_wgts, resolution_model, resolution_abc,
                                        resolution_puppi, plotdir, plotdir_pdf, saveinfo, timestr, trainparams['test_sample'], is_displayplots=is_displayplots)
    make_metplots(met_a, met_p, met_d, resolution_abc, resolution_model, resolution_puppi, plotdir, plotdir_pdf, saveinfo, timestr, is_savefig=is_savefig, is_displayplots=is_displayplots)
    print(f"Last bin ratio {last_bin_ratio*100:.2f} %")
    print('Savinfo: ', saveinfo)
    print('Inputs used for training: ', flist_names)
    print('Total number of inputs: ', len(flist_inputs))

if __name__ == "__main__":
    main()
