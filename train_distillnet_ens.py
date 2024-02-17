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
from data_helpers import fl_inputs, fl_vecs, convertvec_etaphipt, gettraindata, make_lossplot, make_histoweight, make_histoweight_mod, make_metplots
from distillnet_setup import makedataloaders, nn_setup, validation, do_training, load_bestmodel
from distillnet_config import hparams, trainparams
from calc_met import get_mets, resolution, genfunc, make_resolutionplots
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
matplotlib.rc("font", size=22, family="serif")
matplotlib.rcParams["text.usetex"] = True




def maketraining_distill(device: str, run_number: int, numtests: int, wgt: float):
    #num_epochs = 30

    filedir = '/work/tbrandes/work/data/'
    w_sample = 'distill_wjets_emd_prl.h5'
    zprime_sample = 'distill_zprime_emd_prl.h5'

    saveinfo_core = f"tpart_{hparams['maketrain_particles']:.2E}__Batchs_{hparams['batch_size']}__numep_{trainparams['n_epochs']}_75_25_bndrop005_werr{wgt}"
    #plots_dir = 'Plots/Ensembles_nopup/'
    plots_dir = 'Plots/Ensembles_pup/'

    plots_ensdir = plots_dir + saveinfo_core + '/'

    savedir = '/work/tbrandes/work/Delphes_samples/'
    #modelsavedir = os.path.join(savedir, f'Models_Ensemble_nopup/wgt{wgt}/')
    modelsavedir = os.path.join(savedir, f'Models_Ensemble_pup/wgt{wgt}/')

    if not os.path.isdir(modelsavedir):
        os.makedirs(modelsavedir)
    plotdir = os.path.join(savedir, plots_ensdir)
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)
    plotdir_pdf = os.path.join(plotdir, 'pdf/')
    if not os.path.isdir(plotdir_pdf):
        os.makedirs(plotdir_pdf)
    scalerdir = os.path.join(savedir, 'scalers/')
    if not os.path.isdir(scalerdir):
        os.makedirs(scalerdir)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(device)

    flist_inputs = [member.value for member in fl_inputs]
    flist_inputs = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    flist_names = [fl_inputs(i).name for i in flist_inputs]
    print(flist_names)
    print(len(flist_inputs))
    saveinfo = '_' + saveinfo_core + f'__ensemble{run_number}__device{device}__numtests{numtests}'
    Is_displayplots = False
    Is_savefig = True
    Is_remove_padding = True
    Is_min_max_scaler = False
    Is_standard_scaler = True
    Is_dtrans = False
    Is_do_taylor = False
    Is_weighted_error = True
    if Is_min_max_scaler:
        saveinfo = saveinfo + "_minmax"
    if Is_standard_scaler:
        saveinfo = saveinfo + "_std"
    nn_inputdata = gettraindata(filedir, w_sample, flist_inputs, scalerdir, Is_dtrans=Is_dtrans, Is_standard=True, Is_remove_padding=Is_remove_padding,
                                Is_min_max_scaler=Is_min_max_scaler, Is_standard_scaler=Is_standard_scaler, Is_makeplots=False)
    print(saveinfo)
    model, criterion, optimizer, train_loader, test_loader, test, input_size, weights_highval = nn_setup(nn_inputdata, device, hparams['batch_size'],
                                                                                                        hparams['maketrain_particles'], hparams['L1_hsize'],
                                                                                                        hparams['L2_hsize'], hparams['n_outputs'])
    print('Model hyperparams ', hparams)
    print('Model trainparams ', trainparams)
    weights_highval = wgt
    model, losslist, validationloss = do_training(model, criterion, optimizer, device, train_loader, test_loader, test, savedir,
                                                modelsavedir, saveinfo, weights_highval, trainparams['n_epochs'], Is_dotaylor=Is_do_taylor, Is_weighted_error=Is_weighted_error)

    make_lossplot(losslist, validationloss, plotdir, plotdir_pdf, saveinfo, timestr, Is_savefig=Is_savefig, Is_displayplots=Is_displayplots)
    #last_bin_ratio = make_histoweight(test, test_loader, model, device, plotdir, plotdir_pdf, saveinfo, timestr, Is_savefig, Is_displayplots)
    #print(f"Last bin ratio {last_bin_ratio*100:.2f} %")
    met_model = load_bestmodel(saveinfo, savedir, modelsavedir, 'bestmodel_trainloss', device, input_size, hparams['L1_hsize'], hparams['L2_hsize'], hparams['n_outputs'])

    distill_wgts, abc_wgts, puppi_wgts, met_d, met_a, met_p, met_g = [], [], [], [], [], [], []
    maxevent = int(nn_inputdata[2])
    print("Maxevent", maxevent)
    minevent = int(hparams['maketrain_particles'] / 9000)
    print("Minevent", minevent)
    for i in tqdm(range(minevent, maxevent)):
        pred, abc, puppi, met_distill, met_abc, met_puppi, met_gen = get_mets(filedir, w_sample, flist_inputs, met_model, device, i, hparams['maketrain_particles'],
                                                                        Is_min_max_scaler=Is_min_max_scaler, Is_standard_scaler=Is_standard_scaler, Is_dtrans=Is_dtrans)
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
    make_resolutionplots(met_a, met_p, met_d, met_g, plotdir, saveinfo, timestr, Is_displayplots=Is_displayplots)
    last_bin_ratio = make_histoweight_mod(distill_wgts, abc_wgts, puppi_wgts, resolution_model, resolution_abc,
                                        resolution_puppi, plotdir, plotdir_pdf, saveinfo, timestr,w_sample, Is_displayplots=Is_displayplots)
    make_metplots(met_a, met_p, met_d, resolution_abc, resolution_model, resolution_puppi, plotdir, plotdir_pdf, saveinfo, timestr, Is_savefig=True, Is_displayplots=Is_displayplots)

    print(f"Last bin ratio {last_bin_ratio*100:.2f} %")
    print(saveinfo)
  #  metres = np.asarray([met_a, met_p, met_g, met_d])
  #  np.savetxt(savedir + 'met_results/' + saveinfo + "first_try_met.csv", metres, delimiter=",")
  #  distill_wgts, abc_wgts, puppi_wgts = np.concatenate(distill_wgts), np.concatenate(abc_wgts).squeeze(), np.concatenate(puppi_wgts).squeeze()
    #distilres = np.asarray([distill_wgts, abc_wgts, puppi_wgts])
    #np.savetxt(savedir + 'met_results/' + saveinfo + "first_try_wgts.csv", distilres, delimiter=",")
    return resolution_model

#if __name__ == "__main__":
#    main()