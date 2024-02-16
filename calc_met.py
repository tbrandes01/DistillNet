import torch
from torch.utils.data import Dataset
import torch.utils.data as data
import torch.nn as nn
#import tensorflow as tf
#from tensorflow.keras.layers import Input, Dense
#from tensorflow.keras.models import Model
import h5py
import os
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import vector
from tqdm import tqdm
import time
from data_helpers import (
    fl_inputs,
    fl_vecs,
    convertvec_etaphipt,
    gettraindata,
    make_lossplot,
    make_histoweight,
    make_metplots
)
from distillnet_setup import (
    makedataloaders,
    nn_setup,
    validation,
    do_training,
    load_bestmodel,
    modelpredictions,
    FeatureDataset,
)
from distillnet_config import hparams, trainparams
import matplotlib
from joblib import load

matplotlib.rc("font", size=22, family="serif")
matplotlib.rcParams["text.usetex"] = True


def cutfunc(
    pt_scaled,
    features: list,
    weights: list,
    npu: int,
    wcut: float,
    ptcut_central: float,
    ptcut_forward: float,
):
    mask_wcut = np.abs(weights) > wcut
    w_wcut, theta_wcut, pt_wcut, phi_wcut = (
        weights[mask_wcut],
        features[:, 0][mask_wcut],
        pt_scaled[mask_wcut],
        features[:, 1][mask_wcut],
    )
    # theta_wcut,pt_wcut=theta,pt

    mask_ptcentral = np.abs(theta_wcut) <= 2.5  # identify central region
    mask_ptforward = np.abs(theta_wcut) > 2.5  # identify forward region
    pt_central = pt_wcut[
        mask_ptcentral
    ]  # apply mask and only keep particles in regions
    pt_forward = pt_wcut[mask_ptforward]
    phi_central = phi_wcut[mask_ptcentral]
    phi_forward = phi_wcut[mask_ptforward]

    ptcut_central = ptcut_central + 0.007 * npu  # calculate cut according to paper
    ptcut_forward = ptcut_forward + 0.011 * npu

    mask_ptcutc = (
        np.abs(pt_central) > ptcut_central
    )  # apply cut, only keep particles which are greater that ptcut
    mask_ptcutf = np.abs(pt_forward) > ptcut_forward

    pt_c, pt_f, phi_c, phi_f = (
        pt_central[mask_ptcutc],
        pt_forward[mask_ptcutf],
        phi_central[mask_ptcutc],
        phi_forward[mask_ptcutf],
    )
    ptvec_c, ptvec_f = makevec2(pt_c, phi_c), makevec2(
        pt_f, phi_f
    )  # calculate vectors using phi
    pt_vec_x = np.concatenate(
        (ptvec_c[0], ptvec_f[0])
    )  # create one px and one py vector
    pt_vec_y = np.concatenate((ptvec_c[1], ptvec_f[1]))
    ET_missvec = [
        np.sum(pt_vec_x),
        np.sum(pt_vec_y),
    ]  # sum over px and py to get EtMET_vector
    ET_missvec_mag = mag(ET_missvec)  # calculate magnitude of said vector
    return ET_missvec_mag


def resolution(arr, gen):
    q_75_abc = np.quantile(genfunc(arr, gen), 0.75)
    q_25_abc = np.quantile(genfunc(arr, gen), 0.25)
    resolutions = (q_75_abc - q_25_abc) / 2
    return resolutions


def genfunc(arr, gen):
    return (np.array(arr) - np.array(gen)) / np.array(gen)


def makevec(pt, phi):  # create a vector using polar coordinates
    x = pt * np.cos(phi)
    y = pt * np.sin(phi)
    return x, y


def makevec2(pt, phi):  # create a vector using polar coordinates
    x = pt * np.cos(phi)
    y = pt * np.sin(phi)
    return np.array([x, y])


def mag(vec):  # get magnitude of vector
    return np.sqrt((np.sum(vec[0]) ** 2) + (np.sum(vec[1]) ** 2))


def Metcalc(lvec, weights, npu, wcut, ptcut_c, ptcut_f, cut: bool = True):
    pt = lvec[:, 2]
    pt_scaled = weights * pt
    if cut:
        met_cut = cutfunc(pt_scaled, lvec, weights, npu, wcut, ptcut_c, ptcut_f)
        return met_cut
    ptvec = makevec(pt_scaled, lvec[:, 1])
    # print(ptvec)
    # print(ptvec[0])
    # print(ptvec[1])
    ET_missvec = [np.sum(ptvec[0]), np.sum(ptvec[1])]
    ET_magnitude = mag(ET_missvec)
    return ET_magnitude


def Metcalc_check(lvec):
    pt_scaled = lvec[:, 2]
    ptvec = makevec(pt_scaled, lvec[:, 1])
    ET_missvec = [np.sum(ptvec[0]), np.sum(ptvec[1])]
    ET_magnitude = mag(ET_missvec)
    return ET_magnitude


def get_mets(
    filedir: str,
    sample: str,
    flist_inputs: list,
    met_model,
    device,
    Events: int,
    n_particles: int,
    Is_remove_padding: bool = True,
    Is_standard: bool = True,
    Is_min_max_scaler: bool = True,
    Is_standard_scaler: bool = False,
    Is_makeplots: bool = False,
    Is_dtrans: bool = False,
    Is_makeprints: bool = False,
):
    # iwas mit npart mal 90000 oder durch damit man die richtigen Events erwischt
    filename = os.path.join(filedir, sample)
    #print(f"Accessing {filename} for calculating MET")
    veclist = [0, 1, 2, 3]
    #Event_zero = int(n_particles / 9000)
    #Event = int(Events + Event_zero)
    #print(Events)
    with h5py.File(filename, "r") as f:
        featurelist = f["distill_inputs_default"][flist_inputs, Events, :]
        abc_weights = f["data_abc"][4, Events, :]
        puppi_weights = f["data_puppi"][4, Events, :]
        abc_vec = f["data_abc"][veclist, Events, :]
        puppi_vec = f["data_puppi"][veclist, Events, :]
        gen_vec = f["data_gen"][veclist, Events, :]
        default_vec = f["distill_inputs_default"][veclist, Events, :]

        n_features, n_part = featurelist.shape
        features_reshaped = featurelist.reshape(n_features, -1).T
        abc_weights_reshaped = abc_weights.reshape(1, -1).T
        puppi_weights_reshaped = puppi_weights.reshape(1, -1).T
        abc_vec_reshaped = abc_vec.reshape(4, -1).T
        puppi_vec_reshaped = puppi_vec.reshape(4, -1).T
        gen_vec_reshaped = gen_vec.reshape(4, -1).T
        default_vec_reshaped = default_vec.reshape(4, -1).T

        if Is_remove_padding:
           # print("Removing padded particles.....")
            padmask_px = np.abs(features_reshaped[:, 0]) > 0.000001
            gen_padmask = np.abs(gen_vec_reshaped[:, 0]) > 0.000001
            (
                features_np,
                abcw_np,
                puppiw_np,
                abc_vec_np,
                puppi_vec_np,
                default_vec_np,
            ) = (
                features_reshaped[padmask_px],
                abc_weights_reshaped[padmask_px],
                puppi_weights_reshaped[padmask_px],
                abc_vec_reshaped[padmask_px],
                puppi_vec_reshaped[padmask_px],
                default_vec_reshaped[padmask_px],
            )
            gen_vec_np = gen_vec_reshaped[gen_padmask]

        convertvec_etaphipt(features_np, Is_log=True)
        if Is_dtrans:
            if Is_makeprints:
                print("Transforming d0 and dz ")
            features_np[:, 4][np.where(features_np[:, 4] >= 1)] = 1
            features_np[:, 5][np.where(features_np[:, 5] >= 1)] = 1
            features_np[:, 4][np.where(features_np[:, 4] <= -1)] = -1
            features_np[:, 5][np.where(features_np[:, 5] <= -1)] = -1

        if Is_standard:
            if Is_standard_scaler:
                    #print("deepthoughtscaler")
                if len(flist_inputs) == 15:
                    scaler = load("/work/tbrandes/work/Delphes_samples/scalers/std_scaler_nopup_ens.bin")
                if len(flist_inputs) == 16:
                    scaler = load("/work/tbrandes/work/Delphes_samples/scalers/std_scaler.bin")
            if Is_min_max_scaler:
                scaler = load("/work/tbrandes/work/Delphes_samples/scalers/min_max_scaler.bin")
            features_std = scaler.transform(features_np)

        convertvec_etaphipt(default_vec_np)
        convertvec_etaphipt(abc_vec_np)
        convertvec_etaphipt(puppi_vec_np)
        convertvec_etaphipt(gen_vec_np)
        nn_input = (features_std, abcw_np)
        batch_size = len(features_np[:, 2])

        #print(batch_size)
        dataset_test = FeatureDataset(nn_input)
        test_loader = data.DataLoader(
            dataset=dataset_test, shuffle=False, batch_size=batch_size
        )
        predictions = modelpredictions(met_model, test_loader, batch_size, device)
        #print(predictions)
        #print(predictions.shape)
        #predictions = np.concatenate(predictions)
        cut = False
        npu = 100
        met_distillnet = Metcalc(
            default_vec_np, predictions, npu, 0.1, 0.1, 0.2, cut=cut
        )
        met_abc = Metcalc(
            default_vec_np, np.concatenate(abcw_np), npu, 0.1, 0.1, 0.2, cut=cut
        )
        met_puppi = Metcalc(
            default_vec_np, np.concatenate(puppiw_np), npu, 0.1, 0.1, 0.2, cut=cut
        )
        #met_abc_check = Metcalc_check(abc_vec_np)
        #met_puppi_check = Metcalc_check(puppi_vec_np)
        met_gen = Metcalc_check(gen_vec_np)
        # print(met_distillnet)
        # print(met_abc)
        # print(met_abc_check)
        # print(met_puppi)
        # print(met_puppi_check)
        return predictions, abcw_np, puppiw_np, met_distillnet, met_abc, met_puppi, met_gen


def make_resolutionplots(met_a, met_p, met_d, met_g, plotdir, saveinfo, timestr, Is_displayplots: bool = False):
    plt.figure(figsize=(8, 7))
    binsspace = np.arange(-1, 3.1, 0.1)
    ranges = (-1, 3)
    bins_calc, xb, _ = plt.hist(np.clip(genfunc(met_a, met_g), binsspace[0], binsspace[-1]), bins=binsspace, histtype='step',label=r'$E_\mathrm{T}^{\mathrm{miss}}$'+
                                f' ABCNet\nResolution: {resolution(met_a,met_g):.4f}', range=ranges, lw=3)
    bins_puppi, _, _ = plt.hist(np.clip(genfunc(met_p, met_g), binsspace[0], binsspace[-1]), bins=binsspace, histtype='step', label=r'$E_\mathrm{T}^{\mathrm{miss}}$'+
                                f' Puppi\nResolution: {resolution(met_p,met_g):.4f}', range=ranges, lw=3)
    bins_puppi, _, _ = plt.hist(np.clip(genfunc(met_d, met_g), binsspace[0], binsspace[-1]), bins=binsspace, histtype='step', label=r'$E_\mathrm{T}^{\mathrm{miss}}$'+
                                f' DistillNet\nResolution: {resolution(met_d,met_g):.4f}', range=ranges, lw=3)

    plt.legend(fancybox=True, framealpha=0.8, loc='best',prop={'size': 18})#,bbox_to_anchor = (1.02,1),bbox_transform = plt.gca().transAxes,borderaxespad =0)

    plt.xlabel(r"$(E_\mathrm{T}^{\mathrm{miss}}-E_\mathrm{T}^{\mathrm{miss,\,gen}})\;/\;E_\mathrm{T}^{\mathrm{miss,\,gen}}$")#,fontsize=22)#,horizontalalignment='right', x=1.0)

    plt.minorticks_on()

    plt.ylabel(r'$N_\mathrm{Events}\;/\;0.1$')#,fontsize=22)#,horizontalalignment='left', y=0.6)


    _xlim, _ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.xlim(ranges)
    plt.ylim(*_ylim)

    plt.savefig(plotdir + "response_smaller" + saveinfo + '__time_' + timestr + '.png', dpi=500, bbox_inches='tight')
   # plt.savefig(plotdir_pdf + "response" + saveinfo + '__time_' + timestr + '.pdf', bbox_inches='tight')
    if Is_displayplots:
        plt.show()
    else:
        plt.clf()


def model_resolutions(
    filedir: str,
    sample: str,
    flist_inputs: list,
    n_particles: int,
    n_events_max: int, 
    saveinfo: str,
    savedir: str,
    model,
    plotdir: str,
    plotdir_pdf: str,
    timestr: str, 
    Is_remove_padding: bool = True,
    Is_standard: bool = True,
    Is_min_max_scaler: bool = True,
    Is_standard_scaler: bool = False,
    Is_makeplots: bool = False,
    Is_dtrans: bool = False,
):

    distill_wgts, abc_wgts, puppi_wgts, met_d, met_a, met_p, met_g = [], [], [], [], [], [], []

    for i in tqdm(range(int(n_particles / 9000), int(n_events_max))):
        pred, abc, puppi, met_distill, met_abc, met_puppi, met_gen = get_mets(filedir, sample, flist_inputs, met_model, device, i, n_particles,
                                                                    Is_min_max_scaler=Is_min_max_scaler, Is_standard_scaler=Is_standard_scaler,Is_dtrans=Is_dtrans)
        distill_wgts.append(pred)
        abc_wgts.append(abc)
        puppi_wgts.append(puppi)
        met_d.append(met_distill)
        met_a.append(met_abc)
        met_p.append(met_puppi)
        met_g.append(met_gen)

    print(len(met_d))
    resolution_model = resolution(met_d, met_g)
    resolution_abc = resolution(met_a, met_g)
    resolution_puppi = resolution(met_p, met_g)
    print(resolution_model)
    print(resolution_abc)
    print(resolution_puppi)
    make_metplots(met_a, met_p, met_d, resolution_abc, resolution_model, resolution_puppi, plotdir, plotdir_pdf, saveinfo, timestr, Is_savefig=True, Is_displayplots=False)
    metres = np.asarray([met_a, met_p, met_g, met_d])
    np.savetxt(savedir + 'met_results/' + saveinfo + modelname + "first_try_met.csv", metres, delimiter=",")
    #distill_wgts, abc_wgts, puppi_wgts = np.concatenate(distill_wgts), np.concatenate(abc_wgts).squeeze(), np.concatenate(puppi_wgts).squeeze()
    #distilres = np.asarray([distill_wgts, abc_wgts, puppi_wgts])
    #np.savetxt(savedir + 'met_results/' + saveinfo + modelname + "first_try_wgts.csv", distilres, delimiter=",")
    #make_resolutionplots(met_a, met_p, met_d, met_g, plotdir, plotdir_pdf, saveinfo, timestr)


def main():
    import train_distillnet as tdist
    filedir = tdist.filedir
    sample = tdist.w_sample
    device = tdist.device
    Is_dtrans = tdist.Is_dtrans
    #device = 'cpu'
    print(device)
    num_train_particles = int(hparams['maketrain_particles'])
    num_train_particles = int(1.35e7)
    num_maxevents = 12900
    input_size = 16
    batch_size = 256
    num_epochs = 30
    l1_hsize = hparams["L1_hsize"]
    l2_hsize = hparams["L2_hsize"]
    n_outputs = hparams["n_outputs"]
    saveinfo = f"_trainpart_{num_train_particles:.2E}__Batchs_{batch_size}__numep_{num_epochs}_7_3_bn2_std"
    saveinfo = f"_trainpart_{num_train_particles:.2E}__Batchs_{batch_size}__numep_{num_epochs}_6_4__std" # 30 epoch 1.5e7 
    saveinfo = '_tpart_1.35E+07__Batchs_256__numep_48_75_25_bndrop005_werr3__ensemble2__devicecuda:3__numtests5_std'
    saveinfo = '_tpart_1.40E+07__Batchs_256__numep_48_7_3_bndrop005_werr3__ensemble9__devicecuda:0__numtests10_std'
    modelsavedir = os.path.join(tdist.savedir, 'Models_Ensemble/')
    modelname = 'bestmodel_trainloss'
    #modelname = 'bestmodel_valloss'
    #modelname = 'bestmodel'
    plotdir = tdist.savedir + '/Plots/MET/'
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)
    model_resolutions(filedir, sample, tdist.flist_inputs, device, num_train_particles, num_maxevents, saveinfo, tdist.savedir, modelsavedir, modelname,
                      input_size, l1_hsize, l2_hsize, n_outputs, plotdir, tdist.plotdir_pdf, tdist.timestr, Is_min_max_scaler=tdist.Is_min_max_scaler,
                      Is_standard_scaler=tdist.Is_standard_scaler, Is_dtrans=Is_dtrans)


if __name__ == "__main__":
    main()