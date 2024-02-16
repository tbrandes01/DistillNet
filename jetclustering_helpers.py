import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn import preprocessing
import h5py
import os
from joblib import dump, load
import torch.utils.data as data
from tqdm import tqdm
import fastjet
import pandas as pd
import datetime
#from train_distillnet import filedir, w_sample, zprime_sample, savedir
from data_helpers import convertvec_etaphipt
from distillnet_setup import FeatureDataset, modelpredictions
'from allesdoof import *'

# filename = "/work/tbrandes/work/data/best_model_Summer20MiniAODv2_LR_decay_rate_0p5_noPUPPIalpha_x10_KDTree_ttbar_1.h5"
# filename2 = "/storage/9/abal/ABCNet/best_model_Summer20MiniAODv2_LR_decay_rate_0p5_noPUPPIalpha_x10_KDTree_ttbar_2.h5"
# script_dir = os.path.dirname(__file__)
# # script_dir = sys.path[0]  # use this in jupyter
# results_dir = os.path.join(savedir, "Jetclustering/")  # specify desired folder for stuff
# if not os.path.isdir(results_dir):  # is folder does not exist, make one
#     os.makedirs(results_dir)

niceValue = os.nice(19)

def vec3d(pt, phi, eta):  #maybe use vector class 
    px = np.cos(phi) * pt
    py = np.sin(phi) * pt
    pz = np.sinh(eta) * pt
    return px, py, pz


def Jetmatching(dRcrit, pslistjet1, pslistjet2, JetE1, JetE2):  # match 2 Jets based on dR cirteria

    match1 = []  # create lists which should be filled with matching indices
    match2 = []
    for j in range(len(pslistjet1)):  # iterate over first jet
        for i in range(len(pslistjet2)):  # iterate over second jet
            dR = pslistjet1[j].delta_R(pslistjet2[i])  # calculate dR for each jet combination
          #  print(dR)
            if dR < dRcrit:  # if dR satisfies criteria then append list indices from both jets where condition is true
                match1.append(j)
                match2.append(i)

    pslistjet1 = [pslistjet1[k] for k in match1]  # select jets whose dR critera are met
    pslistjet2 = [pslistjet2[k] for k in match2]
    JetE1 = [JetE1[k] for k in match1]  # also select correct energies
    JetE2 = [JetE2[k] for k in match2]

    return pslistjet1, pslistjet2, JetE1, JetE2  # return matched jets


def rescalc(recojet, genjet):  # calculate metric for later resoltion calculation
    return (np.array(recojet) - np.array(genjet)) / np.array(genjet)


def makegenjet(genjetdata):  # create a gen_jet pseudojet from h5 file data
    zeromask = (
        np.abs(genjetdata[:, 3]) > 0.001  # data was padded so remove jets with 0 energy
    )
    eta, phi, pt, e = (  # select features and mask them 
        genjetdata[:, 0][zeromask],
        genjetdata[:, 1][zeromask],
        genjetdata[:, 2][zeromask],
        genjetdata[:, 3][zeromask],
    )
    jets = []  # list of jets
    px, py, pz = vec3d(pt, phi, eta)  # calculate px, py and pz for the jets from eta, pt and phi
    for i in range(0, len(pt)):  # create pseudojet objects for fastjet library
        tmp = fastjet.PseudoJet(np.float64(px[i]), np.float64(py[i]), np.float64(pz[i]), np.float64(e[i]))
        jets.append(tmp)
    jets = fastjet.sorted_by_E(jets)  # sort jets by energy for later calculation
    energy = []  # also append energy in array for later plots
    for elem in jets:
        energy.append(elem.e())
    return jets, energy


def makepuppijet(puppijetdata):  # create puppi_jet from h5 file data, same as gen_jet
    zeromask = (
        np.abs(puppijetdata[:, 3]) > 0.001
    )  # if energy bigger than zero than existent
    eta, phi, pt, e = (
        puppijetdata[:, 0][zeromask],
        puppijetdata[:, 1][zeromask],
        puppijetdata[:, 2][zeromask],  # use RAW pt or so i thought
        puppijetdata[:, 3][zeromask],  # use RAW e
    )  # use raw pt and e
    jets = []
    px, py, pz = vec3d(pt, phi, eta)
    for i in range(0, len(pt)):
        tmp = fastjet.PseudoJet(
            np.float64(px[i]), np.float64(py[i]), np.float64(pz[i]), np.float64(e[i])
        )
        jets.append(tmp)
    jets = fastjet.sorted_by_E(jets)
    energy = []
    for elem in jets:
        energy.append(elem.e())
    return jets, energy


def JetEvent(
    filedir,
    sample,
    Events,
    flist_inputs,
    model,
    device,
    #batch_size,
    dR,
    ptcut,
    Is_standard: bool = True,
    Is_remove_padding: bool = True,
    Is_dtrans: bool = False,
    Is_makeprints: bool = False,
    Is_standard_scaler: bool = True,
    Is_min_max_scaler: bool = False,
):
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
                #print(device)
                if torch.cuda.is_available():
                    #print("deepthoughtscaler")
                    scaler = load("/work/tbrandes/work/Delphes_samples/scalers/std_scaler.bin")
                else:
                    #print("portalscaler")
                    scaler = load("/work/tbrandes/work/Delphes_samples/scalers/std_scaler_portal.bin")
            if Is_min_max_scaler:
                scaler = load("/work/tbrandes/work/Delphes_samples/scalers/min_max_scaler.bin")
            features_std = scaler.transform(features_np)

        convertvec_etaphipt(default_vec_np)  # np is no padding
        convertvec_etaphipt(abc_vec_np)
        convertvec_etaphipt(puppi_vec_np)
        convertvec_etaphipt(gen_vec_np)
        nn_input = (features_std, abcw_np)
        batch_size = len(features_np[:, 2])
        dataset_jetevent = FeatureDataset(nn_input)
        test_loader = data.DataLoader(dataset=dataset_jetevent, shuffle=False, batch_size=batch_size)

        predictions = modelpredictions(model, test_loader, batch_size, device)  # create predicted weights for event

    #    JetAbc = Jetcalc(default_vec_np, np.concatenate(abcw_np), ptcut)  # jet clustering
        #  print("distil")


        # Print the current time

        Jetdistill = Jetcalc(default_vec_np, predictions, ptcut)

        # plt.hist(predictions,histtype='step',bins=40,label='distill')
        # plt.hist(np.concatenate(puppiw_np),histtype='step',bins=40,label='puppi')
        # plt.legend()
        # plt.yscale('log')
        # plt.show()

        #  print("puppi")
   #        Jetpuppi = Jetcalc(default_vec_np, np.concatenate(puppiw_np), ptcut)
        Jetgen = Jetcalc(gen_vec_np, np.ones_like(gen_vec_np[:, 2]), ptcut) #need to provide some kind of weigt array, so just make an array filled with 1
        #JetAbc_test = Jetcalc(abc_vec_np, np.ones_like(abc_vec_np[:, 2]), ptcut)
        #Jetpuppi_test = Jetcalc(puppi_vec_np, np.ones_like(puppi_vec_np[:, 2]), ptcut)


    # abcmatchjets, genabcmatchjets, abcE, genEabc = Jetmatching(
    #     dR, JetAbc[0], Jetgen[0], JetAbc[1], Jetgen[1]
    # )

    distilmatchjets, gendistilmatchjets, distilE, genEdistil = Jetmatching(
        dR, Jetdistill[0], Jetgen[0], Jetdistill[1], Jetgen[1]
    )
    print(Jetdistill[1])
    print(distilE)
    print(Jetgen[1])
    print(genEdistil)
    # puppimatchjets, genpuppimatchjets, puppiE, genEpuppi = Jetmatching(
    #     dR, Jetpuppi[0], Jetgen[0], Jetpuppi[1], Jetgen[1]
    # )

    # puppimatchjets_true, genpuppimatchjets_true, puppiE_test, genEpuppi_test = Jetmatching(
    #     dR, Jetpuppi_test[0], Jetgen[0], Jetpuppi_test[1], Jetgen[1])

    # abcmatchjets_true, genabcmatchjets_true, abcE_test, genEabc_test = Jetmatching(
    #     dR, JetAbc_test[0], Jetgen[0], JetAbc_test[1], Jetgen[1])
   # responseabc = rescalc(abcE, genEabc)
    responsedistil = rescalc(distilE, genEdistil)
    print(responsedistil)
 #   responsepuppi = rescalc(puppiE, genEpuppi)
    # responsepuppi_true = rescalc(puppiE_test, genEpuppi_test)
    # responseabc_true = rescalc(abcE_test, genEabc_test)

    return (
      #  JetAbc,
        Jetdistill,
    #    Jetpuppi,
        Jetgen,
       # Jetpuppi_test,
       # JetAbc_test,
    #    responseabc,
        responsedistil,
        distilE,
        genEdistil,
     #   responsepuppi,
       # responsepuppi_true,
       # responseabc_true,
    )


def Jetcalc(features, weights, ptcut):
    pseudoparticles = []
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    #pt = np.exp(features[:, 2])
    #e = np.exp(features[:, 3])
    pt_scaled = features[:, 2] * weights
    e_scaled = features[:, 3] * weights
    #print(pt_scaled)
    #print(pt_scaled.shape)
    px, py, pz = vec3d(pt_scaled, features[:, 1], features[:, 0])  # pt,phi and eta
    for i in range(0, len(features[:, 2])):  # create pseudojet for each particle
        # print(np.float64(px[i]))
        # print(np.float64(py[i]))
        # print(np.float64(pz[i]))
        # print(np.float64(e_scaled[i]))
    
        tmp = fastjet.PseudoJet(
            np.float64(px[i]),
            np.float64(py[i]),
            np.float64(pz[i]),
            np.float64(e_scaled[i]),
        )
        pseudoparticles.append(tmp)

    clustered = fastjet.ClusterSequence(pseudoparticles, jetdef)  # cluster all pseudoparticles

    inc_jets = clustered.inclusive_jets(ptcut)  # cut at pt= 1.5
    #ptsortedjets = fastjet.sorted_by_pt(inc_jets)
    Esortedjets = fastjet.sorted_by_E(inc_jets)
    energy = []

    for elem in Esortedjets:
        energy.append(elem.e())

    return Esortedjets, energy


def resolution(arr):
    q_75_abc = np.quantile(arr, 0.75)
    q_25_abc = np.quantile(arr, 0.25)
    resolutions = (q_75_abc - q_25_abc) / 2
    return resolutions


def corinputs(dataframe, index):
    npdf = dataframe.to_numpy()
    npdf = np.delete(npdf[index],0)
    npdf = npdf[~pd.isnull(npdf)]
    return npdf


def JetEvent_2(
    filedir,
    sample,
    Events,
    flist_inputs,
    model,
    device,
    #batch_size,
    dR,
    ptcut,
    Is_standard: bool = True,
    Is_remove_padding: bool = True,
    Is_dtrans: bool = False,
    Is_makeprints: bool = False,
    Is_standard_scaler: bool = True,
    Is_min_max_scaler: bool = False,
):
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
                #print(device)
                if torch.cuda.is_available():
                    #print("deepthoughtscaler")
                    scaler = load("/work/tbrandes/work/Delphes_samples/scalers/std_scaler.bin")
                else:
                    #print("portalscaler")
                    scaler = load(filedir + "std_scaler_portal.bin")
            if Is_min_max_scaler:
                scaler = load("/work/tbrandes/work/Delphes_samples/scalers/min_max_scaler.bin")
            features_std = scaler.transform(features_np)

        convertvec_etaphipt(default_vec_np) # np is no padding
        convertvec_etaphipt(abc_vec_np)
        convertvec_etaphipt(puppi_vec_np)
        convertvec_etaphipt(gen_vec_np)
        # nn_input = (features_std, abcw_np)
        # batch_size = len(features_np[:, 2])
        # dataset_jetevent = FeatureDataset(nn_input)
        # test_loader = data.DataLoader(dataset=dataset_jetevent, shuffle=False, batch_size=batch_size)

        # predictions = modelpredictions(model, test_loader, batch_size, device)  # create predicted weights for event

        JetAbc = Jetcalc(default_vec_np, np.concatenate(abcw_np), ptcut)  # jet clustering
        #  print("distil")


        # Print the current time

      #  Jetdistill = Jetcalc(default_vec_np, predictions, ptcut)

        # plt.hist(predictions,histtype='step',bins=40,label='distill')
        # plt.hist(np.concatenate(puppiw_np),histtype='step',bins=40,label='puppi')
        # plt.legend()
        # plt.yscale('log')
        # plt.show()

        #  print("puppi")
        Jetpuppi = Jetcalc(default_vec_np, np.concatenate(puppiw_np), ptcut)
        Jetgen = Jetcalc(gen_vec_np, np.ones_like(gen_vec_np[:, 2]), ptcut) #need to provide some kind of weigt array, so just make an array filled with 1
        #JetAbc_test = Jetcalc(abc_vec_np, np.ones_like(abc_vec_np[:, 2]), ptcut)
        #Jetpuppi_test = Jetcalc(puppi_vec_np, np.ones_like(puppi_vec_np[:, 2]), ptcut)


    abcmatchjets, genabcmatchjets, abcE, genEabc = Jetmatching(
        dR, JetAbc[0], Jetgen[0], JetAbc[1], Jetgen[1]
    )

    # distilmatchjets, gendistilmatchjets, distilE, genEdistil = Jetmatching(
    #     dR, Jetdistill[0], Jetgen[0], Jetdistill[1], Jetgen[1]
    # )

    puppimatchjets, genpuppimatchjets, puppiE, genEpuppi = Jetmatching(
        dR, Jetpuppi[0], Jetgen[0], Jetpuppi[1], Jetgen[1]
    )

    # puppimatchjets_true, genpuppimatchjets_true, puppiE_test, genEpuppi_test = Jetmatching(
    #     dR, Jetpuppi_test[0], Jetgen[0], Jetpuppi_test[1], Jetgen[1])

    # abcmatchjets_true, genabcmatchjets_true, abcE_test, genEabc_test = Jetmatching(
    #     dR, JetAbc_test[0], Jetgen[0], JetAbc_test[1], Jetgen[1])
    responseabc = rescalc(abcE, genEabc)
   # responsedistil = rescalc(distilE, genEdistil)
    responsepuppi = rescalc(puppiE, genEpuppi)
    # responsepuppi_true = rescalc(puppiE_test, genEpuppi_test)
    # responseabc_true = rescalc(abcE_test, genEabc_test)

    return (
        JetAbc,
      #  Jetdistill,
        Jetpuppi,
        Jetgen,
       # Jetpuppi_test,
       # JetAbc_test,
        responseabc,
     #   responsedistil,
        responsepuppi,
       # responsepuppi_true,
       # responseabc_true,
    )


def make_jetenergyplot(distiljetE: list, genjetE: list, plotdir: str, plotdir_pdf: str, saveinfo: str, timestr: str, Is_savefig: bool = True, Is_displayplots: bool = False):
    figure = plt.figure(figsize=(10, 9))

    binsspace = 30
    xmaxbin = 550
    xminbin = 0
    range = (xminbin, xmaxbin)

    bins_distil, xb, _ = plt.hist(np.clip(distiljetE, 0, xmaxbin), bins=binsspace, histtype='step', color='green', label='Jet Energy distilNet', lw=2, range=range)
    bins_gen, _, _ = plt.hist(np.clip(genjetE, 0, xmaxbin), bins=xb, color="black", histtype='step', label='Jet Energy gen', lw=1, range=range)

    _xlim, _ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.xlim(xminbin, xmaxbin)
    plt.ylim(*_ylim)

    plt.ylabel('Number of jets per bin')
    plt.xlabel('Jet energy in GeV')
    plt.legend(fancybox=True, framealpha=0.1, loc='best', prop={'size': 20})
    sample_file_name0 = 'jetE'
    if Is_savefig:
        plt.savefig(plotdir + sample_file_name0 + saveinfo + '__time_' + timestr + '.png', dpi=400, bbox_inches='tight')
        plt.savefig(plotdir_pdf + sample_file_name0 + saveinfo + '__time_' + timestr + '.pdf', bbox_inches='tight')
    if Is_displayplots:
        plt.show()
    else:
        plt.clf()
    plt.close()
    return


def make_jetresolutionplots(responsedistil: list, plotdir: str, plotdir_pdf: str, saveinfo: str, timestr: str, Is_savefig: bool = True, Is_displayplots: bool = False):

    r2 = resolution(responsedistil)

    figure = plt.figure(figsize=(8, 7))

    binsspace = 40
    xmaxbin = 0.8
    xminbin = -0.8
    range = (xminbin, xmaxbin)

    bins_distil, _, _ = plt.hist(responsedistil, color="green",bins=binsspace, histtype='step', label=r'$E_\mathrm{Jet,\,reco}$'+f' DistillNet\nResolution: {r2:.4f}', lw=1.5,range=range)
    #bins_puppi, xb, _ = plt.hist(responsepuppi, bins=binsspace, histtype='step', label='Jets Puppi', lw=1.5,range=range)
    #figure.subplots_adjust(hspace=0)


    _xlim, _ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.xlim(xminbin,xmaxbin)
    plt.ylim(*_ylim)
    plt.vlines(0,ymin=0, ymax=_ylim[1], color='black', alpha=0.99,linestyles="dotted")
    #plt.vlines(xb, ymin=0, ymax=_ylim[1], color='black', alpha=0.1)
    def resline(arr,colors,labels):
        plt.vlines(np.quantile(arr,0.75),ymin=0, ymax=_ylim[1], alpha=0.9,linestyles="dashed",color=colors,label=labels)
        plt.vlines(np.quantile(arr,0.25),ymin=0, ymax=_ylim[1], alpha=0.9,linestyles="dashed",color=colors)

    plt.minorticks_on()

    plt.ylabel(r'$N_\mathrm{Jets,\,reco}\;/\; 0.04$')
  #  plt.xlabel('(recojet - genjet) / genjet')
 #   plt.xlabel(r"Jet $E_\mathrm{reco}$-Jet $E_\mathrm{gen}$ / $\mathrm{Jet} E_\mathrm{gen}$")
    plt.xlabel(r"$(E_\mathrm{Jet,\,reco}-E_\mathrm{Jet,\,gen})\;/\;E_\mathrm{Jet,\,gen}$")
    plt.legend(fancybox=True, framealpha=0.8, loc='best', prop={'size': 17})#,bbox_to_anchor = (1.02,1),bbox_transform = plt.gca().transAxes,borderaxespad =0)
    sample_file_name0 = 'jetresolution'
    if Is_savefig:
        plt.savefig(plotdir + sample_file_name0 + saveinfo + '__time_' + timestr + '.png', dpi=400, bbox_inches='tight')
        plt.savefig(plotdir_pdf + sample_file_name0 + saveinfo + '__time_' + timestr + '.pdf', bbox_inches='tight')
    if Is_displayplots:
        plt.show()
    else:
        plt.clf()
    plt.close()
    return

'''
model1 = torch.load(script_dir + "/Models/" + strinfo + f"removepadding_{True}__featuretest__time_" + "trysaving2.pth")
#model1 = NeuralNet(14,128,1)
#model1.load_state_dict(torch.load(script_dir+"/Training Models/Models_ensemble/distilNet/"+"__trainparticles10000000__Batchsize_256__numepochs_40__ensemble1__devicecuda:3__modlayers128_16__time_bestmodel.pth"))  # original was 20 epochs
model1.to(device)
jetabc = []
jetdistil = []
jetpuppi = []
ptcut = 20  # GeV
nEvents = 20000
dR = 0.4
info = f"Eventsclustered_{nEvents}__dR_{dR}_bencholdmodel"
numjetspuppi, numjetsabc, numjetsdistil, numjetsgen, numjetspuppi_true = 0, 0, 0, 0, 0
(
    evalabc,
    evaldistil,
    evalpuppi,
    evalgen,
    evalpuppi_true,
    resabc,
    resdistil,
    respuppi,
    respuppi_true,
) = ([], [], [], [], [], [], [], [], [])

for i in tqdm(range(6500, nEvents)):  # loop over desired number of events
    (
        jetsabc,
        jetsdistil,
        jetspuppi,
        jetsgen,
        jetspuppi_true,
        repabc,
        repdistil,
        reppuppi,
        reppuppi_true,
    ) = JetEvent(
        filename,
        i,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17],
        model1,
        device,
        batch_size,
        dR,
        ptcut=ptcut,
        standard=False,
        remove_padding=True,
    )

    evalabc = np.append(evalabc, jetsabc[1])
    evaldistil = np.append(evaldistil, jetsdistil[1])
    evalpuppi = np.append(evalpuppi, jetspuppi[1])
    evalgen = np.append(evalgen, jetsgen[1])
    evalpuppi_true = np.append(evalpuppi_true, jetspuppi_true[1])
    resabc = np.append(resabc, repabc)
    resdistil = np.append(resdistil, repdistil)
    respuppi = np.append(respuppi, reppuppi)
    respuppi_true = np.append(respuppi_true, reppuppi_true)
    numjetspuppi += len(jetspuppi[1])
    numjetsabc += len(jetsabc[1])
    numjetsdistil += len(jetsdistil[1])
    numjetsgen += len(jetsgen[1])
    numjetspuppi_true += len(jetspuppi_true[1])
# store data in csv file
rawdata = {
    "jetenergy_ABC": evalabc,
    "jetenergy_distil": evaldistil,
    "jetenergy_puppi": evalpuppi,
    "jetenergy_gen": evalgen,
    "jetenergy_puppitrue": evalpuppi_true,
    "resabc": resabc,
    "resdistil": resdistil,
    "respuppi": respuppi,
    "respuppi_true": respuppi_true,
}
df = pd.DataFrame.from_dict(rawdata, orient="index")
print(df)
strsave = results_dir + info + "clusterresults_file1.csv"
print(strsave)
df.to_csv(strsave)

for i in tqdm(range(0, nEvents)):  # loop over desired number of events
    (
        jetsabc,
        jetsdistil,
        jetspuppi,
        jetsgen,
        jetspuppi_true,
        repabc,
        repdistil,
        reppuppi,
        reppuppi_true,
    ) = JetEvent(
        filename2,
        i,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17],
        model1,
        device,
        batch_size,
        dR,
        ptcut=ptcut,
        standard=False,
        remove_padding=True,
    )

    evalabc = np.append(evalabc, jetsabc[1])
    evaldistil = np.append(evaldistil, jetsdistil[1])
    evalpuppi = np.append(evalpuppi, jetspuppi[1])
    evalgen = np.append(evalgen, jetsgen[1])
    evalpuppi_true = np.append(evalpuppi_true, jetspuppi_true[1])
    resabc = np.append(resabc, repabc)
    resdistil = np.append(resdistil, repdistil)
    respuppi = np.append(respuppi, reppuppi)
    respuppi_true = np.append(respuppi_true, reppuppi_true)
    numjetspuppi += len(jetspuppi[1])
    numjetsabc += len(jetsabc[1])
    numjetsdistil += len(jetsdistil[1])
    numjetsgen += len(jetsgen[1])
    numjetspuppi_true += len(jetspuppi_true[1])
print("numjetspuppi_total: ", numjetspuppi)
print("numjetspuppitrue_total: ", numjetspuppi_true)
print("numjetsabc_total: ", numjetsabc)
print("numjetsdistil_total: ", numjetsdistil)
print("numjetsgen_total: ", numjetsgen)

# store data in csv file
rawdata2 = {
    "jetenergy_ABC": evalabc,
    "jetenergy_distil": evaldistil,
    "jetenergy_puppi": evalpuppi,
    "jetenergy_gen": evalgen,
    "jetenergy_puppitrue": evalpuppi_true,
    "resabc": resabc,
    "resdistil": resdistil,
    "respuppi": respuppi,
    "respuppi_true": respuppi_true,
}
df = pd.DataFrame.from_dict(rawdata2, orient="index")
print(df)
strsave = results_dir + info + "clusterresults_all.csv"
print(strsave)
df.to_csv(strsave)
'''