import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import torch.utils.data as data
import fastjet
import pandas as pd
import gc
from typing import Tuple, List
from data_helpers import convertvec_etaphipt
from distillnet_setup import FeatureDataset, modelpredictions


def vec3d(pt: list, phi: list, eta: list):
    """
    Create px,py and pz vector from eta, phi and pt.
    """
    px = np.cos(phi) * pt
    py = np.sin(phi) * pt
    pz = np.sinh(eta) * pt
    return px, py, pz


def Jetmatching(
    dRcrit: float,
    pslistjet1: list,
    pslistjet2: list,
    JetE1: list,
    JetE2: list,
    Jetpt1: list,
    Jetpt2: list,
):
    """
    Match 2 lists of jets based on dR criteria.
    """

    match1 = []  # create lists which should be filled with matching indices
    match2 = []
    for j in range(len(pslistjet1)):  # iterate over first jet
        for i in range(len(pslistjet2)):  # iterate over second jet
            dR = pslistjet1[j].delta_R(pslistjet2[i])  # calculate dR for each jet combination
            #  print(dR)
            if (
                dR < dRcrit
            ):  # if dR satisfies criteria then append list indices from both jets where condition is true
                match1.append(j)
                match2.append(i)

    pslistjet1 = [pslistjet1[k] for k in match1]  # select jets whose dR critera are met
    pslistjet2 = [pslistjet2[k] for k in match2]
    JetE1 = [JetE1[k] for k in match1]  # also select correct energies
    JetE2 = [JetE2[k] for k in match2]
    Jetpt1 = [Jetpt1[k] for k in match1]
    Jetpt2 = [Jetpt2[k] for k in match2]

    return pslistjet1, pslistjet2, JetE1, JetE2, Jetpt1, Jetpt2  # return matched jets


def rescalc(recojet, genjet):  # calculate metric for later resoltion calculation
    return (np.array(recojet) - np.array(genjet)) / np.array(genjet)


def JetEvent(
    filedir: str,
    sample: str,
    Events: int,
    flist_inputs: list,
    model,
    device: str,
    dR: float,
    ptcut: int,
    scaler,
    is_remove_padding: bool = True,
    is_dtrans: bool = False,
    is_makeprints: bool = False,
    is_abc_puppi: bool = False,
    is_return_all: bool = False,
):
    """
    Cluster and match jets to generator level jets for an event.
    Returns sorted matched jet energies, respective pt to matched energies and energy resolution.
    """
    filename = os.path.join(filedir, sample)
    veclist = [0, 1, 2, 3]

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

        del featurelist, abc_weights, puppi_weights, abc_vec, puppi_vec, gen_vec, default_vec
        gc.collect()
        if is_remove_padding:
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

        del (
            features_reshaped,
            abc_weights_reshaped,
            puppi_weights_reshaped,
            abc_vec_reshaped,
            puppi_vec_reshaped,
            gen_vec_reshaped,
            default_vec_reshaped,
        )
        gc.collect()
        convertvec_etaphipt(features_np, is_log=True)
        if is_dtrans:
            if is_makeprints:
                print("Transforming d0 and dz ")
            features_np[:, 4][np.where(features_np[:, 4] >= 1)] = 1
            features_np[:, 5][np.where(features_np[:, 5] >= 1)] = 1
            features_np[:, 4][np.where(features_np[:, 4] <= -1)] = -1
            features_np[:, 5][np.where(features_np[:, 5] <= -1)] = -1

        features_std = scaler.transform(features_np)

        convertvec_etaphipt(default_vec_np)  # np is no padding
        convertvec_etaphipt(abc_vec_np)
        convertvec_etaphipt(puppi_vec_np)
        convertvec_etaphipt(gen_vec_np)
        nn_input = (features_std, abcw_np)
        batch_size = len(features_np[:, 2])
        dataset_jetevent = FeatureDataset(nn_input)
        test_loader = data.DataLoader(
            dataset=dataset_jetevent, shuffle=False, batch_size=batch_size
        )

        predictions = modelpredictions(
            model, test_loader, batch_size, device
        )  # create predicted weights for event
        del test_loader, dataset_jetevent, nn_input, features_np
        gc.collect()
        if is_return_all:
            JetAbc = Jetcalc(default_vec_np, np.concatenate(abcw_np), ptcut)  # jet clustering
            Jetpuppi = Jetcalc(default_vec_np, np.concatenate(puppiw_np), ptcut)
            Jetdistill = Jetcalc(default_vec_np, predictions, ptcut)
            Jetgen = Jetcalc(
                gen_vec_np, np.ones_like(gen_vec_np[:, 2]), ptcut
            )  # ones like to get a weight vector the same size as the gen vector pt

            abcmatchjets, genabcmatchjets, abcE, genEabc, abc_pt_match, gen_pt_abcmatch = (
                Jetmatching(dR, JetAbc[0], Jetgen[0], JetAbc[1], Jetgen[1], JetAbc[2], Jetgen[2])
            )
            (
                puppimatchjets,
                genpuppimatchjets,
                puppiE,
                genEpuppi,
                puppi_pt_match,
                gen_pt_puppimatch,
            ) = Jetmatching(
                dR, Jetpuppi[0], Jetgen[0], Jetpuppi[1], Jetgen[1], Jetpuppi[2], Jetgen[2]
            )
            (
                distilmatchjets,
                gendistilmatchjets,
                distilE,
                genEdistil,
                distill_pt_match,
                gen_pt_distillmatch,
            ) = Jetmatching(
                dR, Jetdistill[0], Jetgen[0], Jetdistill[1], Jetgen[1], Jetdistill[2], Jetgen[2]
            )
            responseabc = rescalc(abcE, genEabc)
            responsepuppi = rescalc(puppiE, genEpuppi)
            responsedistil = rescalc(distilE, genEdistil)

            return (
                JetAbc,
                Jetpuppi,
                Jetgen,
                responseabc,
                responsepuppi,
                abcE,
                genEabc,
                puppiE,
                genEpuppi,
                Jetdistill,
                Jetgen,
                responsedistil,
                distilE,
                genEdistil,
                abc_pt_match,
                gen_pt_abcmatch,
                puppi_pt_match,
                gen_pt_puppimatch,
                distill_pt_match,
                gen_pt_distillmatch,
            )

        else:
            if is_abc_puppi:
                JetAbc = Jetcalc(
                    default_vec_np, np.concatenate(abcw_np), ptcut
                )  # jet clustering for ABCNet and Puppi
                Jetpuppi = Jetcalc(default_vec_np, np.concatenate(puppiw_np), ptcut)
            else:
                Jetdistill = Jetcalc(
                    default_vec_np, predictions, ptcut
                )  # only jet clustering dor DistillNet
                del default_vec_np, predictions
                gc.collect()

            Jetgen = Jetcalc(
                gen_vec_np, np.ones_like(gen_vec_np[:, 2]), ptcut
            )  # gen has to be clustered either way
            if is_abc_puppi:  # now match only ABCNet and Puppi jets

                abcmatchjets, genabcmatchjets, abcE, genEabc, abc_pt_match, gen_pt_abcmatch = (
                    Jetmatching(
                        dR, JetAbc[0], Jetgen[0], JetAbc[1], Jetgen[1], JetAbc[2], Jetgen[2]
                    )
                )
                (
                    puppimatchjets,
                    genpuppimatchjets,
                    puppiE,
                    genEpuppi,
                    puppi_pt_match,
                    gen_pt_puppimatch,
                ) = Jetmatching(
                    dR, Jetpuppi[0], Jetgen[0], Jetpuppi[1], Jetgen[1], Jetpuppi[2], Jetgen[2]
                )
            else:  # match only DistillJets
                (
                    distilmatchjets,
                    gendistilmatchjets,
                    distilE,
                    genEdistil,
                    distill_pt_match,
                    gen_pt_distillmatch,
                ) = Jetmatching(
                    dR, Jetdistill[0], Jetgen[0], Jetdistill[1], Jetgen[1], Jetdistill[2], Jetgen[2]
                )

            if is_abc_puppi:  # only return ABC and Puppi jets
                responseabc = rescalc(abcE, genEabc)
                responsepuppi = rescalc(puppiE, genEpuppi)
                return (
                    JetAbc,
                    Jetpuppi,
                    Jetgen,
                    responseabc,
                    responsepuppi,
                    abcE,
                    genEabc,
                    puppiE,
                    genEpuppi,
                    abc_pt_match,
                    gen_pt_abcmatch,
                    puppi_pt_match,
                    gen_pt_puppimatch,
                )
            else:  # only return DistillNet jets
                responsedistil = rescalc(distilE, genEdistil)
                return (
                    Jetdistill,
                    Jetgen,
                    responsedistil,
                    distilE,
                    genEdistil,
                    distill_pt_match,
                    gen_pt_distillmatch,
                )


def Jetcalc(features: Tuple[list, ...], weights: list, ptcut: float) -> Tuple[list, list, list]:
    """
    Performs jet clustering and returns energy sorted pseudojet array, sorted energies and respective pt to sorted energies.
    """
    pseudoparticles = []
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    pt_scaled = features[:, 2] * weights
    e_scaled = features[:, 3] * weights

    px, py, pz = vec3d(pt_scaled, features[:, 1], features[:, 0])  # pt,phi and eta
    for i in range(0, len(features[:, 2])):  # create pseudojet for each particle
        tmp = fastjet.PseudoJet(
            np.float64(px[i]),
            np.float64(py[i]),
            np.float64(pz[i]),
            np.float64(e_scaled[i]),
        )
        pseudoparticles.append(tmp)

    clustered = fastjet.ClusterSequence(pseudoparticles, jetdef)  # cluster all pseudoparticles

    inc_jets = clustered.inclusive_jets(ptcut)  # cut at pt= 1.5
    del clustered
    gc.collect()
    Esortedjets = fastjet.sorted_by_E(inc_jets)
    energy = []
    ptrans = []
    for elem in Esortedjets:
        energy.append(elem.e())
        ptrans.append(elem.pt())

    return (
        Esortedjets,
        energy,
        ptrans,
    )  # returns list of pseudojets sorted by energy, list of energies and list of pt's from sorted energies


def resolution(arr: list) -> float:
    q_75_abc = np.quantile(arr, 0.75)
    q_25_abc = np.quantile(arr, 0.25)
    resolutions = (q_75_abc - q_25_abc) / 2
    return resolutions


def corinputs(dataframe: pd.DataFrame, index: int) -> pd.DataFrame:
    npdf = dataframe.to_numpy()
    npdf = np.delete(npdf[index], 0)
    npdf = npdf[~pd.isnull(npdf)]
    return npdf


def make_jetenergyplot(
    distiljetE: list,
    genjetE: list,
    plotdir: str,
    plotdir_pdf: str,
    saveinfo: str,
    timestr: str,
    is_savefig: bool = True,
    is_displayplots: bool = False,
):
    figure = plt.figure(figsize=(10, 9))

    binsspace = 30
    xmaxbin = 550
    xminbin = 0
    range = (xminbin, xmaxbin)

    bins_distil, xb, _ = plt.hist(
        np.clip(distiljetE, 0, xmaxbin),
        bins=binsspace,
        histtype="step",
        color="green",
        label="Jet Energy distilNet",
        lw=2,
        range=range,
    )
    bins_gen, _, _ = plt.hist(
        np.clip(genjetE, 0, xmaxbin),
        bins=xb,
        color="black",
        histtype="step",
        label="Jet Energy gen",
        lw=1,
        range=range,
    )

    _xlim, _ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.xlim(xminbin, xmaxbin)
    plt.ylim(*_ylim)

    plt.ylabel("Number of jets per bin")
    plt.xlabel("Jet energy in GeV")
    plt.legend(fancybox=True, framealpha=0.1, loc="best", prop={"size": 20})
    sample_file_name0 = "jetE"
    if is_savefig:
        plt.savefig(
            plotdir + sample_file_name0 + saveinfo + "__time_" + timestr + ".png",
            dpi=400,
            bbox_inches="tight",
        )
        plt.savefig(
            plotdir_pdf + sample_file_name0 + saveinfo + "__time_" + timestr + ".pdf",
            bbox_inches="tight",
        )
    if is_displayplots:
        plt.show()
    else:
        plt.clf()
    plt.close()
    return


def make_jetresolutionplots(
    responsedistil: list,
    plotdir: str,
    plotdir_pdf: str,
    saveinfo: str,
    timestr: str,
    is_savefig: bool = True,
    is_displayplots: bool = False,
):

    r2 = resolution(responsedistil)

    figure = plt.figure(figsize=(8, 7))

    binsspace = 40
    xmaxbin = 0.8
    xminbin = -0.8
    range = (xminbin, xmaxbin)

    bins_distil, _, _ = plt.hist(
        responsedistil,
        color="green",
        bins=binsspace,
        histtype="step",
        label=r"$E_\mathrm{Jet,\,reco}$" + f" DistillNet\nResolution: {r2:.4f}",
        lw=1.5,
        range=range,
    )

    _xlim, _ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.xlim(xminbin, xmaxbin)
    plt.ylim(*_ylim)
    plt.vlines(0, ymin=0, ymax=_ylim[1], color="black", alpha=0.99, linestyles="dotted")

    # plt.vlines(xb, ymin=0, ymax=_ylim[1], color='black', alpha=0.1)
    def resline(arr, colors, labels):
        plt.vlines(
            np.quantile(arr, 0.75),
            ymin=0,
            ymax=_ylim[1],
            alpha=0.9,
            linestyles="dashed",
            color=colors,
            label=labels,
        )
        plt.vlines(
            np.quantile(arr, 0.25),
            ymin=0,
            ymax=_ylim[1],
            alpha=0.9,
            linestyles="dashed",
            color=colors,
        )

    plt.minorticks_on()

    plt.ylabel(r"$N_\mathrm{Jets,\,reco}\;/\; 0.04$")
    #  plt.xlabel('(recojet - genjet) / genjet')
    #   plt.xlabel(r"Jet $E_\mathrm{reco}$-Jet $E_\mathrm{gen}$ / $\mathrm{Jet} E_\mathrm{gen}$")
    plt.xlabel(r"$(E_\mathrm{Jet,\,reco}-E_\mathrm{Jet,\,gen})\;/\;E_\mathrm{Jet,\,gen}$")
    plt.legend(
        fancybox=True, framealpha=0.8, loc="best", prop={"size": 17}
    )  # ,bbox_to_anchor = (1.02,1),bbox_transform = plt.gca().transAxes,borderaxespad =0)
    sample_file_name0 = "jetresolution"
    if is_savefig:
        plt.savefig(
            plotdir + sample_file_name0 + saveinfo + "__time_" + timestr + ".png",
            dpi=400,
            bbox_inches="tight",
        )
        plt.savefig(
            plotdir_pdf + sample_file_name0 + saveinfo + "__time_" + timestr + ".pdf",
            bbox_inches="tight",
        )
    if is_displayplots:
        plt.show()
    else:
        plt.clf()
    plt.close()
    return
