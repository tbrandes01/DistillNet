import torch.utils.data as data
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm
import matplotlib
from joblib import load
from data_helpers import fl_inputs, fl_vecs, convertvec_etaphipt, gettraindata, make_lossplot, make_histoweight, make_metplots
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

matplotlib.rc("font", size=22, family="serif")
matplotlib.rcParams["text.usetex"] = True


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


def makevec_vers2(pt, phi):  # create a vector using polar coordinates
    x = pt * np.cos(phi)
    y = pt * np.sin(phi)
    return np.array([x, y])


def mag(vec):  # get magnitude of vector
    return np.sqrt((np.sum(vec[0]) ** 2) + (np.sum(vec[1]) ** 2))


def Metcalc(lvec, weights):
    pt = lvec[:, 2]
    pt_scaled = weights * pt
    ptvec = makevec(pt_scaled, lvec[:, 1])
    ET_missvec = [np.sum(ptvec[0]), np.sum(ptvec[1])]
    ET_magnitude = mag(ET_missvec)
    return ET_magnitude


def Metcalc_gen(lvec):
    pt_scaled = lvec[:, 2]
    ptvec = makevec(pt_scaled, lvec[:, 1])
    ET_missvec = [np.sum(ptvec[0]), np.sum(ptvec[1])]
    ET_magnitude = mag(ET_missvec)
    return ET_magnitude


def get_mets(
    filedir: str,
    scalerdir: str,
    sample: str,
    flist_inputs: list,
    met_model,
    device: str,
    Events: int,
    is_remove_padding: bool = True,
    is_standard: bool = True,
    is_min_max_scaler: bool = True,
    is_standard_scaler: bool = False,
    is_dtrans: bool = False,
    is_makeprints: bool = False,
):
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

        if is_remove_padding:
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

        convertvec_etaphipt(features_np, is_log=True)
        if is_dtrans:
            if is_makeprints:
                print("Transforming d0 and dz ")
            features_np[:, 4][np.where(features_np[:, 4] >= 1)] = 1
            features_np[:, 5][np.where(features_np[:, 5] >= 1)] = 1
            features_np[:, 4][np.where(features_np[:, 4] <= -1)] = -1
            features_np[:, 5][np.where(features_np[:, 5] <= -1)] = -1

        if is_standard:
            if is_standard_scaler:
                if len(flist_inputs) == 15:
                    f"/std_scaler_feat{len(flist_inputs)}.bin"
                    scaler = load(scalerdir + f"/std_scaler_feat{len(flist_inputs)}.bin")
                if len(flist_inputs) == 16:
                    scaler = load(scalerdir + f"/std_scaler_feat{len(flist_inputs)}.bin")
            if is_min_max_scaler:
                scaler = load(scalerdir + "/min_max_scaler.bin")
            features_std = scaler.transform(features_np)

        convertvec_etaphipt(default_vec_np)
        convertvec_etaphipt(abc_vec_np)
        convertvec_etaphipt(puppi_vec_np)
        convertvec_etaphipt(gen_vec_np)
        nn_input = (features_std, abcw_np)
        batch_size = len(features_np[:, 2])

        dataset_test = FeatureDataset(nn_input)
        test_loader = data.DataLoader(dataset=dataset_test, shuffle=False, batch_size=batch_size)
        predictions = modelpredictions(met_model, test_loader, batch_size, device)

        met_distillnet = Metcalc(default_vec_np, predictions)
        met_abc = Metcalc(default_vec_np, np.concatenate(abcw_np))
        met_puppi = Metcalc(default_vec_np, np.concatenate(puppiw_np))
        met_gen = Metcalc_gen(gen_vec_np)

        return predictions, abcw_np, puppiw_np, met_distillnet, met_abc, met_puppi, met_gen


def make_resolutionplots(met_a, met_p, met_d, met_g, plotdir, saveinfo, timestr, is_displayplots: bool = False):
    plt.figure(figsize=(8, 7))
    binsspace = np.arange(-1, 3.1, 0.1)
    ranges = (-1, 3)
    bins_calc, xb, _ = plt.hist(
        np.clip(genfunc(met_a, met_g), binsspace[0], binsspace[-1]),
        bins=binsspace,
        histtype="step",
        label=r"$E_\mathrm{T}^{\mathrm{miss}}$" + f" ABCNet\nResolution: {resolution(met_a,met_g):.4f}",
        range=ranges,
        lw=3,
    )
    bins_puppi, _, _ = plt.hist(
        np.clip(genfunc(met_p, met_g), binsspace[0], binsspace[-1]),
        bins=binsspace,
        histtype="step",
        label=r"$E_\mathrm{T}^{\mathrm{miss}}$" + f" Puppi\nResolution: {resolution(met_p,met_g):.4f}",
        range=ranges,
        lw=3,
    )
    bins_puppi, _, _ = plt.hist(
        np.clip(genfunc(met_d, met_g), binsspace[0], binsspace[-1]),
        bins=binsspace,
        histtype="step",
        label=r"$E_\mathrm{T}^{\mathrm{miss}}$" + f" DistillNet\nResolution: {resolution(met_d,met_g):.4f}",
        range=ranges,
        lw=3,
    )

    plt.legend(fancybox=True, framealpha=0.8, loc="best", prop={"size": 18})

    plt.xlabel(r"$(E_\mathrm{T}^{\mathrm{miss}}-E_\mathrm{T}^{\mathrm{miss,\,gen}})\;/\;E_\mathrm{T}^{\mathrm{miss,\,gen}}$")

    plt.minorticks_on()

    plt.ylabel(r"$N_\mathrm{Events}\;/\;0.1$")

    _xlim, _ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.xlim(ranges)
    plt.ylim(*_ylim)

    plt.savefig(plotdir + "response_smaller" + saveinfo + "__time_" + timestr + ".png", dpi=500, bbox_inches="tight")
    if is_displayplots:
        plt.show()
    else:
        plt.clf()


def main():
    print("Hello, you seem to be running the wrong script")


if __name__ == "__main__":
    main()
