import h5py
import os
import torch
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import vector
import matplotlib
from joblib import dump, load
import pandas as pd

matplotlib.rc("font", size=22, family="serif")
matplotlib.rcParams["text.usetex"] = True


class fl_inputs(Enum):  # features of data
    px = 0
    py = 1
    pz = 2
    e = 3
    d0 = 4
    dz = 5
    puppiw = 6
    charge = 7
    pid_1 = 8
    pid_2 = 9
    pid_3 = 10
    pid_4 = 11
    pid_5 = 12
    pid_6 = 13
    pid_7 = 14
    pid_8 = 15
    # puppiw_2 = 16


class fl_inputs_eta(Enum):  # features of data
    eta = 0
    phi = 1
    pt = 2
    e = 3
    d0 = 4
    dz = 5
    puppiw = 6
    charge = 7
    pid_1 = 8
    pid_2 = 9
    pid_3 = 10
    pid_4 = 11
    pid_5 = 12
    pid_6 = 13
    pid_7 = 14
    pid_8 = 15


class fl_vecs(Enum):
    px = 0
    py = 1
    pz = 2
    e = 3
    wgt = 4


def makelog(arr):
    arr[np.where(arr == 0)] = 1
    return np.log(arr)


def calcresponse(arr, genarr):
    return (np.array(arr) - np.array(genarr)) / np.array(genarr)


def join_and_makedir(parent_path: str, Folder: str):
    new_dir = os.path.join(parent_path, Folder)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    return new_dir


def convertvec_etaphipt(p_vec, is_log: bool = False, is_remove_padding: bool = False):
    vec_input = vector.array({"px": p_vec[:, 0], "py": p_vec[:, 1], "pz": p_vec[:, 2], "energy": p_vec[:, 3]})
    p_vec[:, 0], p_vec[:, 1], p_vec[:, 2] = vec_input.eta, vec_input.phi, vec_input.pt
    if is_log:
        if is_remove_padding:
            p_vec[:, 2], p_vec[:, 3] = np.log(p_vec[:, 2]), np.log(p_vec[:, 3])
            return
        else:
            p_vec[:, 2], p_vec[:, 3] = makelog(p_vec[:, 2]), makelog(p_vec[:, 3])
            return
    return


def makeratio(val_of_bins_x1, val_of_bins_x2):

    ratio = np.divide(val_of_bins_x1, val_of_bins_x2, where=(val_of_bins_x2 != 0))

    error = np.divide(
        val_of_bins_x1 * np.sqrt(val_of_bins_x2) + val_of_bins_x2 * np.sqrt(val_of_bins_x1),
        np.power(val_of_bins_x2, 2),
        where=(val_of_bins_x2 != 0),
    )
    return ratio, error


def gettraindata(
    filedir: str,
    sample: str,
    sample_test: str,
    flist_inputs: list,
    scalerdir: str,
    is_dtrans: bool = False,
    is_remove_padding: bool = True,
    is_standard: bool = True,
    is_min_max_scaler: bool = False,
    is_standard_scaler: bool = True,
    is_makeplots: bool = False,
    is_makeprints: bool = True,
):

    filename = os.path.join(filedir, sample)
    filename_test = os.path.join(filedir, sample_test)
    with h5py.File(filename_test, "r") as f:
        featurelist_test = f["distill_inputs_default"][flist_inputs, :, :]
        _, n_events_test, _ = featurelist_test.shape
    if is_makeprints:
        print(f"Accessing {filename} for training DistillNet")

    with h5py.File(filename, "r") as f:
        if is_makeprints:
            print(f.keys())
        featurelist = f["distill_inputs_default"][flist_inputs, :, :]

        abc_truth = f["data_abc"][4, :, :]

        n_features, n_events, n_particles = featurelist.shape
        features_reshaped = featurelist.reshape(n_features, -1).T
        abc_truth_reshaped = abc_truth.reshape(1, -1).T
        if is_makeprints:
            print(f"Shape of input matrix with padding {features_reshaped.shape}")
            print(f"Shape of abc truth with padding {abc_truth_reshaped.shape}")

        if is_remove_padding:
            if is_makeprints:
                print("Removing padded particles.....")
            padmask_px = np.abs(features_reshaped[:, 0]) > 0.000001
            features_nopad, abc_nopad = features_reshaped[padmask_px], abc_truth_reshaped[padmask_px]
            if is_makeprints:
                print(f"Shape of input matrix without padding {features_nopad.shape}")
                print(f"Shape of abc truth without padding {abc_nopad.shape}")
        else:
            print("No padding removal")
            features_nopad, abc_nopad = features_reshaped, abc_truth_reshaped
            print(f"Shape of input matrix with padding {features_nopad.shape}")
            print(f"Shape of abc truth with padding {abc_nopad.shape}")

        convertvec_etaphipt(features_nopad, is_log=True, is_remove_padding=is_remove_padding)
        if is_dtrans:
            if is_makeprints:
                print("Transforming d0 and dz ")
            features_nopad[:, 4][np.where(features_nopad[:, 4] >= 1)] = 1
            features_nopad[:, 5][np.where(features_nopad[:, 5] >= 1)] = 1
            features_nopad[:, 4][np.where(features_nopad[:, 4] <= -1)] = -1
            features_nopad[:, 5][np.where(features_nopad[:, 5] <= -1)] = -1

        if is_standard:
            if is_standard_scaler:
                scaler = preprocessing.StandardScaler()
                scaler.fit(features_nopad)
                dump(scaler, scalerdir + f"/std_scaler_feat{len(flist_inputs)}.bin", compress=True)
                if is_makeprints:
                    print("Applying standard scaler transformation")
            if is_min_max_scaler:
                scaler = preprocessing.MinMaxScaler()
                scaler.fit(features_nopad)
                dump(scaler, scalerdir + f"/min_max_scaler_feat{len(flist_inputs)}.bin", compress=True)
                print("Applying Min Max scaler transformation")

            features_std = scaler.transform(features_nopad)
        else:
            features_std = features_nopad
        if is_makeplots:
            fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(20, 35))
            fig.suptitle("Features before transformation", fontsize="x-large")
            for j in range(0, len(featurelist[:, 0, 0])):
                ax = plt.subplot(6, 3, j + 1)
                ax.hist(features_nopad[:, j], label=f"{fl_inputs_eta(j).name}, var {j}", bins=20, color="b", alpha=0.5)
                ax.set_yscale("log")
                ax.legend()
            plt.show()

            fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(20, 35))
            fig.suptitle("Features after transformation", fontsize="x-large")
            for j in range(0, len(featurelist[:, 0, 0])):
                ax = plt.subplot(6, 3, j + 1)
                ax.hist(features_std[:, j], label=f"{fl_inputs_eta(j).name}, var {j}", bins=20, color="b", alpha=0.5)
                ax.set_yscale("log")
                ax.legend()
            plt.show()

        return features_std, abc_nopad, n_events, n_events_test


def getdata_deposits(
    filedir: str,
    sample: str,
    flist_inputs: list,
    scalerdir: str,
    device: str,
    min_event: int,
    is_remove_padding: bool = True,
    is_standard: bool = True,
    is_min_max_scaler: bool = True,
    is_standard_scaler: bool = False,
    is_makeplots: bool = False,
    is_dtrans: bool = False,
    is_makeprints: bool = True,
):

    filename = os.path.join(filedir, sample)
    print(f"Accessing {filename} for testing DistillNet energy deposits")
    min_event = int(min_event)
    with h5py.File(filename, "r") as f:
        print(f.keys())
        veclist = [0, 1, 2, 3]
        featurelist = f["distill_inputs_default"][flist_inputs, min_event:, :]
        pu_vec = f["distill_inputs_default"][veclist, min_event:, :]
        nopu_vec = f["data_nopu"][veclist, min_event:, :]
        abc_truth = f["data_abc"][4, min_event:, :]
        puppi_weights = f["data_puppi"][4, min_event:, :]
        print(featurelist.shape)
        print(pu_vec.shape)

        n_features, n_events, n_particles = featurelist.shape
        features_reshaped = featurelist.reshape(n_features, -1).T
        abc_truth_reshaped = abc_truth.reshape(1, -1).T
        puppiw_reshaped = puppi_weights.reshape(1, -1).T
        pu_vec_reshaped = pu_vec.reshape(4, -1).T
        nopu_vec_resphaed = nopu_vec.reshape(4, -1).T
        print(f"Shape of input matrix with padding {features_reshaped.shape}")
        print(f"Shape of abc truth with padding {abc_truth_reshaped.shape}")
        print(f"Shape of PUPPI with padding {puppiw_reshaped.shape}")

        if is_remove_padding:
            print("Removing padded particles.....")
            padmask_px = np.abs(features_reshaped[:, 0]) > 0.000001
            features_nopad, abc_nopad, puppiw_nopad, pu_vec_nopad, nopu_vec_nopad = (
                features_reshaped[padmask_px],
                abc_truth_reshaped[padmask_px],
                puppiw_reshaped[padmask_px],
                pu_vec_reshaped[padmask_px],
                nopu_vec_resphaed[padmask_px],
            )
            print(f"Shape of input matrix without padding {features_nopad.shape}")
            print(f"Shape of abc truth without padding {abc_nopad.shape}")
        else:
            print("No padding removal")
            features_nopad, abc_nopad = features_reshaped, abc_truth_reshaped
            print(f"Shape of input matrix with padding {features_nopad.shape}")
            print(f"Shape of abc truth with padding {abc_nopad.shape}")

        convertvec_etaphipt(features_nopad, is_log=True, is_remove_padding=is_remove_padding)
        convertvec_etaphipt(pu_vec_nopad)
        convertvec_etaphipt(nopu_vec_nopad)
        if is_dtrans:
            if is_makeprints:
                print("Transforming d0 and dz ")
            features_nopad[:, 4][np.where(features_nopad[:, 4] >= 1)] = 1
            features_nopad[:, 5][np.where(features_nopad[:, 5] >= 1)] = 1
            features_nopad[:, 4][np.where(features_nopad[:, 4] <= -1)] = -1
            features_nopad[:, 5][np.where(features_nopad[:, 5] <= -1)] = -1

        if is_standard:
            if is_standard_scaler:
                if torch.cuda.is_available():
                    print("deepthoughtscaler")
                    scaler = load(scalerdir + "std_scaler.bin")
                else:
                    print("portalscaler")
                    scaler = load(scalerdir + "std_scaler_nopup_ens_portal1.bin")
            if is_min_max_scaler:
                scaler = load(scalerdir + "min_max_scaler.bin")

                # dump(scaler, scalerdir + "/min_max_scaler.bin", compress=True)
                print("Applying Min Max scaler transformation")

            features_std = scaler.transform(features_nopad)

        if is_makeplots:
            fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(20, 35))
            fig.suptitle("Features before transformation", fontsize="x-large")
            for j in range(0, len(featurelist[:, 0, 0])):
                ax = plt.subplot(6, 3, j + 1)
                ax.hist(features_nopad[:, j], label=f"{fl_inputs_eta(j).name}, var {j}", bins=20, color="b", alpha=0.5)
                ax.set_yscale("log")
                ax.legend()
            plt.show()

            fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(20, 35))
            fig.suptitle("Features after transformation", fontsize="x-large")
            for j in range(0, len(featurelist[:, 0, 0])):
                ax = plt.subplot(6, 3, j + 1)
                ax.hist(features_std[:, j], label=f"{fl_inputs_eta(j).name}, var {j}", bins=20, color="b", alpha=0.5)
                ax.set_yscale("log")
                ax.legend()
            plt.show()
        return features_std, abc_nopad, puppiw_nopad, pu_vec_nopad, nopu_vec_nopad, n_events


def make_lossplot(
    losslist: list,
    validationloss: list,
    plotdir: str,
    plotdir_pdf: str,
    saveinfo: str,
    timestr: str,
    is_savefig: bool = False,
    is_displayplots: bool = False,
):
    plt.figure(figsize=(8, 7))
    # plot loss over epochs
    plt.plot(np.arange(0, len(losslist)), losslist, color="b", label="Train loss", zorder=1)
    plt.scatter(np.arange(0, len(validationloss)), validationloss, s=10, color="r", label="Validation loss", zorder=2)
    _xlim, _ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.minorticks_on()

    plt.ylim(*_ylim)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(fancybox=True, framealpha=0.8, loc="upper right", prop={"size": 20})
    sample_file_name0 = "Loss"
    if is_savefig:
        plt.savefig(plotdir + sample_file_name0 + saveinfo + "__time_" + timestr + ".png", dpi=400, bbox_inches="tight")
        plt.savefig(plotdir_pdf + sample_file_name0 + saveinfo + "__time_" + timestr + ".pdf", bbox_inches="tight")
    if is_displayplots:
        plt.show()
    else:
        plt.clf()
    plt.close()
    return


def do_weightpred(test_loader, model, device):
    with torch.no_grad():
        weight_prediction = []
        for i, (features, labels) in enumerate(test_loader):
            features = features.to(device)
            labels = labels.to(device)

            _weight_prediction = model.forward(features)
            op = _weight_prediction.to("cpu").numpy()
            op = np.squeeze(op)
            weight_prediction.append(op)

        weight_prediction = np.concatenate(weight_prediction)
    return weight_prediction


def make_histoweight(
    test: list,
    test_loader,
    model,
    device: str,
    plotdir: str,
    plotdir_pdf: str,
    saveinfo: str,
    timestr: str,
    is_savefig: bool = False,
    is_displayplots: bool = False,
):
    truth = test[1]
    truth = np.concatenate(truth)
    weight_prediction = do_weightpred(test_loader, model, device)

    plt.figure(figsize=(10, 7))

    bins, xb, xr = plt.hist(truth, bins=20, range=(0, 1), label="ABC Net truth weights", histtype="step", lw=1.5)
    bins_distil, _, _ = plt.hist(weight_prediction, bins=20, range=(0, 1), label="predicted weights", histtype="step", lw=1.5)

    plt.yscale("log")
    _xlim, _ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.xlim(*_xlim)
    plt.ylim(*_ylim)
    ratiolast = bins_distil[-1] / bins[-1]
    ratiofirst = bins_distil[0] / bins[0]
    plt.plot([], [], " ", label=f"first bin (between 0 and 0.05) ratio: {ratiofirst*100:.2f} percent")
    plt.plot([], [], " ", label=f"last bin (between 0.95 and 1) ratio: {ratiolast*100:.2f} percent")

    plt.vlines(xb, ymin=0, ymax=_ylim[1], color="black", alpha=0.1)
    plt.legend(prop={"size": 15}, fancybox=True, framealpha=0.8, loc="upper center")
    plt.xlabel("Weight")
    plt.ylabel("Number of particles")
    sample_file_name1 = "wgts"
    if is_savefig:
        plt.savefig(plotdir + sample_file_name1 + saveinfo + "__time_" + timestr + ".png", dpi=400, bbox_inches="tight")
        plt.savefig(plotdir_pdf + sample_file_name1 + saveinfo + "__time_" + timestr + ".pdf", bbox_inches="tight")
    if is_displayplots:
        plt.show()
    else:
        plt.clf()
    plt.close()
    return ratiolast


def make_metplots(
    abcMET: list,
    puppiMET: list,
    distilMET: list,
    res_abc: float,
    res_model: float,
    res_puppi: float,
    plotdir: str,
    plotdir_pdf: str,
    saveinfo: str,
    timestr: str,
    is_savefig: bool = True,
    is_displayplots: bool = False,
):

    figure, ax = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [0.8, 0.2]})
    binsspace = np.arange(0, 160, 8)
    bins_abc, xb, _ = ax[0].hist(
        np.clip(abcMET, binsspace[0], binsspace[-1]),
        bins=binsspace,
        histtype="step",
        label=r"$E_\mathrm{T}^{\mathrm{miss}}$ GNN",
        range=(0, 300),
        lw=1.7,
    )
    bins_puppi, _, _ = ax[0].hist(
        np.clip(puppiMET, binsspace[0], binsspace[-1]),
        bins=xb,
        histtype="step",
        label=r"$E_\mathrm{T}^{\mathrm{miss}}$ PUPPI",
        lw=1.4,
    )
    bins_distil, _, _ = ax[0].hist(
        np.clip(distilMET, binsspace[0], binsspace[-1]),
        bins=xb,
        histtype="step",
        label=r"$E_\mathrm{T}^{\mathrm{miss}}$ DistillNet",
        lw=1.7,
    )

    ax[0].set_xticks([])
    ax[0].set_yticks([200, 400, 600, 800, 1000, 1200, 1400])
    ax[1].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax[1].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax[0].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax[0].plot(
        [],
        [],
        " ",
        label=f"\nResolution GNN: {res_abc:.4f}"
        + f"\nResolution Model: {res_model:.4f}"
        + f"\nResolution PUPPI: {res_puppi:.4f}",
    )

    ax[1].set_yticks([0.8, 1.2])
    figure.subplots_adjust(hspace=0)

    ax[0].set_ylabel(r"$N_\mathrm{Events}\;/\;8\,\mathrm{GeV}$")

    ax[1].set_ylabel(r"Ratio")

    _xlim, _ylim = ax[0].get_xlim(), ax[0].get_ylim()
    ax[0].set_xlim(0, 150)
    ax[1].set_xlim(0, 150)
    ax[0].set_ylim(*_ylim)
    ax[1].set_ylim(0.7, 1.3)
    ax[1].hlines(1, *_xlim, color="black")

    ratio3, error3 = makeratio(bins_distil, bins_abc)
    ratio4, error4 = makeratio(bins_puppi, bins_abc)

    bincenter = 5 + xb[:-1]
    ax[1].errorbar(
        bincenter,
        ratio4,
        yerr=error4,
        fmt="^",
        c="tab:orange",
        ecolor="tab:orange",
        capsize=5,
        ms=6,
        label=f"Ratio PUPPI" + r"and GNN $E_\mathrm{T}^{\mathrm{miss}}$",
    )

    ax[1].errorbar(
        bincenter,
        ratio3,
        yerr=error3,
        fmt=".k",
        c="g",
        ecolor="g",
        capsize=5,
        ms=10,
        label=f"Ratio DistillNet" + r"and GNN $E_\mathrm{T}^{\mathrm{miss}}$",
    )

    ax[1].set_xlabel(r"Missing Transverse Energy $E_\mathrm{T}^{\mathrm{miss}}$ in GeV")
    ax[0].legend(fancybox=True, framealpha=0.8, loc="best", prop={"size": 14})
    sample_file_name1 = "met_plot"
    if is_savefig:
        plt.savefig(plotdir + sample_file_name1 + saveinfo + "__time_" + timestr + ".png", dpi=400, bbox_inches="tight")
        plt.savefig(plotdir_pdf + sample_file_name1 + saveinfo + "__time_" + timestr + ".pdf", bbox_inches="tight")
    if is_displayplots:
        plt.show()
    else:
        plt.clf()
    plt.close()
    return


def make_histoweight_mod(
    predictions,
    truth,
    puppiw,
    res_model,
    res_abc,
    res_puppi,
    plotdir,
    plotdir_pdf,
    saveinfo,
    timestr,
    sample,
    is_savefig: bool = True,
    is_displayplots: bool = False,
):
    predictions = np.concatenate(predictions)
    truth = np.concatenate(truth)
    puppiw = np.concatenate(puppiw)

    def makehistograms(predictions, truth, puppiw):
        bins = 20
        ranges = (0, 1)
        bin1, xb, xr = plt.hist(truth, bins=bins, range=ranges, label=f"GNN truth weights", histtype="step", lw=2)
        bins_pup, _, _ = plt.hist(
            puppiw, bins=bins, range=ranges, label="PUPPI weights ", histtype="step", lw=1.7, color="#ff7f0e"
        )
        bin2, _, _ = plt.hist(predictions, bins=bins, range=ranges, label=f"DistillNet weights", histtype="step", lw=2, color="g")
        return bin1, bin2, xb, xr

    plt.figure(figsize=(10, 6))

    bins1, bins2, xb, xr = makehistograms(predictions, truth, puppiw)
    ratiolast = bins2[-1] / bins1[-1]
    ratiofirst = bins2[0] / bins1[0]
    plt.plot(
        [],
        [],
        " ",
        label=f"First bin ratio:"
        + r"$\,$"
        + f"{ratiofirst*100:.2f}"
        + r"$\%$"
        + f"\nLast bin ratio: {ratiolast*100:.2f}"
        + r"$\%$"
        + f"\nResolution GNN: {res_abc:.4f}"
        + f"\nResolution Model: {res_model:.4f}"
        + f"\nResolution PUPPI: {res_puppi:.4f}",
    )
    plt.yscale("log")
    _xlim, _ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.xlim(*_xlim)
    plt.xlim(0, 1)
    plt.ylim(*_ylim)

    plt.minorticks_on()
    plt.legend(
        prop={"size": 18},
        fancybox=True,
        framealpha=0.8,
        loc="upper center",
        bbox_to_anchor=(1.23, 1),
        bbox_transform=plt.gca().transAxes,
        borderaxespad=0,
    )
    plt.xlabel(r"Particle Weight $w$")
    plt.ylabel(r"$N_\mathrm{Particles}\;/\;0.05$")
    plt.minorticks_on()

    print(f"last bin (between 0.95 and 1) ratio: { bins2[-1] / bins1[-1]:.3f}")
    sample_file_name1 = "wgts"
    if is_savefig:
        plt.savefig(
            plotdir + sample_file_name1 + saveinfo + sample.replace(".h5", "") + "__time_" + timestr + ".png",
            dpi=400,
            bbox_inches="tight",
        )
        plt.savefig(
            plotdir_pdf + sample_file_name1 + saveinfo + sample.replace(".h5", "") + "__time_" + timestr + ".pdf",
            bbox_inches="tight",
        )
    if is_displayplots:
        plt.show()
    else:
        plt.clf()
    plt.close()
    return ratiolast


def corinputs(dataframe, index):
    npdf = dataframe.to_numpy()
    npdf = np.delete(npdf[index], 0)
    npdf = npdf[~pd.isnull(npdf)]
    return npdf


def makeratio(val_of_bins_x1, val_of_bins_x2):
    ratio = np.divide(val_of_bins_x1, val_of_bins_x2, where=(val_of_bins_x2 != 0))
    error = np.divide(
        val_of_bins_x1 * np.sqrt(val_of_bins_x2) + val_of_bins_x2 * np.sqrt(val_of_bins_x1),
        np.power(val_of_bins_x2, 2),
        where=(val_of_bins_x2 != 0),
    )
    return ratio, error


def resolution_response(arr):
    q_75_abc = np.quantile(arr, 0.75)
    q_25_abc = np.quantile(arr, 0.25)
    resolutions = (q_75_abc - q_25_abc) / 2
    return resolutions


def plot_jetresolution(
    responseabc,
    responsedistil,
    responsepuppi,
    results_dir: str,
    results_dir_pdf: str,
    ptcut: int,
    is_savefig: bool = True,
    is_corr: bool = False,
    is_ptcorr: bool = False,
):
    print("responseabcres: ", resolution_response(responseabc))
    print("Distiljetres:", resolution_response(responsedistil))
    print("puppijetres:", resolution_response(responsepuppi))
    r1, r2, r4 = resolution_response(responseabc), resolution_response(responsedistil), resolution_response(responsepuppi)

    figure = plt.figure(figsize=(8, 7))

    binsspace = 40
    xmaxbin = 0.8
    xminbin = -0.8
    range = (xminbin, xmaxbin)
    bins_abc, xb, _ = plt.hist(
        responseabc,
        bins=binsspace,
        histtype="step",
        label=r"$E_\mathrm{Jet,\,reco}$" + f" GNN\nResolution: {r1:.4f}",
        lw=1.5,
        range=range,
    )
    bins_puppi, xb, _ = plt.hist(
        responsepuppi,
        color="#ff7f0e",
        bins=binsspace,
        histtype="step",
        label=r"$E_\mathrm{Jet,\,reco}$" + f" PUPPI\nResolution: {r4:.4f}",
        lw=1.5,
        range=range,
    )
    bins_distil, _, _ = plt.hist(
        responsedistil,
        color="green",
        bins=xb,
        histtype="step",
        label=r"$E_\mathrm{Jet,\,reco}$" + f" DistillNet\nResolution: {r2:.4f}",
        lw=1.5,
        range=range,
    )
    figure.subplots_adjust(hspace=0)

    _xlim, _ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.xlim(xminbin, xmaxbin)
    plt.ylim(*_ylim)
    plt.vlines(0, ymin=0, ymax=_ylim[1], color="black", alpha=0.99, linestyles="dotted")
    plt.vlines(
        np.mean(responsedistil),
        ymin=0,
        ymax=_ylim[1],
        color="green",
        alpha=0.99,
        linestyles="dotted",
        label=f"DistillNet mean\n response: {np.mean(responsedistil):.5f}",
    )
    plt.minorticks_on()

    plt.ylabel(r"$N_\mathrm{Jets,\,reco}\;/\; 0.04$")
    plt.xlabel("(recojet - genjet) / genjet")
    plt.xlabel(r"Jet $E_\mathrm{reco}$-Jet $E_\mathrm{gen}$ / $\mathrm{Jet} E_\mathrm{gen}$")
    plt.xlabel(r"$(E_\mathrm{Jet,\,reco}-E_\mathrm{Jet,\,gen})\;/\;E_\mathrm{Jet,\,gen}$")
    plt.legend(fancybox=True, framealpha=0.8, loc="best", prop={"size": 16})
    if is_savefig:
        if is_corr:
            print(f"corr mean is {np.mean(responsedistil)}")
            plt.savefig(results_dir + f"Jet_response_corr_ptcut{ptcut}", dpi=500, bbox_inches="tight")
            plt.savefig(results_dir_pdf + f"Jet_response_corr_ptcut{ptcut}.pdf", bbox_inches="tight")
        elif is_ptcorr:
            plt.savefig(results_dir + f"Jet_response_PTcorr_ptcut{ptcut}", dpi=500, bbox_inches="tight")
            plt.savefig(results_dir_pdf + f"Jet_response_PTcorr_ptcut{ptcut}.pdf", bbox_inches="tight")
        else:
            print(f"mean is {np.mean(responsedistil)}")
            plt.savefig(results_dir + f"Jet_response_ptcut{ptcut}", dpi=500, bbox_inches="tight")
            plt.savefig(results_dir_pdf + f"Jet_response_ptcut{ptcut}.pdf", bbox_inches="tight")
    plt.clf()
    return


def plot_jetenergy(
    abcjetE, puppijetE, distiljetE, genjetE, results_dir: str, results_dir_pdf: str, ptcut: int, is_savefig: bool = True
):
    figure = plt.figure(figsize=(16, 8))

    binsspace = 50
    xmaxbin = 550
    xminbin = 0
    range = (xminbin, xmaxbin)
    bins_abc, xb, _ = plt.hist(
        np.clip(abcjetE, 0, xmaxbin), bins=binsspace, histtype="step", label="Jet Energy GNN", lw=2, range=range
    )
    bins_puppi, xb, _ = plt.hist(
        np.clip(puppijetE, 0, xmaxbin), bins=binsspace, histtype="step", label="Jet Energy PUPPI true", lw=2, range=range
    )

    bins_distil, _, _ = plt.hist(
        np.clip(distiljetE, 0, xmaxbin), bins=xb, histtype="step", label="Jet Energy distilNet", lw=2, range=range
    )
    bins_gen, _, _ = plt.hist(
        np.clip(genjetE, 0, xmaxbin), bins=xb, color="black", histtype="step", label="Jet Energy gen", lw=1, range=range
    )

    _xlim, _ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    plt.xlim(xminbin, xmaxbin)
    plt.ylim(*_ylim)

    plt.vlines(xb, ymin=0, ymax=_ylim[1], color="black", alpha=0.1)
    plt.ylabel("Number of jets per bin")
    plt.xlabel("Jet energy in GeV")
    plt.legend(fancybox=True, framealpha=0.1, loc="best", prop={"size": 20})
    if is_savefig:
        plt.savefig(results_dir + f"Jet_energy_ptcut{ptcut}", dpi=500, bbox_inches="tight")
        plt.savefig(results_dir_pdf + f"Jet_energy_ptcut{ptcut}.pdf", bbox_inches="tight")
    plt.clf()
    return


def plot_jetratio(
    abcjetE,
    puppijetE_true,
    distiljetE,
    results_dir: str,
    results_dir_pdf: str,
    ptcut: int,
    is_savefig: bool = True,
    is_corr: bool = False,
    is_ptcorr: bool = False,
):

    figure, ax = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [0.8, 0.2]})
    binsspace = 25
    xmaxbin = 400
    xminbin = 0
    range = (xminbin, xmaxbin)
    bins_abc, xb, _ = ax[0].hist(
        np.clip(abcjetE, 0, xmaxbin), bins=binsspace, histtype="step", label=r"$E_\mathrm{Jet,\,reco}\;$ GNN", range=range, lw=1.7
    )
    bins_puppi, _, _ = ax[0].hist(
        np.clip(puppijetE_true, 0, xmaxbin),
        bins=xb,
        histtype="step",
        label=r"$E_\mathrm{Jet,\,reco}\;$ PUPPI",
        range=range,
        lw=1.4,
    )
    bins_distil, _, _ = ax[0].hist(
        np.clip(distiljetE, 0, xmaxbin),
        bins=xb,
        histtype="step",
        label=r"$E_\mathrm{Jet,\,reco}\;$ DistillNet",
        range=range,
        lw=1.7,
    )

    ax[1].set_yticks([0.8, 1.2])
    figure.subplots_adjust(hspace=0)
    ax[0].set_xticks([])
    ax[0].set_ylabel(r"$N_\mathrm{Jets,\,reco}\;/\;16\,\mathrm{GeV}$")
    ax[1].set_ylabel("Ratio")

    _xlim, _ylim = ax[0].get_xlim(), ax[0].get_ylim()
    ax[0].set_xlim(0, xmaxbin)
    ax[1].set_xlim(0, xmaxbin)
    ax[0].set_ylim(*_ylim)
    ax[1].set_ylim(0.7, 1.3)

    ax[1].hlines(1, *_xlim, color="black")

    ratio3, error3 = makeratio(bins_distil, bins_abc)
    ratio4, error4 = makeratio(bins_puppi, bins_abc)

    bincenter = 5 + xb[:-1]

    ax[1].errorbar(
        bincenter,
        ratio3,
        yerr=error3,
        fmt=".k",
        c="g",
        ecolor="g",
        capsize=5,
        ms=15,
        label="Ratio between distilNet and GNN jet energy",
    )
    ax[1].errorbar(
        bincenter,
        ratio4,
        yerr=error4,
        fmt="^",
        c="tab:orange",
        ecolor="tab:orange",
        capsize=5,
        ms=6,
        label="Ratio between PUPPI and GNN jet energy",
    )

    ax[1].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax[1].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax[0].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    ax[1].set_xlabel(r"Reconstruced Jet Energy $E_\mathrm{Jet,\,reco}$ in GeV")
    ax[0].legend(fancybox=True, framealpha=0.8, loc="upper right", prop={"size": 18})  # ,bbox_to_anchor=(0.98,1))
    if is_savefig:
        if is_corr:
            plt.savefig(results_dir_pdf + f"Jet_energy_ratio_ptcut{ptcut}_corr.pdf", bbox_inches="tight")
            plt.savefig(results_dir + f"Jet_energy_ratio_ptcut{ptcut}_corr", dpi=500, bbox_inches="tight")
        elif is_ptcorr:
            plt.savefig(results_dir_pdf + f"Jet_energy_ratio_ptcut{ptcut}_PTcorr.pdf", bbox_inches="tight")
            plt.savefig(results_dir + f"Jet_energy_ratio_ptcut{ptcut}_PTcorr", dpi=500, bbox_inches="tight")
        else:
            plt.savefig(results_dir_pdf + f"Jet_energy_ratio_ptcut{ptcut}.pdf", bbox_inches="tight")
            plt.savefig(results_dir + f"Jet_energy_ratio_ptcut{ptcut}", dpi=500, bbox_inches="tight")
    plt.clf()
    return


def make_depositplots(
    nparticles,
    nstart,
    pu_dat,
    npu_dat,
    abcw,
    puppiw,
    predictions,
    plotdir,
    plotdir_pdf,
    is_savefig: bool = True,
    is_abc_puppisave: bool = False,
):
    pts = pu_dat[:, 2][nstart:nparticles]
    etas = pu_dat[:, 0][nstart:nparticles]
    phis = pu_dat[:, 1][nstart:nparticles]

    pts_nopu = npu_dat[:, 2][nstart:nparticles]
    etas_nopu = npu_dat[:, 0][nstart:nparticles]
    phis_nopu = npu_dat[:, 1][nstart:nparticles]
    nbins = 35
    puppi_w_sub = np.ravel(puppiw[nstart:nparticles])
    abc_w_sub = np.ravel(abcw[nstart:nparticles])
    distill_w_sub = np.ravel(predictions[nstart:nparticles])
    print(puppi_w_sub.shape)
    print(abc_w_sub.shape)
    print(distill_w_sub.shape)
    etabins = np.linspace(-4.7, 4.7, nbins)
    phibins = np.linspace(-3.1416, 3.1416, nbins)

    H_ptweighted_puppi, xedges, yedges = np.histogram2d(etas, phis, weights=puppi_w_sub * pts, bins=(etabins, phibins))
    fig, ax = plt.subplots(figsize=[16, 8])
    img = ax.imshow(
        H_ptweighted_puppi,
        interpolation="nearest",
        origin="lower",
        extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]],
        aspect="0.65",
    )
    cb = plt.colorbar(img, fraction=0.046, label="Particles")
    cb.ax.tick_params(labelsize=13)
    text = cb.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(size=18)
    text.set_font_properties(font)
    plt.xlabel("$\\phi$", fontsize=18)
    plt.ylabel("$\\eta$", fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    etabins = np.linspace(-5.2, 5.2, nbins)
    phibins = np.linspace(-3.1416, 3.1416, nbins)

    H_ptweighted_model, xedges, yedges = np.histogram2d(etas, phis, weights=abc_w_sub * pts, bins=(etabins, phibins))
    fig, ax = plt.subplots(figsize=[16, 8])
    img = ax.imshow(
        H_ptweighted_model,
        interpolation="nearest",
        origin="lower",
        extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]],
        aspect="0.65",
    )
    cb = plt.colorbar(img, fraction=0.046, label="Particles")
    cb.ax.tick_params(labelsize=13)
    text = cb.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(size=18)
    text.set_font_properties(font)
    plt.xlabel("$\\phi$", fontsize=18)
    plt.ylabel("$\\eta$", fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.clf()

    etabins = np.linspace(-5.2, 5.2, nbins)
    phibins = np.linspace(-3.1416, 3.1416, nbins)

    H_ptweighted_distill, xedges, yedges = np.histogram2d(etas, phis, weights=distill_w_sub * pts, bins=(etabins, phibins))
    fig, ax = plt.subplots(figsize=[16, 8])
    img = ax.imshow(
        H_ptweighted_distill,
        interpolation="nearest",
        origin="lower",
        extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]],
        aspect="0.65",
    )
    cb = plt.colorbar(img, fraction=0.046, label="Particles")
    cb.ax.tick_params(labelsize=13)
    text = cb.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(size=18)
    text.set_font_properties(font)
    plt.xlabel("$\\phi$", fontsize=18)
    plt.ylabel("$\\eta$", fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.clf()

    etabins = np.linspace(-4.7, 4.7, nbins)
    phibins = np.linspace(-3.1416, 3.1416, nbins)

    H_nopu, xedges, yedges = np.histogram2d(etas_nopu, phis_nopu, bins=(etabins, phibins))
    fig, ax = plt.subplots(figsize=[16, 8])
    img = ax.imshow(
        H_nopu, interpolation="nearest", origin="lower", extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]], aspect="0.65"
    )
    cb = plt.colorbar(img, fraction=0.046, label="Particles")
    cb.ax.tick_params(labelsize=13)
    text = cb.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(size=18)
    text.set_font_properties(font)
    plt.xlabel("$\\phi$", fontsize=18)
    plt.ylabel("$\\eta$", fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.clf()

    etabins = np.linspace(-5.2, 5.2, nbins)
    phibins = np.linspace(-3.1416, 3.1416, nbins)

    H_ptweighted_nopu, xedges, yedges = np.histogram2d(etas_nopu, phis_nopu, weights=pts_nopu, bins=(etabins, phibins))
    fig, ax = plt.subplots(figsize=[16, 8])
    img = ax.imshow(
        H_ptweighted_nopu,
        interpolation="nearest",
        origin="lower",
        extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]],
        aspect="0.65",
    )
    cb = plt.colorbar(img, fraction=0.046, label="Particles")
    cb.ax.tick_params(labelsize=13)
    text = cb.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(size=18)
    text.set_font_properties(font)
    plt.xlabel("$\\phi$", fontsize=18)
    plt.ylabel("$\\eta$", fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.clf()

    etabins = np.linspace(-5.2, 5.2, nbins)
    phibins = np.linspace(-3.1416, 3.1416, nbins)

    fig, ax = plt.subplots(figsize=[16, 8])
    img = ax.imshow(
        H_ptweighted_model / H_ptweighted_nopu,
        interpolation="nearest",
        origin="lower",
        extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]],
        aspect="0.65",
        vmin=0,
        vmax=2,
        cmap=plt.get_cmap("RdBu"),
    )
    cb = plt.colorbar(img, fraction=0.046, label="GNN / NoPU Sample")
    cb.ax.tick_params(labelsize=13)
    text = cb.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(size=20)
    text.set_font_properties(font)
    plt.xlabel("$\\phi$", fontsize=20)
    plt.ylabel("$\\eta$", fontsize=20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if is_abc_puppisave:
        plt.savefig(plotdir_pdf + "2D_ptweighted_ratio_ABCNet_nopu.pdf", bbox_inches="tight")
        plt.savefig(plotdir + "2D_ptweighted_ratio_ABCNet_nopu.png", dpi=400, bbox_inches="tight")
    plt.clf()
    etabins = np.linspace(-5.2, 5.2, nbins)
    phibins = np.linspace(-3.1416, 3.1416, nbins)

    fig, ax = plt.subplots(figsize=[16, 8])
    img = ax.imshow(
        H_ptweighted_distill / H_ptweighted_nopu,
        interpolation="nearest",
        origin="lower",
        extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]],
        aspect="0.65",
        vmin=0,
        vmax=2,
        cmap=plt.get_cmap("RdBu"),
    )
    cb = plt.colorbar(img, fraction=0.046, label="DistillNet / NoPU Sample")
    cb.ax.tick_params(labelsize=13)
    text = cb.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(size=20)
    text.set_font_properties(font)
    plt.xlabel("$\\phi$", fontsize=20)
    plt.ylabel("$\\eta$", fontsize=20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if is_savefig:
        plt.savefig(plotdir_pdf + "2D_ptweighted_ratio_DistillNet_bestmodel.pdf", bbox_inches="tight")
        plt.savefig(plotdir + "2D_ptweighted_ratio_DistillNet_bestmodel.png", dpi=400, bbox_inches="tight")
    plt.clf()

    etabins = np.linspace(-5.2, 5.2, nbins)
    phibins = np.linspace(-3.1416, 3.1416, nbins)
    fig, ax = plt.subplots(figsize=[16, 8])
    img = ax.imshow(
        H_ptweighted_puppi / H_ptweighted_nopu,
        interpolation="nearest",
        origin="lower",
        extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]],
        aspect="0.65",
        vmin=0,
        vmax=2,
        cmap=plt.get_cmap("RdBu"),
    )
    cb = plt.colorbar(img, fraction=0.046, label="PUPPI / NoPU Sample")
    cb.ax.tick_params(labelsize=13)
    text = cb.ax.yaxis.label
    font = matplotlib.font_manager.FontProperties(size=20)
    text.set_font_properties(font)
    plt.xlabel("$\\phi$", fontsize=20)
    plt.ylabel("$\\eta$", fontsize=20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if is_abc_puppisave:
        plt.savefig(plotdir_pdf + "2D_ptweighted_ratio_Puppi_nopu.pdf", bbox_inches="tight")
        plt.savefig(plotdir + "2D_ptweighted_ratio_Puppi_nopu.png", dpi=400, bbox_inches="tight")
    plt.clf()
    return


def derive_corrections(
    lower_bound: int,
    upper_bound: int,
    jetenergy_algorithm: list,
    jetenergy_gen: list,
    ):
    print(f"Selection between{lower_bound} and {upper_bound}GeV")

    selected_jet_energy_algo, mask = apply_mask_bounds(lower_bound, upper_bound, jetenergy_algorithm)
    selected_jet_energy_gen = jetenergy_gen[mask]
    ratio = np.mean(selected_jet_energy_algo / selected_jet_energy_gen)
    return ratio


def derive_corrections_pt(lower_bound: int, upper_bound: int, jetpt_algo: list, jetpt_gen: list):
    print(f"Pt selection between{lower_bound} and {upper_bound}GeV")
    selected_jet_pt_algo, mask = apply_mask_bounds(lower_bound, upper_bound, jetpt_algo)
    selected_jet_pt_gen = jetpt_gen[mask]
    ratio = np.mean(selected_jet_pt_algo / selected_jet_pt_gen)
    return ratio


def apply_jetcorrections(corrections_intervals: list, matched_algo, gen_matched_algo, is_prints: bool = False):
    matched_algo_copy = matched_algo
    for lower_bound, upper_bound in corrections_intervals:
        ratio = derive_corrections(lower_bound, upper_bound, matched_algo, gen_matched_algo, is_histoplots=False)
        if is_prints:
            print(matched_algo)
        matched_algo = np.where(
            (matched_algo_copy >= lower_bound) & (matched_algo_copy <= upper_bound), matched_algo / ratio, matched_algo
        )
        if is_prints:
            print(matched_algo)
    response_corrected = calcresponse(matched_algo, gen_matched_algo)
    return matched_algo, response_corrected 

def apply_jetcorrections_pt(
    corrections_intervals: list,
    matched_pt_algo: list,
    matched_pt_gen: list,
    matched_algo: list,
    gen_matched_algo: list,
    is_prints: bool = False,
):
    matched_pt_algo_copy = matched_pt_algo
    for lower_bound, upper_bound in corrections_intervals:
        ratio = derive_corrections_pt(lower_bound, upper_bound, matched_pt_algo, matched_pt_gen)
        if is_prints:
            print(matched_algo)
        matched_algo = np.where(
            (matched_pt_algo_copy >= lower_bound) & (matched_pt_algo_copy <= upper_bound), matched_algo / ratio, matched_algo
        )  
        if is_prints:
            print(matched_algo)
    response_corrected = calcresponse(matched_algo, gen_matched_algo)
    return (
        matched_algo,
        response_corrected,
    ) 

def apply_mask_bounds(lower_bound, upper_bound, jetenergies):
    mask1 = np.abs(lower_bound) <= jetenergies
    mask2 = jetenergies <= np.abs(upper_bound)
    combmask = np.logical_and(mask1, mask2)
    return jetenergies[combmask], combmask


def make_alljetplots(df_jetdata_abc_puppi, plotdir, plotdir_pdf, ptcut, is_savefig: bool = True, is_prints: bool = False):

    print(df_jetdata_abc_puppi)
    abcjetE = corinputs(df_jetdata_abc_puppi, 0)
    puppijetE = corinputs(df_jetdata_abc_puppi, 1)
    genjetE_abc = corinputs(df_jetdata_abc_puppi, 2)
    responseabc = corinputs(df_jetdata_abc_puppi, 3)
    responsepuppi = corinputs(df_jetdata_abc_puppi, 4)
    abcjetE_matched = corinputs(df_jetdata_abc_puppi, 5)
    genjetE_abc_matched = corinputs(df_jetdata_abc_puppi, 6)
    puppijetE_matched = corinputs(df_jetdata_abc_puppi, 7)
    genjetE_puppi_matched = corinputs(df_jetdata_abc_puppi, 8)

    distiljetE = corinputs(df_jetdata_abc_puppi, 9)
    # genjetE = corinputs(df_jetdata_abc_puppi,10)
    responsedistil = corinputs(df_jetdata_abc_puppi, 10)
    distilljetE_matched = corinputs(df_jetdata_abc_puppi, 11)
    genjetE_distill_matched = corinputs(df_jetdata_abc_puppi, 12)

    abcjetpt_matched = corinputs(df_jetdata_abc_puppi, 13)
    genjetpt_abc_matched = corinputs(df_jetdata_abc_puppi, 14)
    puppijetpt_matched = corinputs(df_jetdata_abc_puppi, 15)
    genjetpt_puppi_matched = corinputs(df_jetdata_abc_puppi, 16)

    distilljetpt_matched = corinputs(df_jetdata_abc_puppi, 17)
    genjetpt_distill_matched = corinputs(df_jetdata_abc_puppi, 18)

    corr_intervals = [(i, i + 100) for i in range(0, 601, 100)]
    corr_intervals.append((700, 4500))
    # CHANGED GENJETE to GENJETE_ABC
    plot_jetresolution(
        responseabc, responsedistil, responsepuppi, plotdir, plotdir_pdf, ptcut, is_savefig=is_savefig, is_corr=False
    )

    distilljetE_matched_corrected, response_distill_corrected = apply_jetcorrections(
        corr_intervals, distilljetE_matched, genjetE_distill_matched, is_prints=is_prints
    )
    abcjetE_matched_corrected, responseabc = apply_jetcorrections(
        corr_intervals, abcjetE_matched, genjetE_abc_matched, is_prints=is_prints
    )
    puppijetE_matched_corrected, responsepuppi = apply_jetcorrections(
        corr_intervals, puppijetE_matched, genjetE_puppi_matched, is_prints=is_prints
    )

    distilljetE_matched_ptcorrected, response_distill_ptcorr = apply_jetcorrections_pt(
        corr_intervals,
        distilljetpt_matched,
        genjetpt_distill_matched,
        distilljetE_matched,
        genjetE_distill_matched,
        is_prints=is_prints,
    )
    abcjetE_matched_ptcorrected, responseabc_ptcorr = apply_jetcorrections_pt(
        corr_intervals, abcjetpt_matched, genjetpt_abc_matched, abcjetE_matched, genjetE_abc_matched, is_prints=is_prints
    )
    puppijetE_matched_ptcorrected, responsepuppi_ptcorr = apply_jetcorrections_pt(
        corr_intervals, puppijetpt_matched, genjetpt_puppi_matched, puppijetE_matched, genjetE_puppi_matched, is_prints=is_prints
    )

    plot_jetresolution(
        responseabc, response_distill_corrected, responsepuppi, plotdir, plotdir_pdf, ptcut, is_savefig=is_savefig, is_corr=True
    )
    plot_jetresolution(
        responseabc_ptcorr,
        response_distill_ptcorr,
        responsepuppi_ptcorr,
        plotdir,
        plotdir_pdf,
        ptcut,
        is_savefig=is_savefig,
        is_ptcorr=True,
    )

    plot_jetenergy(abcjetE, puppijetE, distiljetE, genjetE_abc, plotdir, plotdir_pdf, ptcut, is_savefig=is_savefig)
    plot_jetratio(abcjetE, puppijetE, distiljetE, plotdir, plotdir_pdf, ptcut, is_savefig=is_savefig, is_corr=False)
    plot_jetratio(
        abcjetE_matched_corrected,
        puppijetE_matched_corrected,
        distilljetE_matched_corrected,
        plotdir,
        plotdir_pdf,
        ptcut,
        is_savefig=is_savefig,
        is_corr=True,
    )
    plot_jetratio(
        abcjetE_matched_ptcorrected,
        puppijetE_matched_ptcorrected,
        distilljetE_matched_corrected,
        plotdir,
        plotdir_pdf,
        ptcut,
        is_savefig=is_savefig,
        is_ptcorr=True,
    )
    return
