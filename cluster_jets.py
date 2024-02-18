"""
THis code is designed to run in batch mode on a computing cluster due to long computational times for each event,
as the clustering is highly resource intensive. Therefore, a range of i.e. 1000 events can be
divided into 10 subintervals that can be then executed as 
10 parallel jobs as opposed to one big job with 1000 events. Afterwards, all 
subdataframes can then be combined into one dataframe.
In previous versions the GNN was called ABCNet, therefore all the "abc" variables.
"""

from jetclustering_helpers import JetEvent, resolution
import numpy as np
import os
import pandas as pd
from distillnet_setup import load_bestmodel
from distillnet_config import hparams, fl_inputs, bool_val, dirs
from data_helpers import makescaler, join_and_makedir
from tqdm import tqdm
import time
import argparse

timestr = time.strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser(description="Parsing Event_args")

# Step 3: Define the command-line arguments
parser.add_argument("--min_event", type=int, help="Min_event for jet clustering")
parser.add_argument("--max_event", type=int, help="Max_event for jet clustering")
parser.add_argument("--sample", type=str, help="Sample processed, options are wjets or ttbar")
parser.add_argument("--ptcut", type=int, help="Applied cut on pt values", default=15)
parser.add_argument(
    "--is_abc", action="store_true", help="Perform ABC and PUPPI clustering (default: True)"
)
parser.add_argument(
    "--no-is_abc",
    dest="is_abc",
    action="store_false",
    help="Do not perform ABC and PUPPI clustering",
)
parser.set_defaults(is_abc=True)
parser.add_argument(
    "--is_return_all",
    action="store_true",
    help="Perform all clustering -> GNN, Puppi and DistillNet(default: True)",
)
parser.add_argument(
    "--no-is_return_all",
    dest="is_return_all",
    action="store_false",
    help="Do not perform all clustering",
)
parser.set_defaults(is_return_all=True)

# Step 5: Access and use the parsed arguments
# Step 4: Parse the command-line arguments
args = parser.parse_args()
min_event = args.min_event
max_event = args.max_event
is_abc_puppi = args.is_abc
is_return_all = args.is_return_all
Samplechoice = args.sample
ptcut = args.ptcut
print(f"min_event: {min_event}")
print(f"max_event: {max_event}")
print(f"is_abc_puppi: {is_abc_puppi}")
print(f"Sample: {Samplechoice}")
print(f"ptcut: {ptcut}")

savedir = join_and_makedir(dirs["savedir"], "Results_Jetclustering/")
filedir = dirs["filedir"]
modelsavedir = join_and_makedir(dirs["savedir"], "Models/")

w_sample = "distill_wjets_emd_prl.h5"
ttbar_sample = "distill_ttbar_emd_prl.h5"


if Samplechoice == "wjets":
    sample = w_sample
elif Samplechoice == "ttbar":
    sample = ttbar_sample
print(sample)

saveinfo = f"trainpart_{hparams['maketrain_particles']:.2E}__Batchs_{hparams['batch_size']}__numep_{trainparams['n_epochs']}__wgt_{trainparams['weightedlossval']}"
if bool_val["is_min_max_scaler"]:
    saveinfo += "_minmaxscaler"
if bool_val["is_standard_scaler"]:
    saveinfo += "_stdscaler"
print(saveinfo)

flist_inputs = [member.value for member in fl_inputs]

if is_abc_puppi:
    dir_saveinoabc = (
        f"Jetclustering_ptcut{ptcut}_ptmatch/Abc_puppi__" + sample.replace(".h5", "") + "/"
    )
    results_dir = os.path.join(savedir, dir_saveinoabc)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

elif is_return_all:
    dir_saveinoabc = (
        f"Jetclustering_ptcut{ptcut}_ptmatch/Allresults__"
        + saveinfo
        + "__"
        + sample.replace(".h5", "")
        + "/"
    )
    results_dir = os.path.join(savedir, dir_saveinoabc)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
else:
    dir_saveinfo = (
        f"Jetclustering_ptcut{ptcut}_ptmatch/" + saveinfo + "__" + sample.replace(".h5", "") + "/"
    )
    results_dir = os.path.join(savedir, dir_saveinfo)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

device = "cpu"
dR = 0.4  # Radius used for matching the gen and clustered jets

is_savefig = True
is_displayplots = False


input_size = 16
l1_hsize = hparams["L1_hsize"]
l2_hsize = hparams["L2_hsize"]
n_outputs = hparams["n_outputs"]

model = load_bestmodel(
    saveinfo,
    modelsavedir,
    "bestmodel_trainloss",
    device,
    input_size,
    hparams["L1_hsize"],
    hparams["L2_hsize"],
    hparams["n_outputs"],
)
scaler = makescaler(filedir, sample, flist_inputs, is_standard_scaler=True, is_min_max_scaler=False)
print("Scaler created")
numjetspuppi, numjetsabc, numjetsdistill, numjetsgen, numjetspuppi_test, numjetsabc_test = (
    0,
    0,
    0,
    0,
    0,
    0,
)
(
    e_valabc,  # Energy values of individual jets, test stands for pre-clustered jets that can be cross checked with the own clustered jets
    e_valdistil,
    e_valpuppi,
    e_valgen,
    e_valpuppi_test,
    e_valabc_test,
    resabc,  # Resolution of matched jets -> Clustered jets for which a generator-level jet could be found within R < 0.4
    resdistil,
    respuppi,
    respuppi_test,
    resabc_test,
    energy_abc_matched,  # Energy of matched jets, abc jet for which a generator level jet could be matched
    energy_gen_abcmatch,  # generator level jet, for which and abcjet match could be found
    energy_puppi_matched,
    energy_gen_puppimatch,
    energy_distill_matched,
    energy_gen_distillmatch,
    pt_abc_matched,  # pt of matched jets
    pt_gen_abcmatch,
    pt_puppi_matched,
    pt_gen_puppimatch,
    pt_distill_matched,
    pt_gen_distillmatch,
) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
if is_abc_puppi:
    print("Return ABCNet and Puppi clustering")
    for i in tqdm(range(min_event, max_event)):  # loop over desired number of events
        (
            jetsabc,
            jetspuppi,
            jetsgen,
            repabc,
            reppuppi,
            energy_abc_m,
            energy_gen_abcm,
            energy_puppi_m,
            energy_gen_puppim,
            pt_abc_m,
            pt_gen_abcm,
            pt_puppi_m,
            pt_gen_puppim,
        ) = JetEvent(
            filedir,
            sample,
            i,
            flist_inputs,
            model,
            device,
            dR,
            ptcut,
            scaler,
            is_standard=True,
            is_remove_padding=True,
            is_abc_puppi=True,
        )

        e_valabc = np.append(e_valabc, jetsabc[1])
        e_valpuppi = np.append(e_valpuppi, jetspuppi[1])
        e_valgen = np.append(e_valgen, jetsgen[1])
        energy_abc_matched = np.append(energy_abc_matched, energy_abc_m)
        energy_gen_abcmatch = np.append(energy_gen_abcmatch, energy_gen_abcm)
        energy_puppi_matched = np.append(energy_puppi_matched, energy_puppi_m)
        energy_gen_puppimatch = np.append(energy_gen_puppimatch, energy_gen_puppim)

        pt_abc_matched = np.append(pt_abc_matched, pt_abc_m)
        pt_gen_abcmatch = np.append(pt_gen_abcmatch, pt_gen_abcm)
        pt_puppi_matched = np.append(pt_puppi_matched, pt_puppi_m)
        pt_gen_puppimatch = np.append(pt_gen_puppimatch, pt_gen_puppim)

        resabc = np.append(resabc, repabc)

        respuppi = np.append(respuppi, reppuppi)

        numjetspuppi += len(jetspuppi[1])
        numjetsabc += len(jetsabc[1])
        numjetsgen += len(jetsgen[1])
    print("numjetspuppi_total: ", numjetspuppi)
    print("numjetsabc_total: ", numjetsabc)
    print("numjetsgen_total: ", numjetsgen)

    print("responseabcres: ", resolution(resabc))
    print("puppijetres:", resolution(respuppi))

    rawdata = {
        "jetenergy_ABC": e_valabc,
        "jetenergy_puppi": e_valpuppi,
        "jetenergy_gen": e_valgen,
        "resabc": resabc,
        "respuppi": respuppi,
        "jetenergy_abc_matched": energy_abc_matched,
        "jetenergy_gen_abcmatched": energy_gen_abcmatch,
        "jetenergy_puppi_matched": energy_puppi_matched,
        "jetenergy_gen_puppimatched": energy_gen_puppimatch,
        "jetpt_abc_matched": pt_abc_matched,
        "jetpt_gen_abcmatched": pt_gen_abcmatch,
        "jetpt_puppi_matched": pt_puppi_matched,
        "jetpt_gen_puppimatched": pt_gen_puppimatch,
    }
    df = pd.DataFrame.from_dict(rawdata, orient="index")
    print(df)
    event_str = f"__min_event{min_event}_max_event{max_event}__"
    h5_save = results_dir + event_str + "abc_puppi__clusterresults.h5"
    df.to_hdf(h5_save, key="jetresults", mode="w")
    print("ABC_Puppi_DONEEEEEEEEEEEEEEEE")


elif is_return_all:
    print("RETURN All, ABCNet (Gnn), Puppi and DistillNet selected.")
    for i in tqdm(range(min_event, max_event)):  # loop over desired number of events
        (
            jetsabc,
            jetspuppi,
            jetsgen,
            repabc,
            reppuppi,
            energy_abc_m,
            energy_gen_abcm,
            energy_puppi_m,
            energy_gen_puppim,
            jetsdistil,
            jetsgen,
            repdistil,
            energy_distill_m,
            energy_gen_distillm,
            pt_abc_m,
            pt_gen_abcm,
            pt_puppi_m,
            pt_gen_puppim,
            pt_distill_m,
            pt_gen_distillm,
        ) = JetEvent(
            filedir,
            sample,
            i,
            flist_inputs,
            model,
            device,
            dR,
            ptcut,
            scaler,
            is_standard=True,
            is_remove_padding=True,
            is_abc_puppi=False,
            is_return_all=True,
        )

        e_valabc = np.append(e_valabc, jetsabc[1])
        e_valpuppi = np.append(e_valpuppi, jetspuppi[1])
        e_valgen = np.append(e_valgen, jetsgen[1])
        energy_abc_matched = np.append(energy_abc_matched, energy_abc_m)
        energy_gen_abcmatch = np.append(energy_gen_abcmatch, energy_gen_abcm)
        energy_puppi_matched = np.append(energy_puppi_matched, energy_puppi_m)
        energy_gen_puppimatch = np.append(energy_gen_puppimatch, energy_gen_puppim)

        pt_abc_matched = np.append(pt_abc_matched, pt_abc_m)
        pt_gen_abcmatch = np.append(pt_gen_abcmatch, pt_gen_abcm)
        pt_puppi_matched = np.append(pt_puppi_matched, pt_puppi_m)
        pt_gen_puppimatch = np.append(pt_gen_puppimatch, pt_gen_puppim)

        e_valdistil = np.append(e_valdistil, jetsdistil[1])
        e_valgen = np.append(e_valgen, jetsgen[1])
        energy_distill_matched = np.append(energy_distill_matched, energy_distill_m)
        energy_gen_distillmatch = np.append(energy_gen_distillmatch, energy_gen_distillm)

        pt_distill_matched = np.append(pt_distill_matched, pt_distill_m)
        pt_gen_distillmatch = np.append(pt_gen_distillmatch, pt_gen_distillm)

        resabc = np.append(resabc, repabc)
        respuppi = np.append(respuppi, reppuppi)
        resdistil = np.append(resdistil, repdistil)

        numjetspuppi += len(jetspuppi[1])
        numjetsabc += len(jetsabc[1])
        numjetsgen += len(jetsgen[1])
        numjetsdistill += len(jetsdistil[1])
        numjetsgen += len(jetsgen[1])
    print("numjetsdistil_total: ", numjetsdistill)
    print("numjetsgen_total: ", numjetsgen)
    print("numjetspuppi_total: ", numjetspuppi)
    print("numjetsabc_total: ", numjetsabc)
    print("numjetsgen_total: ", numjetsgen)

    print("responseabcres: ", resolution(resabc))
    print("puppijetres:", resolution(respuppi))
    print("Distiljetres:", resolution(resdistil))

    rawdata = {
        "jetenergy_ABC": e_valabc,
        "jetenergy_puppi": e_valpuppi,
        "jetenergy_gen": e_valgen,
        "resabc": resabc,
        "respuppi": respuppi,
        "jetenergy_abc_matched": energy_abc_matched,
        "jetenergy_gen_abcmatched": energy_gen_abcmatch,
        "jetenergy_puppi_matched": energy_puppi_matched,
        "jetenergy_gen_puppimatched": energy_gen_puppimatch,
        "jetenergy_distill": e_valdistil,
        "jetenergy_gen": e_valgen,
        "resdistil": resdistil,
        "energy_distill_matched": energy_distill_matched,
        "energy_gen_distillmatch": energy_gen_distillmatch,
        "jetpt_abc_matched": pt_abc_matched,
        "jetpt_gen_abcmatched": pt_gen_abcmatch,
        "jetpt_puppi_matched": pt_puppi_matched,
        "jetpt_gen_puppimatched": pt_gen_puppimatch,
        "jetpt_distill_matched": pt_distill_matched,
        "jetpt_gen_distillmatch": pt_gen_distillmatch,
    }
    df = pd.DataFrame.from_dict(rawdata, orient="index")
    print(df)
    event_str = f"__min_event{min_event}_max_event{max_event}__"
    h5_save = results_dir + saveinfo + event_str + "abc_puppi_distill_clusterresults.h5"
    df.to_hdf(h5_save, key="jetresults", mode="w")
    print("ALL Doneee")

else:
    print("Executing Distillclustering without ABCNet or Puppi.")
    for i in tqdm(range(min_event, max_event)):  # loop over desired number of events
        (
            jetsdistil,
            jetsgen,
            repdistil,
            energy_distill_m,
            energy_gen_distillm,
            pt_distill_m,
            pt_gen_distillm,
        ) = JetEvent(
            filedir,
            sample,
            i,
            flist_inputs,
            model,
            device,
            dR,
            ptcut,
            scaler,
            is_standard=True,
            is_remove_padding=True,
            is_abc_puppi=False,
        )

        e_valdistil = np.append(e_valdistil, jetsdistil[1])
        e_valgen = np.append(e_valgen, jetsgen[1])
        energy_distill_matched = np.append(energy_distill_matched, energy_distill_m)
        energy_gen_distillmatch = np.append(energy_gen_distillmatch, energy_gen_distillm)

        pt_distill_matched = np.append(pt_distill_matched, pt_distill_m)
        pt_gen_distillmatch = np.append(pt_gen_distillmatch, pt_gen_distillm)

        resdistil = np.append(resdistil, repdistil)
        numjetsdistill += len(jetsdistil[1])
        numjetsgen += len(jetsgen[1])
    print("numjetsdistil_total: ", numjetsdistill)
    print("numjetsgen_total: ", numjetsgen)

    print("Distiljetres:", resolution(resdistil))

    rawdata = {
        "jetenergy_distill": e_valdistil,
        "jetenergy_gen": e_valgen,
        "resdistil": resdistil,
        "energy_distill_matched": energy_distill_matched,
        "energy_gen_distillmatch": energy_gen_distillmatch,
        "jetpt_distill_matched": pt_distill_matched,
        "jetpt_gen_distillmatch": pt_gen_distillmatch,
    }
    df = pd.DataFrame.from_dict(rawdata, orient="index")
    print(df)
    event_str = f"__min_event{min_event}_max_event{max_event}__"
    h5_save = results_dir + saveinfo + event_str + "clusterresults.h5"
    df.to_hdf(h5_save, key="jetresults", mode="w")
    print("File saved")
