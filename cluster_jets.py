from jetclustering_helpers import JetEvent, resolution, make_jetenergyplot, make_jetresolutionplots, corinputs
from data_helpers import join_and_makedir
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import pandas as pd
from train_distillnet import filedir, w_sample, flist_inputs, zprime_sample, savedir
from distillnet_setup import load_bestmodel
from distillnet_config import hparams, trainparams
from tqdm import tqdm
import matplotlib
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
matplotlib.rc("font", size=20, family="serif")
matplotlib.rcParams["text.usetex"] = True

results_dir = join_and_makedir(savedir, "Jetclustering/")
plotdir = join_and_makedir(savedir, 'Plots/Jetclustering/')
plotdir_pdf = join_and_makedir(plotdir, 'pdf/')
modelsavedir = os.path.join(savedir, 'Models/')

niceValue = os.nice(18)
device = 'cpu'
dR = 0.4
ptcut = 10
sample = trainparams['test_sample']

is_savefig = True
is_displayplots = True

maketrain_part = int(1.4e7)
min_event = int(maketrain_part / 9000)
max_event = 12900
#max_event = 2500 - 1165

input_size = 16
l1_hsize = hparams["L1_hsize"]
l2_hsize = hparams["L2_hsize"]
n_outputs = hparams["n_outputs"]
saveinfo = '_trainpart_2.20E+07__Batchs_768__numep_44_7_3_bndrop005_werr3_std'
model = load_bestmodel(
    saveinfo,
    savedir,
    modelsavedir,
    'bestmodel_trainloss',
    device,
    input_size,
    hparams['L1_hsize'],
    hparams['L2_hsize'],
    hparams['n_outputs'],
)

numjetspuppi, numjetsabc, numjetsdistill, numjetsgen, numjetspuppi_test, numjetsabc_test = 0, 0, 0, 0, 0, 0
(
    e_valabc,
    e_valdistil,
    e_valpuppi,
    e_valgen,
    e_valpuppi_test,
    e_valabc_test,
    resabc,
    resdistil,
    respuppi,
    respuppi_test,
    resabc_test,
    energy_match_distill,
    energy_match_distillgen,
) = ([], [], [], [], [], [], [], [], [], [], [], [], [])
try:
    for i in tqdm(range(min_event, max_event)):  # loop over desired number of events
        (
    #     jetsabc,
            jetsdistil,
    #     jetspuppi,
            jetsgen,
        #   jetspuppi_true,
        #   jetsabc_test,
        #    repabc,
            repdistil,
            jetsdistil_match,
            jetsgen_matchd,
        #   reppuppi,
        #  reppuppi_test,
        #    respabc_test,
        ) = JetEvent(
            filedir,
            sample,
            i,
            flist_inputs,
            model,
            device,
            dR,
            ptcut=ptcut,
            is_standard=True,
            is_remove_padding=True,
        )

    #   e_valabc = np.append(e_valabc, jetsabc[1])
        e_valdistil = np.append(e_valdistil, jetsdistil[1])
    #   e_valpuppi = np.append(e_valpuppi, jetspuppi[1])
        e_valgen = np.append(e_valgen, jetsgen[1])
    # e_valpuppi_test = np.append(e_valpuppi_test, jetspuppi_true[1])
    # e_valabc_test = np.append(e_valabc_test, jetsabc_test[1])
        energy_match_distill = np.append(energy_match_distill, jetsdistil_match)
        energy_match_distillgen = np.append(energy_match_distillgen, jetsgen_matchd)
    #   resabc = np.append(resabc, repabc)
        resdistil = np.append(resdistil, repdistil)
    #   respuppi = np.append(respuppi, reppuppi)
    #  respuppi_test = np.append(respuppi_test, reppuppi_test)
    # resabc_test = np.append(resabc_test, respabc_test)

    #   numjetspuppi += len(jetspuppi[1])
    #   numjetsabc += len(jetsabc[1])
        numjetsdistill += len(jetsdistil[1])
        numjetsgen += len(jetsgen[1])
    #   numjetspuppi_test += len(jetspuppi_true[1])
    #  numjetsabc_test += len(jetsabc_test[1])

except KeyboardInterrupt:
    print("Keyboard interrupt detected. Exiting the loop.")

#print("numjetspuppi_total: ", numjetspuppi)
#print("numjetspuppitest_total: ", numjetspuppi_test)
#print("numjetsabc_total: ", numjetsabc)
#print("numjetsabctest_total: ", numjetsabc_test)
print("numjetsdistil_total: ", numjetsdistill)
print("numjetsgen_total: ", numjetsgen)

#print("responseabcres: ", resolution(resabc))
print("Distiljetres:", resolution(resdistil))
#print("puppijetres:", resolution(respuppi))
#print("puppijetres_test:", resolution(respuppi_test))
#print("abcjetres_test:", resolution(resabc_test))


rawdata = {
  #  "jetenergy_ABC": e_valabc,
    "jetenergy_distill": e_valdistil,
 #   "jetenergy_puppi": e_valpuppi,
    "jetenergy_gen": e_valgen,
   # "jetenergy_puppitest": e_valpuppi_test,
  #  "jetenergy_abctest": e_valabc_test,
 #   "resabc": resabc,
    "resdistil": resdistil,
    'jetmatch_distillgen': energy_match_distill,
    'jetmatch_gend': energy_match_distillgen,
 #   "respuppi": respuppi,
  #  "respuppi_true": respuppi_test,
 #   "resabc_true": resabc_test,
}
df = pd.DataFrame.from_dict(rawdata, orient="index")
print(df)
h5_save = results_dir + saveinfo + 'clusterresults_alldata.h5'
#h5_save = results_dir + 'abc_puppi_results_2e7trainpart__clusterresults.h5'
df.to_hdf(h5_save, key='jetresults', mode='w')


distiljetE = corinputs(df, 0)
genjetE = corinputs(df, 1)
responsedistil = corinputs(df, 2)

#make_jetenergyplot(distiljetE, genjetE, plotdir, plotdir_pdf, saveinfo, timestr, is_savefig=is_savefig, is_displayplots=is_displayplots)
#make_jetresolutionplots(responsedistil, plotdir, plotdir_pdf, saveinfo, timestr, is_savefig=is_savefig, is_displayplots=is_displayplots)
