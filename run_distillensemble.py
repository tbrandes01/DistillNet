import torch
from tqdm import tqdm
import numpy as np
from train_distillnet_ens import maketraining_distill
from tap import Tap
import os
from distillnet_config import hparams, trainparams


class Argparser(Tap):
    cuda: int
    numtests: int
    wmin: int
    wmax: int
    step: float


def makedirs():
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, "resolutions/")
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    return script_dir, results_dir


def getresolutions(device, args):
    resolutions_all = []
    wlist = np.arange(args.wmin, args.wmax, args.step)
    for wgt in wlist:
        resolutions = []
        for i in tqdm(range(args.numtests)):

            res = maketraining_distill(device, i, args.numtests, wgt)

            resolutions.append(res)
            print(f"resolution at model{i}: {res}")
        resolutions_all.append(resolutions)
    return resolutions


def makeprints(array, results_dir, args):
    strinfo = f"resolutions_fromdevice{args.cuda}_numtests{args.numtests}__tpart_{hparams['maketrain_particles']:.2E}__Batchs_{hparams['batch_size']}__numep_{trainparams['n_epochs']}_7_3_bndrop005_werr3"
    print("Results for distillation ensemble")
    np.savetxt(results_dir + strinfo + "csv", array, delimiter=";")

    print(array)
    print(f"best resolution at model number: {np.argmin(array)} with {array[np.argmin(array)]}")
    print("mean resolution:", np.mean(array))
    print(f"max resolution at model number: {np.argmax(array)} with {array[np.argmax(array)]}")


def main():
    script_dir, results_dir = makedirs()
    args = Argparser().parse_args()

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    print(device)
    resolutions = getresolutions(device, args)
    makeprints(resolutions, results_dir, args)


if __name__ == "__main__":
    main()