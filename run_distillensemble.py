"""
Control script for running DistillNet ensemble test.
"""

import torch
from tqdm import tqdm
import numpy as np
from train_distillnet_ens_v2 import do_training_and_physicstest_DistillNet
from tap import Tap
from distillnet_config import dirs


class Argparser(Tap):
    """
    Arguments: Specifiy cuda number for which gpu you wish to use, specify savedir for directory to save ensembles in, 
    numtests for the number of trainings to be executed,
    wmin as minimum weighted loss multiplicity value, wmax as maximum weighted loss multiplicity value, 
    step for the number of steps between wmin and wmax,
    """

    cuda: int = 0
    savedir: str = "./Ensembletest"
    numtests: int = 5
    wmin: int = 2
    wmax: int = 4
    step: float = 6


def getresolutions(device, args):
    """
    Runs ensemble test based on input args and return list of DistillNet MET resolutions.
    """
    resolutions_all = []
    wlist = np.arange(args.wmin, args.wmax, args.step)
    for wgt in wlist:
        resolutions = []
        for i in tqdm(range(args.numtests)):
            res = do_training_and_physicstest_DistillNet(
                dirs["filedir"], args.savedir, device, i, args.numtests, wgt, is_ensembletest=True
            )
            resolutions.append(res)
            print(f"resolution at model{i}: {res}")
        resolutions_all.append(resolutions)
    return resolutions


def makeprints(array, args):
    print("Results for DistillNet ensemble")
    for wgtlist, idx in enumerate(array):
        wlist = np.arange(args.wmin, args.wmax, args.step)
        print("Weight used in Loss: ", wlist[idx])
        print("Achieved Resolution:", wgtlist)
        print(
            f"best resolution at model number: {np.argmin(wgtlist)} with {wgtlist[np.argmin(wgtlist)]}"
        )
        print("mean resolution:", np.mean(wgtlist))
        print(
            f"max resolution at model number: {np.argmax(wgtlist)} with {wgtlist[np.argmax(wgtlist)]}"
        )


def main():
    args = Argparser().parse_args()
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    print(device)
    resolutions = getresolutions(device, args)
    makeprints(resolutions, args.savedir, args)


if __name__ == "__main__":
    main()
