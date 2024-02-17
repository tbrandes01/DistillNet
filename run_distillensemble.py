import torch
from tqdm import tqdm
import numpy as np
from train_distillnet_ens_v2 import maketraining_distill
from tap import Tap
import os
from distillnet_config import hparams, trainparams, dirs


class Argparser(Tap):
    cuda: int
    savedir: str
    numtests: int
    wmin: int
    wmax: int
    step: float


def getresolutions(device, args):
    resolutions_all = []
    wlist = np.arange(args.wmin, args.wmax, args.step)
    for wgt in wlist:
        resolutions = []
        for i in tqdm(range(args.numtests)):
            res = maketraining_distill(dirs['filedir'], args.savedir, device, i, args.numtests, wgt, is_ensembletest=True)
            resolutions.append(res)
            print(f"resolution at model{i}: {res}")
        resolutions_all.append(resolutions)
    return resolutions


def makeprints(array, args):
    print("Results for DistillNet ensemble")
    for wgtlist, idx in enumerate(array):
        wlist = np.arange(args.wmin, args.wmax, args.step)
        print('Weight used in Loss: ', wlist[idx])
        print('Achieved Resolution:' , wgtlist)
        print(f"best resolution at model number: {np.argmin(wgtlist)} with {wgtlist[np.argmin(wgtlist)]}")
        print("mean resolution:", np.mean(wgtlist))
        print(f"max resolution at model number: {np.argmax(wgtlist)} with {wgtlist[np.argmax(wgtlist)]}")


def main():
    args = Argparser().parse_args()
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    print(device)
    resolutions = getresolutions(device, args)
    makeprints(resolutions, results_dir, args)


if __name__ == "__main__":
    main()