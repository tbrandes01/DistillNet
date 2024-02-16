import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import torch.utils.data as data
import torch.nn as nn
from sklearn import preprocessing
from joblib import  load

"""
Helper functions for METEvent calculation :)
"""
class FeatureDataset(Dataset):  # create Dataset object for Dataloader to iterate over
    def __init__(self, data, transform=None, target_transform=None):
        # define traindata and truth labels
        alldata, labelNN = data[0], data[1]
        # transform feature_inputvector to torch tensor
        self.ftensor = torch.tensor(alldata).float()
        # transform truth_weight to torch tensor
        self.NNweight = torch.tensor(labelNN).float()

    # define iterator for dataloader, returns the inputvector and truth_value
    def __getitem__(self, index):
        return self.ftensor[index], self.NNweight[index]

    def __len__(self):
        return self.NNweight.size(0)

    def numfeatures(
        self,
    ):  # get length if input vector
        return len(self.ftensor[0])
def modelpredictions(model,dataloader,batch_size,device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        weight_prediction = []
        for i, (features,labels) in enumerate(dataloader):
            features = features.to(device)
           
            outputs = model.forward(features)
            outputs = outputs.view(batch_size)
            op = outputs.to("cpu").numpy()
    return op

def validation(model, device, valid_loader, loss_function):

    model.eval()
    loss_total = 0

    # Test validation data
    with torch.no_grad():
        for i, (features, labels) in enumerate(valid_loader):  # iterate over testloader
            features = features.to(device)
            labels = labels.to(device)

            output = model.forward(features)  # calculate model output
            loss = loss_function(output, labels)  # calulate loss
            loss_total += loss.item()

    return loss_total / len(valid_loader)

def gettraindata(filename, flist, standard: bool = True, remove_padding: bool = True):
    
    with h5py.File(filename, "r") as f:

        featurelist = f["data"][:, :, flist]  # get needed features from data
        # obtain number of events, particles and features from the list
        Events, Particles, nfeatures = featurelist.shape
        # reshape from (20000 Events*4000 Particles) to (80000000) as we are interested in a per particle view
        feature_allpart = featurelist.reshape((Events * Particles, nfeatures))

        # get puppi weights as another input for our neural net
        puppiw = f["puppi_w"][()]
        puppiw_allpart = puppiw.reshape(
            (Events * Particles, 1)
        )  # also reshape per particle

        # obtain truth labels which are the output weights of ABC Net
        NNw = f["DNN"][()]
        NNw_allpart = NNw.reshape((Events * Particles), 1)  # reshape per particle

        # combine features and puppi weights to the input vector for the net
        combinedfeatures = np.append(feature_allpart, puppiw_allpart, axis=1)

        if remove_padding:
            padmask_pt = np.abs(combinedfeatures[:, 2]) > 0.0001
            combinedfeatures, NNw_allpart = (
                combinedfeatures[padmask_pt],
                NNw_allpart[padmask_pt],
            )

        if standard:  # fit data via scipy to later apply standardization
            scaler = preprocessing.StandardScaler()
            scaler.fit(combinedfeatures)
          #  dump(
          #      scaler,
           #     f"removepadding{remove_padding}" + "std_scaler.bin",
            #    compress=True,
           # )
        # if standardization is not wanted still returns combinedfeatures_trans
        combinedfeatures_trans = combinedfeatures
        if standard:
            # apply standardization transformation to data
            combinedfeatures_trans = scaler.transform(combinedfeatures)

    # return input vector for net and truth label
    return combinedfeatures_trans, NNw_allpart


def resolution(arr,gen):
    q_75_abc = np.quantile(genfunc(arr,gen),0.75)
    q_25_abc = np.quantile(genfunc(arr,gen),0.25)
    resolutions = (q_75_abc-q_25_abc)/2
    return resolutions
def genfunc(arr,gen):
    return (np.array(arr)-np.array(gen))/np.array(gen)

def makevec(pt,phi): # create a vector using polar coordinates
    x=pt*np.cos(phi)
    y=pt*np.sin(phi)
    return x,y
def makevec2(pt,phi): # create a vector using polar coordinates
    x=pt*np.cos(phi)
    y=pt*np.sin(phi)
    return np.array([x,y])

def mag(vec): #get magnitude of vector
    return np.sqrt((np.sum(vec[0])**2)+(np.sum(vec[1])**2))
def cutfunc(pt_scaled,features:list,weights:list,npu:int,wcut:float,ptcut_central:float,ptcut_forward:float):
    mask_wcut = np.abs(weights) > wcut 
    w_wcut,theta_wcut,pt_wcut,phi_wcut = weights[mask_wcut], features[:,0][mask_wcut], pt_scaled[mask_wcut],features[:,1][mask_wcut]
   # theta_wcut,pt_wcut=theta,pt

    mask_ptcentral = np.abs(theta_wcut) <= 2.5   #identify central region
    mask_ptforward = np.abs(theta_wcut) > 2.5   #identify forward region
    pt_central=pt_wcut[mask_ptcentral]    #apply mask and only keep particles in regions
    pt_forward=pt_wcut[mask_ptforward]
    phi_central=phi_wcut[mask_ptcentral]
    phi_forward=phi_wcut[mask_ptforward]

    ptcut_central=ptcut_central + 0.007 * npu   #calculate cut according to paper
    ptcut_forward=ptcut_forward + 0.011 * npu    

    mask_ptcutc = np.abs(pt_central) > ptcut_central #apply cut, only keep particles which are greater that ptcut
    mask_ptcutf = np.abs(pt_forward) > ptcut_forward

    pt_c, pt_f, phi_c, phi_f = pt_central[mask_ptcutc],pt_forward[mask_ptcutf],phi_central[mask_ptcutc],phi_forward[mask_ptcutf] 
    ptvec_c,ptvec_f=makevec2(pt_c,phi_c),makevec2(pt_f,phi_f) #calculate vectors using phi 
    pt_vec_x=np.concatenate((ptvec_c[0],ptvec_f[0])) #create one px and one py vector 
    pt_vec_y=np.concatenate((ptvec_c[1],ptvec_f[1]))
    ET_missvec=[np.sum(pt_vec_x),np.sum(pt_vec_y)] #sum over px and py to get EtMET_vector
    ET_missvec_mag=mag(ET_missvec) #calculate magnitude of said vector
    return ET_missvec_mag
   
def Metcalc(features, weights, npu, wcut, ptcut_c, ptcut_f,cut : bool = True,):
    pt = np.exp(features[:,2])
    pt_scaled = pt * weights
    if cut:
        MEt=cutfunc(pt_scaled,features,weights,npu,wcut,ptcut_c,ptcut_f)
        
        return MEt
    ptvec= makevec(pt_scaled,features[:,1])
    #print(ptvec)
    #print(ptvec[0])
    #print(ptvec[1])
    ET_missvec=[np.sum(ptvec[0]),np.sum(ptvec[1])]
    ET_magnitude=mag(ET_missvec)
    
    return ET_magnitude
def MetEvent(Event, model,device,flist):
    filename = "/work/tbrandes/work/data/best_model_Summer20MiniAODv2_LR_decay_rate_0p5_noPUPPIalpha_x10_KDTree_ttbar_1.h5"
    
    with h5py.File(filename,"r")as f:
        featurelist = f['data'][Event, :, flist]
        Particles, nfeatures = featurelist.shape
      #  print(featurelist)
        feature_allpart = featurelist.reshape((Particles, nfeatures))
        puppiw = f['puppi_w'][Event]
        puppiw = np.array([puppiw])
       # print(feature_allpart)
       # print(puppiw)
        # obtain truth labels which are the output weights of ABC Net
        NNw = f['DNN'][Event]
        NNw_allpart = NNw # reshape per particle

        # combine features and puppi weights to the input vector for the net
        combinedfeatures = np.concatenate((feature_allpart, puppiw.T), axis=1)
       # print(combinedfeatures)
        #print("FEATURES",combinedfeatures)
        METpuppi = f['PUPPI_MET'][Event]
        METgen = f['genMET'][Event]
        npu =f['NPU'][Event]
        puppiw = f['puppi_w'][Event]
        remove_padding = True
        if remove_padding:
            padmask_pt = np.abs(combinedfeatures[:,2]) > 0.0001
          #  print(padmask_pt.size)
            combinedfeatures, NNw_allpart,puppiw = combinedfeatures[padmask_pt] , NNw_allpart[padmask_pt], puppiw[padmask_pt]

        
        #scaler = load("/work/tbrandes/work/NeuralNet/removepaddingTruestd_scaler.bin")
        scaler = load("/work/tbrandes/work/Delphes_samples/removepaddingTruestd_scaler_reducedfeatures.bin")

        
        combinedfeatures_trans = scaler.transform(combinedfeatures)
            
        alldata = (combinedfeatures_trans, NNw_allpart)  
        
        batch_size = len(combinedfeatures[:,2]) 
        dataset_test = FeatureDataset(alldata)
        test_loader = data.DataLoader(dataset=dataset_test, shuffle=False, batch_size=batch_size) 
        predictions = modelpredictions(model,test_loader,batch_size,device)
        cut=True
        METDISTIl=Metcalc(combinedfeatures,predictions,npu,0.1,0.1,0.2,cut=cut)    
        return METDISTIl,METgen
def MetEvent2(Event, model,device,flist):
    filename = "/storage/9/abal/ABCNet/best_model_Summer20MiniAODv2_LR_decay_rate_0p5_noPUPPIalpha_x10_KDTree_ttbar_2.h5"
    
    with h5py.File(filename,"r")as f:
        featurelist = f['data'][Event, :, flist]
        Particles, nfeatures = featurelist.shape
      #  print(featurelist)
        feature_allpart = featurelist.reshape((Particles, nfeatures))
        puppiw = f['puppi_w'][Event]
        puppiw = np.array([puppiw])
       # print(feature_allpart)
       # print(puppiw)
        # obtain truth labels which are the output weights of ABC Net
        NNw = f['DNN'][Event]
        NNw_allpart = NNw # reshape per particle

        # combine features and puppi weights to the input vector for the net
        combinedfeatures = np.concatenate((feature_allpart, puppiw.T), axis=1)
       # print(combinedfeatures)
        #print("FEATURES",combinedfeatures)
        METpuppi = f['PUPPI_MET'][Event]
        METgen = f['genMET'][Event]
        npu =f['NPU'][Event]
        puppiw = f['puppi_w'][Event]
        remove_padding = True
        if remove_padding:
            padmask_pt = np.abs(combinedfeatures[:,2]) > 0.0001
          #  print(padmask_pt.size)
            combinedfeatures, NNw_allpart,puppiw = combinedfeatures[padmask_pt] , NNw_allpart[padmask_pt], puppiw[padmask_pt]

        #scaler = load("/work/tbrandes/work/NeuralNet/removepaddingTruestd_scaler.bin")
        scaler = load("/work/tbrandes/work/Delphes_samples/removepaddingTruestd_scaler_reducedfeatures.bin")
        combinedfeatures_trans = scaler.transform(combinedfeatures)
            
        alldata = (combinedfeatures_trans, NNw_allpart)  
        
        batch_size = len(combinedfeatures[:,2]) 
        dataset_test = FeatureDataset(alldata)
        test_loader = data.DataLoader(dataset=dataset_test, shuffle=False, batch_size=batch_size) 
        predictions = modelpredictions(model,test_loader,batch_size,device)
        cut=True
        METDISTIl=Metcalc(combinedfeatures,predictions,npu,0.1,0.1,0.2,cut=cut)    
        return METDISTIl,METgen


