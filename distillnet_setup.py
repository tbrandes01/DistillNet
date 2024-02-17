import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.utils.data as data
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
from distillnet_config import hparams, trainparams
from tayloranalysis.cls import TaylorAnalysis
import time


class WeightedMAE(nn.L1Loss):
    '''
    Class for computing Mean Absolute-Error loss with optional weights per prediction.
    For compatability with using basic PyTorch losses, weights are passed during initialisation rather than when computing the loss.

    Arguments:
        weight: sample weights as PyTorch Tensor, to be used with data to be passed when computing the loss

    Examples::
        >>> loss = WeightedMAE()
        >>>
        >>> loss = WeightedMAE(weights)
    '''

    def __init__(self, weight):
        super().__init__(reduction='mean' if weight is None else 'none')
        self.weights = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        '''
        Evaluate loss for given predictions

        Arguments:
            input: prediction tensor
            target: target tensor

        Returns:
            (weighted) loss
        '''

        if self.weights is not None:
            return torch.mean(self.weights * super().forward(input, target))
        else:
            return super().forward(input, target)


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


class DistillNet(
    nn.Module
):  # define Neural Net, feed forward net with 5 input nodes, 2 hidden layers and relu,sigmoid activation
    def __init__(self, input_size, hidden_size_l1, hidden_size_l2, num_classes):
        super(DistillNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size_l1)
        self.l2 = nn.Linear(hidden_size_l1, hidden_size_l2)
        self.bn2 = nn.BatchNorm1d(hidden_size_l2)
        self.l3 = nn.Linear(hidden_size_l2, num_classes)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.sig(out)
        return out
    

def load_bestmodel(saveinfo, savedir, modelsavedir: str, modelname: str, device: str,
                input_size: int, hidden_size_l1: int, hidden_size_l2: int, num_classes: int):
    model = DistillNet(input_size, hidden_size_l1, hidden_size_l2, num_classes)

    #print(saveinfo)
    #modelname = "bestmodel"
    #modelsavedir = os.path.join(savedir, "Models/")
    model.load_state_dict(torch.load(modelsavedir + modelname + saveinfo + '.pth', map_location=device), strict=True)
    model.to(device)
    return model



def nn_setup(data, device, batch_size, maketrain_particles, l1_hsize, l2_hsize, n_outputs):
    train_loader, test_loader, input_size, test, weights_highval = makedataloaders(data, batch_size, maketrain_particles)
    model = Net_drop_mod(input_size, l1_hsize, l2_hsize, n_outputs) #CHANGE
    model.to(device)
    criterion = nn.L1Loss()  #Maybe add reg term
    #criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    return model, criterion, optimizer, train_loader, test_loader, test, input_size, weights_highval


def calc_datasetweights(truth, is_makeprints: bool = False):
    count_between_0_and_0_05 = np.sum((truth >= 0) & (truth <= 0.05))
    count_between_0_95_and_1 = np.sum((truth >= 0.95) & (truth <= 1))
    total = count_between_0_and_0_05 + count_between_0_95_and_1
    weights_highval = total / count_between_0_95_and_1
    if is_makeprints:
        print(f"Values between 0 and 0.05: {count_between_0_and_0_05}")
        print(f"Values between 0.95 and 1: {count_between_0_95_and_1}")
        print(f"Total number of values for calc: {total}")
        print(f"Weight for high val: {weights_highval}")
    return weights_highval


def makedataloaders(dat: tuple, batch_size: int, num_particles: int):
    Particles, nfeatures = len(dat[0]), len(dat[0][0])
    num_particles = int(num_particles)
    # slice Dataset into train and validation, train containing 80% of the data
    numtrain = int(num_particles * 0.75)
    numvalid = int(num_particles * 0.25)  # Validation containing 20%
    test = (
        dat[0][numtrain: num_particles],
        dat[1][numtrain: num_particles],
    )  # slice validation dataset
    # slice train dataset again because we dont want to train over ALL 80 billion particles
    train = (dat[0][0:numtrain], dat[1][0:numtrain])
    print("Shape of train dataset: ", len(train[0]))
    print("Shape of test dataset: ", len(test[0]))

    # define train dataset as class object
    dataset_train = FeatureDataset(train)
    # define validation dataset as class object
    dataset_test = FeatureDataset(test)
    train_loader = data.DataLoader(
        dataset=dataset_train, shuffle=True, batch_size=batch_size)  # use pytorch dataloader to later iterate over train dataset

    test_loader = data.DataLoader(
        dataset=dataset_test, shuffle=False, batch_size=batch_size
    )  # use dataloader to iterate over validation dataset

    input_size = dataset_train.numfeatures()  # size of input vector
    weights_highval = calc_datasetweights(dat[1][0:num_particles], is_makeprints=True)
    weights_highval = 3

    return train_loader, test_loader, input_size, test, weights_highval


def validation(model, device, valid_loader, loss_function, weights_highval, is_weighted_error: bool = False):
    model.eval()
    loss_total = 0

    # Test validation data
    with torch.no_grad():
        for i, (features, labels) in enumerate(valid_loader):  # iterate over testloader
            features = features.to(device)
            labels = labels.to(device)

            if is_weighted_error:
                wloss_tensor = torch.ones_like(labels)
                #print(wloss_tensor)
                highval_mask = (labels >= 0.95) & (labels <= 1)
                wloss_tensor[highval_mask] *= weights_highval
                #print(wloss_tensor)
                loss_function = WeightedMAE(wloss_tensor)

            output = model.forward(features)  # calculate model output
            loss = loss_function(output, labels)  # calulate loss
            loss_total += loss.item()

    return loss_total / len(valid_loader)


def do_training(model, criterion, optimizer, device, train_loader, test_loader, test, savedir: str, modelsavedir: str, saveinfo: str,
                weights_highval: int, num_epochs: int, is_earlystopping: bool = True, is_dotaylor: bool = False, is_weighted_error: bool = False):
    num_epochs = int(num_epochs)
    #num_epochs = 40
    # training part
    examples = iter(train_loader)  #test if dataloader produces desired output
    samples = next(examples)
    inputsize = len(samples[0][0])
    print(inputsize)
    n_total_steps = len(train_loader)
    losslist = []
    validationloss = []
    if inputsize == 15:
        varlist = [0, 1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12, 13, 14]
        varnames = ["Eta", "Phi", "Logpt", "LogE", 'd0', 'dz', 'charge', 'pid 1', 'pid 2', 'pid 3', 'pid 4', 'pid 5', 'pid 6', 'pid 7', "pid 8"]
        print('taylor without puppi')
    if inputsize == 16:
        varlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15]
        varnames = ["Eta", "Phi", "Logpt", "LogE", 'd0', 'dz', 'puppiw', 'charge', 'pid 1', 'pid 2', 'pid 3', 'pid 4', 'pid 5', 'pid 6', 'pid 7', "pid 8"]
        print('taylor with puppi')
    if is_dotaylor:
        model = TaylorAnalysis(model)
        model.setup_tc_checkpoints(
            number_of_variables_in_data=inputsize,
            considered_variables_idx=varlist,
            variable_names=varnames,
            derivation_order=2,
    )

    # parameters for early stopping
    last_loss = 100
    best_loss_train = 5
    patience = trainparams['patience']
    triggertimes = 0
    valid = is_earlystopping
    best_loss = trainparams['best_loss']
    # training loop
    for epoch in tqdm(range(num_epochs)):
        _losslist = []
        train_loss = 0
        model.train()
        for i, (features, labels) in enumerate(train_loader):  # iterate over trainloader
            #  should get features and push them to device
            features = features.to(device)
            labels = labels.to(device)

            if is_weighted_error:
                wloss_tensor = torch.ones_like(labels)
                #print(wloss_tensor)
                highval_mask = (labels >= 0.95) & (labels <= 1)
                wloss_tensor[highval_mask] *= weights_highval
                #print(wloss_tensor)
                criterion = WeightedMAE(wloss_tensor)


            # Forward pass
            outputs = model.forward(features)  # calculate outputs
            loss = criterion.forward(outputs, labels)  # calculate loss
            _losslist.append(loss.to("cpu").detach().numpy())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # break

            if (i + 1) % 5000 == 0:  # some print statement on progress
                # /numsteps
                print(
                    f"Epoch: [{epoch+1}/{num_epochs}], Batch [{i+1}/{n_total_steps}], :Loss : {loss.item():.4f}"
                )

        # calculate training loss as average over all batches in one epoch
        losslist.append(np.mean(_losslist))
        train_loss = np.mean(_losslist)
        if train_loss < best_loss_train:
            best_loss_train = train_loss
            modelname = "bestmodel_trainloss"
            #modelsavedir = os.path.join(savedir, "Models/")
            if not os.path.isdir(modelsavedir):
                os.makedirs(modelsavedir)
            # print(modelsavedir)
            if is_dotaylor:
                torch.save(model.model.state_dict(), modelsavedir + modelname + saveinfo + '.pth')
            else:
                torch.save(model.state_dict(), modelsavedir + modelname + saveinfo + '.pth')
            print("best model at: ", epoch)
        print("Training loss per epoch: ", np.mean(_losslist))
        if valid:  # early stopping, if wanted
            current_loss = validation(model, device, test_loader, criterion, weights_highval, is_weighted_error)
            validationloss.append(current_loss)
            print("The Current Loss:", current_loss)
            if current_loss < best_loss:
                best_loss = current_loss
                modelname = "bestmodel_valloss"
                #modelsavedir = os.path.join(savedir, "Models_v2/")
                if not os.path.isdir(modelsavedir):
                    os.makedirs(modelsavedir)
                # print(modelsavedir)
                if is_dotaylor:
                    torch.save(model.model.state_dict(), modelsavedir + modelname + saveinfo + '.pth')
                else:
                    torch.save(model.state_dict(), modelsavedir + modelname + saveinfo + '.pth')
                print("best model at: ", epoch)
            # Early stopping
            if current_loss > last_loss:
                triggertimes += 1
                print("Trigger Times:", triggertimes)

                if triggertimes >= patience:
                    print("Early stopping!\nStart to test process.")
                    return model, losslist, validationloss

            else:
                print("trigger times: 0")
                triggertimes = 0

            last_loss = current_loss
        if is_dotaylor:
            model.tc_checkpoint(features, epoch=epoch)
            model._apply_abs = True
            x_test = torch.tensor(test[0][0:50000], dtype=torch.float).to(device)
        if is_dotaylor:
            model.plot_taylor_coefficients(
                x_test,
                considered_variables_idx=varlist,
                variable_names=varnames,

                derivation_order=2,
                path=[savedir + '/Taylor/' + saveinfo + "_coefficients.pdf"],
            )
            model.plot_checkpoints(path=[savedir + '/Taylor/checkpoints/' + saveinfo + "tc_training.pdf"])
    if is_dotaylor:
        return model.model, losslist, validationloss
    else:
        return model, losslist, validationloss


def modelpredictions(model, dataloader, batch_size: int, device: str):
    model.to(device)
    model.eval()
    with torch.no_grad():
        weight_prediction = []
        for i, (features, labels) in enumerate(dataloader):
            features = features.to(device)

            outputs = model.forward(features)
            outputs = outputs.view(batch_size)
            op = outputs.to("cpu").numpy()
    return op


def modelpredictions_complete(model, dataloader, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        weight_prediction = []
        for i, (features, labels) in enumerate(dataloader):
            features = features.to(device)

            _weight_prediction = model.forward(features)
            op = _weight_prediction.to("cpu").numpy()
            op = np.squeeze(op)
            weight_prediction.append(op)
    #print(weight_prediction)
    #weight_prediction = np.concatenate(weight_prediction)
    weight_prediction = np.array(weight_prediction, dtype='object')
    weight_prediction = np.ravel(weight_prediction)
   # print(weight_prediction)
    return weight_prediction