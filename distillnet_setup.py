"""
This script contains the functions needed for setting up DistillNet, e.g.the pytorch Distillnet and dataset/dataloader classes.
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.utils.data as data
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from distillnet_config import hparams, trainparams
from tayloranalysis.cls import TaylorAnalysis
from typing import Union, List, Tuple, Dict, Any


class WeightedMAE(nn.L1Loss):
    """
    Class for computing Mean Absolute-Error loss with optional weights per prediction.
    For compatability with using basic PyTorch losses, weights are passed during initialisation rather than when computing the loss.

    Arguments:
        weight: sample weights as PyTorch Tensor, to be used with data to be passed when computing the loss

    Examples::
        >>> loss = WeightedMAE()
        >>>
        >>> loss = WeightedMAE(weights)
    """

    def __init__(self, weight):
        super().__init__(reduction="mean" if weight is None else "none")
        self.weights = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Evaluate loss for given predictions

        Arguments:
            input: prediction tensor
            target: target tensor

        Returns:
            (weighted) loss
        """

        if self.weights is not None:
            return torch.mean(self.weights * super().forward(input, target))
        else:
            return super().forward(input, target)


class FeatureDataset(Dataset):  # create Dataset object for Dataloader to iterate over
    """
    DistillNet pytorch training dataset for later dataloader instance. Contains DistillNet per-particle features as
    input vector as well as the GNN's per-particle soft targets as truth vector.
    """

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


class DistillNet(nn.Module):
    """
    Pytorch DistillNet Neural Network class instance.
    """

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


def load_bestmodel(
    saveinfo: str,
    modelsavedir: str,
    modelname: str,
    device: str,
    input_size: int,
    hidden_size_l1: int,
    hidden_size_l2: int,
    num_classes: int,
) -> Union[DistillNet, nn.Module]:
    """
    Load best model from saved models directory, either best training or best validation loss.
    """
    model = DistillNet(input_size, hidden_size_l1, hidden_size_l2, num_classes)
    model.load_state_dict(
        torch.load(modelsavedir + modelname + saveinfo + ".pth", map_location=device), strict=True
    )
    model.to(device)
    return model


def nn_setup(
    data: tuple[list, list, int, int],
    device: str,
    batch_size: int,
    maketrain_particles: int,
    trainsplit: float,
    l1_hsize: int,
    l2_hsize: int,
    n_outputs: int,
) -> tuple[DistillNet, Any, torch.optim.Optimizer, data.DataLoader, data.DataLoader, list, int]:
    """
    Setup Distillnet for training, initialize dataloaders, model, loss and optimizer.

    Args:
        data (tuple[list, list, int, int]): A tuple containing the training and testing data, input size, and test flag.
        device (str): The device to use for training (e.g., 'cpu', 'cuda').
        batch_size (int): The batch size for training.
        maketrain_particles (int): The number of particles to use for training.
        trainsplit (float): The ratio of training data to total data.
        l1_hsize (int): The size of the first hidden layer in the model.
        l2_hsize (int): The size of the second hidden layer in the model.
        n_outputs (int): The number of output units in the model.

    Returns:
        tuple: A tuple containing the model, criterion, optimizer, train_loader, test_loader, test, and input_size.
    """
    train_loader, test_loader, input_size, test = makedataloaders(
        data, trainsplit, batch_size, maketrain_particles
    )
    model = DistillNet(input_size, l1_hsize, l2_hsize, n_outputs)
    model.to(device)
    criterion = nn.L1Loss()  # placeholder criterion for later modified weighted MAE Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    print("Model hyperparams ", hparams)
    print("Model trainparams ", trainparams)
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    return model, criterion, optimizer, train_loader, test_loader, test, input_size


def calc_datasetweights(truth: list, is_makeprints: bool = False) -> float:
    """
    Calculate weights for class imbalance.

    Args:
        truth (list): A list of truth values.
        is_makeprints (bool, optional): Flag to indicate whether to print intermediate values. Defaults to False.

    Returns:
        float: The weight for high values.

    """
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


def makedataloaders(
    dat: Tuple[list, list, int, int], trainsplit: float, batch_size: int, num_particles: int
) -> Tuple[data.DataLoader, data.DataLoader, int, list]:
    """
    Create train and test data loaders for a given dataset.

    Args:
        dat (Tuple[list, list, int, int]): The input dataset, consisting of features and labels.
        trainsplit (float): The ratio of data to be used for training.
        batch_size (int): The batch size for the data loaders.
        num_particles (int): The total number of particles in the dataset.

    Returns:
        Tuple[data.DataLoader, data.DataLoader, int, list]: A tuple containing the train data loader,
        test data loader, input size, and test dataset.

    """
    Particles, nfeatures = len(dat[0]), len(dat[0][0])
    num_particles = int(num_particles)
    # slice Dataset into train and validation, train containing 80% of the data
    numtrain = int(num_particles * trainsplit)
    test = (
        dat[0][numtrain:num_particles],
        dat[1][numtrain:num_particles],
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
        dataset=dataset_train, shuffle=True, batch_size=batch_size
    )  # use pytorch dataloader to later iterate over train dataset

    test_loader = data.DataLoader(
        dataset=dataset_test, shuffle=False, batch_size=batch_size
    )  # use dataloader to iterate over validation dataset

    input_size = dataset_train.numfeatures()  # size of input vector
    weights_highval = calc_datasetweights(dat[1][0:num_particles], is_makeprints=False)

    return train_loader, test_loader, input_size, test


def validation(
    model: Union[DistillNet, nn.Module],
    device: str,
    valid_loader,
    loss_function,
    weights_highval: int,
    is_weighted_error: bool = False,
):
    """
    Calculate validation loss per epoch for model.

    Args:
        model (Union[DistillNet, nn.Module]): The model to evaluate.
        device (str): The device to run the evaluation on.
        valid_loader: The data loader for the validation dataset.
        loss_function: The loss function to calculate the loss.
        weights_highval (int): The weight to assign to high-value labels.
        is_weighted_error (bool, optional): Whether to use weighted error calculation. Defaults to False.

    Returns:
        float: The average validation loss per epoch.
    """
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for i, (features, labels) in enumerate(valid_loader):  # iterate over testloader
            features = features.to(device)
            labels = labels.to(device)

            if is_weighted_error:
                wloss_tensor = torch.ones_like(labels)
                highval_mask = (labels >= 0.95) & (labels <= 1)
                wloss_tensor[highval_mask] *= weights_highval
                loss_function = WeightedMAE(wloss_tensor)

            output = model.forward(features)  # calculate model output
            loss = loss_function(output, labels)  # calulate loss
            loss_total += loss.item()

    return loss_total / len(valid_loader)


def do_training(
    model: Union[DistillNet, nn.Module],
    criterion,
    optimizer,
    device: str,
    train_loader,
    test_loader,
    test: list,
    savedir: str,
    modelsavedir: str,
    saveinfo: str,
    weights_highval: float,
    num_epochs: int,
    is_earlystopping: bool = True,
    is_dotaylor: bool = False,
    is_weighted_error: bool = False,
) -> Tuple[Union[DistillNet, nn.Module], List[float], List[float]]:
    """
    Trains the DistillNet model.

    Args:
        model (Union[DistillNet, nn.Module]): The model to be trained.
        criterion: The loss function.
        optimizer: The optimizer for updating the model's parameters.
        device (str): The device to be used for training (e.g., 'cpu', 'cuda').
        train_loader: The data loader for the training dataset.
        test_loader: The data loader for the test dataset.
        test (list): The test dataset.
        savedir (str): The directory to save the training results.
        modelsavedir (str): The directory to save the trained model.
        saveinfo (str): Additional information to be included in the saved model's filename.
        weights_highval (float): The weight for high-value samples in the loss calculation.
        num_epochs (int): The number of training epochs.
        is_earlystopping (bool, optional): Whether to use early stopping based on validation loss. Defaults to True.
        is_dotaylor (bool, optional): Whether to perform Taylor analysis. Defaults to False.
        is_weighted_error (bool, optional): Whether to use weighted MAE as the loss function. Defaults to False.

    Returns:
        Tuple: A tuple containing the trained model, training loss list, and validation loss list.
    """
    num_epochs = int(num_epochs)
    examples = iter(train_loader)  # test if dataloader produces desired output
    samples = next(examples)
    inputsize = len(samples[0][0])
    print(inputsize)
    n_total_steps = len(train_loader)
    losslist = []
    validationloss = []
    if inputsize == 15:  # In case Taylor analysis is to be done without puppiweights as input
        varlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        varnames = [
            "Eta",
            "Phi",
            "Logpt",
            "LogE",
            "d0",
            "dz",
            "charge",
            "pid 1",
            "pid 2",
            "pid 3",
            "pid 4",
            "pid 5",
            "pid 6",
            "pid 7",
            "pid 8",
        ]
        print("taylor without puppi")
    if inputsize == 16:  # Taylor analysis with puppiweights as input
        varlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        varnames = [
            "Eta",
            "Phi",
            "Logpt",
            "LogE",
            "d0",
            "dz",
            "puppiw",
            "charge",
            "pid 1",
            "pid 2",
            "pid 3",
            "pid 4",
            "pid 5",
            "pid 6",
            "pid 7",
            "pid 8",
        ]
        print("taylor with puppi")
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
    patience = trainparams["patience"]
    triggertimes = 0
    valid = is_earlystopping
    best_loss = trainparams["best_loss"]
    # training loop
    for epoch in tqdm(range(num_epochs)):
        _losslist = []
        train_loss = 0
        model.train()
        for i, (features, labels) in enumerate(train_loader):  # iterate over trainloader
            features = features.to(device)
            labels = labels.to(device)

            if is_weighted_error:
                wloss_tensor = torch.ones_like(labels)
                highval_mask = (labels >= 0.95) & (labels <= 1)
                wloss_tensor[highval_mask] *= weights_highval
                criterion = WeightedMAE(wloss_tensor)

            outputs = model.forward(features)  # calculate outputs
            loss = criterion.forward(outputs, labels)  # calculate loss
            _losslist.append(loss.to("cpu").detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 5000 == 0:
                print(
                    f"Epoch: [{epoch+1}/{num_epochs}], Batch [{i+1}/{n_total_steps}], :Loss : {loss.item():.4f}"
                )

        # calculate training loss as average over all batches in one epoch
        losslist.append(np.mean(_losslist))
        train_loss = np.mean(_losslist)
        if train_loss < best_loss_train:
            best_loss_train = train_loss
            modelname = "bestmodel_trainloss"
            if is_dotaylor:
                torch.save(model.model.state_dict(), modelsavedir + modelname + saveinfo + ".pth")
            else:
                torch.save(model.state_dict(), modelsavedir + modelname + saveinfo + ".pth")
            print("best training model at epoch: ", epoch)
        print("Training loss per epoch: ", np.mean(_losslist))
        if valid:  # validation
            current_loss = validation(
                model, device, test_loader, criterion, weights_highval, is_weighted_error
            )
            validationloss.append(current_loss)
            print("The current validation loss:", current_loss)
            if current_loss < best_loss:
                best_loss = current_loss
                modelname = "bestmodel_valloss"
                if is_dotaylor:
                    torch.save(
                        model.model.state_dict(), modelsavedir + modelname + saveinfo + ".pth"
                    )
                else:
                    torch.save(model.state_dict(), modelsavedir + modelname + saveinfo + ".pth")
                print("best validation model at epoch: ", epoch)
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
                path=[savedir + "/Taylor/" + saveinfo + "_coefficients.pdf"],
            )
            model.plot_checkpoints(
                path=[savedir + "/Taylor/checkpoints/" + saveinfo + "tc_training.pdf"]
            )
    if is_dotaylor:
        return model.model, losslist, validationloss
    else:
        return model, losslist, validationloss


def modelpredictions(model: Union[DistillNet, nn.Module], dataloader, batch_size: int, device: str) -> np.ndarray:
    """
    Make DistillNet weight predictions for input dataloader.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (features, labels) in enumerate(dataloader):
            features = features.to(device)
            outputs = model.forward(features)
            outputs = outputs.view(batch_size)
            op = outputs.to("cpu").numpy()
    return op


def modelpredictions_complete(model: Union[DistillNet, nn.Module], dataloader, device: str) -> np.ndarray:
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
    weight_prediction = np.array(weight_prediction, dtype="object")
    weight_prediction = np.ravel(weight_prediction)
    return weight_prediction
