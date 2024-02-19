"""
Distillnet config dictionaries. Please add filedir and savedir for own use. The selected hyperparameters are default choices and are not opimized.
"""
hparams = {
    "batch_size": 256,  # Batch size
    "maketrain_particles": 8e6,  # Particles utlized for training
    'train_split': 0.75, #training-validation data split; train split is percentage used for training
    "L1_hsize": 128,  # Size Hidden Layer 1
    "L2_hsize": 64,  # Size Hidden Layer 2
    "n_outputs": 1,  # Number of output classes
    "lr": 0.0015,  # Learning rate
}

trainparams = {
    "n_epochs": 3,
    "patience": 5,
    "best_loss": 0.17,
    "weightedlossval": 3,  # Weight of Weighted MAE Loss
    "train_sample": "distill_wjets_swd.h5",  # sample for training distill_wjets_swd.h5
    "test_sample": "distill_ttbar_swd.h5",  # sample for testing distill_ttbar_swd.h5
    "bestmodel_losstype": "bestmodel_trainloss",  # choose whether to calculate physics results on bestmodel_trainloss or bestmodel_valloss
}


dirs = {"filedir": "/downloads/", "savedir": "./Results"}

bool_val = {
    "is_displayplots": False,  # works depending on GUI, recommended is leaving this False and examining the saved figures
    "is_savefig": True,  # Save figures in savedir
    "is_remove_padding": True,  # removal of zero-padded particles in events, recommended to leave True, otherwise results degrade
    "is_min_max_scaler": False,  # Use min max scaler for scaling input training data
    "is_standard_scaler": True,  # Use standard scaler for scaling input training data
    "is_dtrans": False,  # Trainsform Input data from variable d0 and dZ to be in Interval abs(d0 or dZ) < 1; to aid input normalization due to outliers
    "is_do_taylor": False,  # Do Taylor analysis on input features
    "is_weighted_error": True,  # Use weighted MAE Error in training
}
