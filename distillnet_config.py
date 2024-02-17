hparams = {'batch_size': 256,
           'maketrain_particles': 1.4e7,
           'L1_hsize': 128,
           'L2_hsize': 64,
           'n_outputs': 1,
           'lr': 0.0015,
           }

trainparams = {'n_epochs': 20,
               "patience": 5,
               "best_loss": 0.17,
               "train_sample": "distill_wjets_swd", #sample for training
               "test_sample": "distill_ttbar_swd",  #sample for testing
               }

bool_val = {}